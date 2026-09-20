"""Model weights and per-workload CUDA storage; no prompt content is reused."""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from kernels.rmsnorm import rms_norm
from kernels.pointwise import add_rms_norm, swiglu, prepare_qkv
from kernels.attention import decode_attention
from kernels.linear import choose_projections, project


class Layer:
    def __init__(self, layer):
        attn, mlp = layer.self_attn, layer.mlp
        self.qkv = torch.cat((attn.q_proj.weight, attn.k_proj.weight,
                              attn.v_proj.weight), dim=0)
        self.gate_up = torch.cat((mlp.gate_proj.weight, mlp.up_proj.weight), dim=0)
        self.out = attn.o_proj.weight
        self.down = mlp.down_proj.weight
        self.input_norm = layer.input_layernorm.weight
        self.post_norm = layer.post_attention_layernorm.weight
        self.q_norm = attn.q_norm.weight
        self.k_norm = attn.k_norm.weight


class Model:
    def __init__(self, reference):
        config = reference.config
        self.nq = config.num_attention_heads
        self.nkv = config.num_key_value_heads
        self.dim = reference.model.layers[0].self_attn.head_dim
        self.hidden = config.hidden_size
        self.eps = config.rms_norm_eps
        self.embedding = reference.model.embed_tokens.weight
        self.lm_head = reference.lm_head.weight
        self.norm = reference.model.norm.weight
        self.rope = reference.model.rotary_emb
        self.layers = [Layer(layer) for layer in reference.model.layers]
        self.projection_plans = {}
        self.packed_weights = {}

    def forward(self, ids, state, prefill=False, all_logits=False):
        batch, length = ids.shape
        plans = {} if prefill else state.projections
        x = F.embedding(ids, self.embedding)
        residual = None
        for i, layer in enumerate(self.layers):
            if residual is None:
                residual = x
                normed = rms_norm(x, layer.input_norm, self.eps)
            else:
                residual, normed = add_rms_norm(x, residual, layer.input_norm, self.eps)
            qkv = project(normed, layer.qkv, plans.get("qkv"))
            q = prepare_qkv(
                qkv, layer.q_norm, layer.k_norm, state.cos, state.sin,
                state.position, state.keys[i], state.values[i],
                self.nq, self.nkv, self.dim, self.eps,
            )
            if prefill:
                # Only initialized prompt slots are visible. Flash SDPA's GQA
                # avoids copying the eight KV heads into 32 physical heads.
                last_layer = i == len(self.layers) - 1
                if last_layer:
                    # All prompt K/V entries are needed by future decode steps,
                    # but only the last query's output reaches the logits.
                    q = q[:, :, -1:, :]
                    residual = residual[:, -1:, :].contiguous()
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    attn = F.scaled_dot_product_attention(
                        q, state.keys[i][:, :, :length],
                        state.values[i][:, :, :length],
                        # This query is at the END of the prompt and sees all
                        # its keys. A one-row causal mask would expose only K[0].
                        is_causal=not last_layer, enable_gqa=True,
                    )
                attn = attn.transpose(1, 2).reshape(batch, 1 if last_layer else length, -1)
            else:
                attn = decode_attention(
                    q, state.keys[i], state.values[i], state.position,
                    state.partial, state.lse, state.attention_block,
                )
                attn = attn.view(batch, 1, -1) if length == 1 else attn.transpose(1, 2).reshape(batch, length, -1)
            projected = project(attn, layer.out, plans.get("out"))
            residual, normed = add_rms_norm(projected, residual, layer.post_norm, self.eps)
            gate_up = project(normed, layer.gate_up, plans.get("gate_up"))
            x = project(swiglu(gate_up), layer.down, plans.get("down"))
        # Only the last prompt position needs final normalization and logits.
        if all_logits:
            _, normed = add_rms_norm(x, residual, self.norm, self.eps)
            return project(normed, self.lm_head, state.projections.get("lm_head"))
        residual, normed = add_rms_norm(
            x[:, -1:, :].contiguous(), residual[:, -1:, :].contiguous(),
            self.norm, self.eps,
        )
        return project(normed[:, 0], self.lm_head, state.projections.get("lm_head"))


class Generation:
    def __init__(self, model, shape, reserve=0):
        self.model, self.shape = model, shape
        batch, prompt, output = shape
        if batch not in model.projection_plans:
            model.projection_plans[batch] = choose_projections(model, batch)
        self.projections = model.projection_plans[batch]
        self.capacity = prompt + output + reserve
        device = model.embedding.device
        self.position = torch.zeros((), dtype=torch.int32, device=device)
        self.tokens = torch.zeros((batch, 1), dtype=torch.int64, device=device)
        self.prompt = torch.zeros((batch, prompt), dtype=torch.int64, device=device)
        cache_shape = (batch, model.nkv, self.capacity, model.dim)
        self.keys = [torch.empty(cache_shape, dtype=torch.bfloat16, device=device)
                     for _ in model.layers]
        self.values = [torch.empty_like(key) for key in self.keys]
        # Build tables with the pinned Transformers rotary implementation.
        positions = torch.arange(self.capacity, device=device).unsqueeze(0)
        self.cos, self.sin = (t.squeeze(0).contiguous() for t in model.rope(
            model.embedding, positions))
        # More splits keep small batches parallel; larger batches amortize
        # tile setup with longer blocks. Fixed for the entire workload.
        self.attention_block = 64 if batch <= 2 else 256
        splits = (self.capacity + self.attention_block - 1) // self.attention_block
        self.partial = torch.empty((batch, model.nq, splits, model.dim),
                                   dtype=torch.float32, device=device)
        self.lse = torch.empty((batch, model.nq, splits), dtype=torch.float32, device=device)
        self.graph = None
        if output > 1:
            self.capture()
        self.capture_prefill()

    def step(self):
        logits = self.model.forward(self.tokens, self)
        torch.argmax(logits, dim=-1, keepdim=True, out=self.tokens)
        self.position.add_(1)

    def capture(self):
        # Compilation and cuBLAS initialization finish outside capture.
        # Position zero reads only the slot written by prepare_qkv.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.position.zero_()
                self.step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.position.zero_()
        stream.wait_stream(torch.cuda.current_stream())
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.step()
        torch.cuda.current_stream().wait_stream(stream)

    def prefill_step(self):
        self.position.zero_()
        logits = self.model.forward(self.prompt, self, prefill=True)
        torch.argmax(logits, dim=-1, keepdim=True, out=self.tokens)
        # Prompt slots were overwritten; later slots remain masked until used.
        self.position.fill_(self.shape[1])

    def capture_prefill(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.prefill_step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        stream.wait_stream(torch.cuda.current_stream())
        self.prefill_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.prefill_graph, stream=stream):
            self.prefill_step()
        torch.cuda.current_stream().wait_stream(stream)

    def prefill(self, input_ids):
        self.prompt.copy_(torch.tensor(input_ids, dtype=torch.int64, device="cpu"))
        self.prefill_graph.replay()
