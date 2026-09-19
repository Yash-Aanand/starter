"""Model weights and per-workload CUDA storage; no prompt content is reused."""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from kernels.rmsnorm import rms_norm
from kernels.pointwise import add_rms_norm, swiglu, prepare_qkv
from kernels.attention import decode_attention


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

    def forward(self, ids, state, prefill=False):
        batch, length = ids.shape
        x = F.embedding(ids, self.embedding)
        residual = None
        for i, layer in enumerate(self.layers):
            if residual is None:
                residual = x
                normed = rms_norm(x, layer.input_norm, self.eps)
            else:
                residual, normed = add_rms_norm(x, residual, layer.input_norm, self.eps)
            qkv = F.linear(normed, layer.qkv)
            q = prepare_qkv(
                qkv, layer.q_norm, layer.k_norm, state.cos, state.sin,
                state.position, state.keys[i], state.values[i],
                self.nq, self.nkv, self.dim, self.eps,
            )
            if prefill:
                # Only initialized prompt slots are visible. Flash SDPA's GQA
                # avoids copying the eight KV heads into 32 physical heads.
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    attn = F.scaled_dot_product_attention(
                        q, state.keys[i][:, :, :length],
                        state.values[i][:, :, :length],
                        is_causal=True, enable_gqa=True,
                    )
                attn = attn.transpose(1, 2).reshape(batch, length, -1)
            else:
                attn = decode_attention(
                    q, state.keys[i], state.values[i], state.position,
                    state.partial, state.lse,
                ).view(batch, 1, -1)
            projected = F.linear(attn, layer.out)
            residual, normed = add_rms_norm(projected, residual, layer.post_norm, self.eps)
            x = F.linear(swiglu(F.linear(normed, layer.gate_up)), layer.down)
        # Only the last prompt position needs final normalization and logits.
        residual, normed = add_rms_norm(
            x[:, -1:, :].contiguous(), residual[:, -1:, :].contiguous(),
            self.norm, self.eps,
        )
        return F.linear(normed[:, 0], self.lm_head)


class Generation:
    def __init__(self, model, shape):
        self.model, self.shape = model, shape
        batch, prompt, output = shape
        self.capacity = prompt + output
        device = model.embedding.device
        self.position = torch.zeros((), dtype=torch.int32, device=device)
        self.tokens = torch.zeros((batch, 1), dtype=torch.int64, device=device)
        self.prompt = torch.empty((batch, prompt), dtype=torch.int64, device=device)
        cache_shape = (batch, model.nkv, self.capacity, model.dim)
        self.keys = [torch.empty(cache_shape, dtype=torch.bfloat16, device=device)
                     for _ in model.layers]
        self.values = [torch.empty_like(key) for key in self.keys]
        # Build tables with the pinned Transformers rotary implementation.
        positions = torch.arange(self.capacity, device=device).unsqueeze(0)
        self.cos, self.sin = (t.squeeze(0).contiguous() for t in model.rope(
            model.embedding, positions))
        splits = (self.capacity + 255) // 256
        self.partial = torch.empty((batch, model.nq, splits, model.dim),
                                   dtype=torch.float32, device=device)
        self.lse = torch.empty((batch, model.nq, splits), dtype=torch.float32, device=device)
        self.graph = None
        if output > 1:
            self.capture()

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

    def prefill(self, input_ids):
        self.position.zero_()
        self.prompt.copy_(torch.tensor(input_ids, dtype=torch.int64, device="cpu"))
        logits = self.model.forward(self.prompt, self, prefill=True)
        torch.argmax(logits, dim=-1, keepdim=True, out=self.tokens)
        # Prompt slots were overwritten; later slots remain masked until used.
        self.position.fill_(self.shape[1])
