"""Numerical and state tests on CUDA; no checkpoint downloads.

Run with the benchmark-pinned packages:
    python -m unittest discover -s tests -p test_engine_gpu.py -v
"""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from engine import Engine
from runtime import Model
from kernels.rmsnorm import rms_norm
from kernels.pointwise import add_rms_norm, prepare_qkv, swiglu
from kernels.attention import decode_attention
from kernels.linear import Projection
from kernels.packed_linear import PackedProjection, pack


def norm(x, weight, eps=1e-6):
    f = x.float()
    return (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + eps)).to(x.dtype) * weight


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class KernelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    @torch.inference_mode()
    def test_bf16_projection_split_reduction_and_masked_tails(self):
        for batch in (1, 4, 16):
            x = torch.randn(batch, 1003, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(37, 1003, device="cuda", dtype=torch.bfloat16)
            expected = F.linear(x.double(), w.double()).bfloat16()
            configs = [("gemm", 32, 64, 1), ("gemm", 64, 64, 4), ("gemm", 64, 64, 8)]
            if batch == 1:
                configs += [("gemv", 4, 512, 1), ("gemv", 4, 512, 4)]
            for config in configs:
                actual = Projection(batch, 37, 1003, config)(x, w)
                torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.002)

    @torch.inference_mode()
    def test_norm_residual_and_swiglu_rounding(self):
        for width in (128, 2560):
            x = torch.randn(33, width, device="cuda", dtype=torch.bfloat16)
            r = torch.randn_like(x)
            w = torch.randn(width, device="cuda", dtype=torch.bfloat16)
            torch.testing.assert_close(rms_norm(x, w, 1e-6), norm(x, w), rtol=0.01, atol=0.008)
            summed, y = add_rms_norm(x, r, w, 1e-6)
            torch.testing.assert_close(summed, x + r, rtol=0, atol=0)
            torch.testing.assert_close(y, norm(x + r, w), rtol=0.01, atol=0.008)
        packed = torch.randn(19, 19456, device="cuda", dtype=torch.bfloat16)
        gate, up = packed.chunk(2, -1)
        torch.testing.assert_close(swiglu(packed), F.silu(gate) * up, rtol=0.008, atol=0.001)

    @torch.inference_mode()
    def test_qk_norm_rope_cache_layout_and_absolute_positions(self):
        batch, length, nq, nk, dim, start, cap = 3, 19, 32, 8, 128, 7, 39
        packed = torch.randn(batch, length, (nq + 2 * nk) * dim,
                             device="cuda", dtype=torch.bfloat16)
        qw = (1 + torch.randn(dim, device="cuda") * 0.2).bfloat16()
        kw = (1 + torch.randn(dim, device="cuda") * 0.2).bfloat16()
        angles = torch.randn(cap, dim // 2, device="cuda").repeat(1, 2)
        cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
        key = torch.full((batch, nk, cap, dim), float("nan"), device="cuda", dtype=torch.bfloat16)
        value = torch.full_like(key, float("nan"))
        pos = torch.tensor(start, device="cuda", dtype=torch.int32)
        q = prepare_qkv(packed, qw, kw, cos, sin, pos, key, value, nq, nk, dim, 1e-6)
        qr, kr, vr = packed.split((nq * dim, nk * dim, nk * dim), -1)
        qr = norm(qr.reshape(batch, length, nq, dim), qw).transpose(1, 2)
        kr = norm(kr.reshape(batch, length, nk, dim), kw).transpose(1, 2)
        qr, kr = apply_rotary_pos_emb(qr, kr, cos[None, start:start + length], sin[None, start:start + length])
        torch.testing.assert_close(q, qr, rtol=0.01, atol=0.032)
        torch.testing.assert_close(key[:, :, start:start + length], kr, rtol=0.01, atol=0.032)
        torch.testing.assert_close(value[:, :, start:start + length],
                                   vr.reshape(batch, length, nk, dim).transpose(1, 2), rtol=0, atol=0)
        self.assertTrue(key[:, :, :start].isnan().all().item())
        self.assertTrue(key[:, :, start + length:].isnan().all().item())

    @torch.inference_mode()
    def test_attention_partial_blocks_grouped_heads_and_poisoned_tail(self):
        for batch, block in ((1, 64), (1, 256), (3, 64), (3, 256), (16, 256)):
            nq, nk, dim, cap = 32, 8, 128, 2121
            splits = (cap + block - 1) // block
            partial = torch.empty(batch, nq, splits, dim, device="cuda", dtype=torch.float32)
            lse = torch.empty(batch, nq, splits, device="cuda", dtype=torch.float32)
            for length in (1, 63, 64, 65, 255, 256, 257, 513, 2112):
                q = torch.randn(batch, nq, 1, dim, device="cuda", dtype=torch.bfloat16)
                k = torch.randn(batch, nk, cap, dim, device="cuda", dtype=torch.bfloat16)
                v = torch.randn_like(k)
                k[:, :, length:] = float("nan")
                v[:, :, length:] = float("nan")
                pos = torch.tensor(length - 1, device="cuda", dtype=torch.int32)
                actual = decode_attention(q, k, v, pos, partial, lse, block)
                expected = F.scaled_dot_product_attention(
                    q, k[:, :, :length].repeat_interleave(nq // nk, dim=1),
                    v[:, :, :length].repeat_interleave(nq // nk, dim=1),
                ).squeeze(2)
                torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.004)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GenerationTests(unittest.TestCase):
    @torch.inference_mode()
    def test_graph_replay_teacher_forced_prefix_reset_and_shape_change(self):
        torch.manual_seed(42)
        config = Qwen3Config(
            hidden_size=256, intermediate_size=768, num_hidden_layers=12,
            num_attention_heads=8, num_key_value_heads=2, head_dim=128,
            vocab_size=1024, max_position_embeddings=4096,
            rope_theta=5000000.0, tie_word_embeddings=True,
        )
        config._attn_implementation = "sdpa"
        reference = Qwen3ForCausalLM(config).eval().to(device="cuda", dtype=torch.bfloat16)
        engine = Engine.__new__(Engine)
        engine.model = Model(reference)
        engine.state = None
        self.assertEqual(list(engine.generate([[1, 2, 3]], 0)), [])
        for batch, length, count in ((2, 17, 4), (2, 17, 4), (1, 257, 5), (3, 19, 1)):
            ids = torch.randint(0, config.vocab_size, (batch, length), device="cuda")
            emitted = list(engine.generate(ids.tolist(), count))
            self.assertEqual(len(emitted), count)
            self.assertTrue(all(len(step) == batch for step in emitted))
            continuation = torch.tensor(emitted, device="cuda").T
            full = torch.cat((ids, continuation), dim=1)
            replay = reference(full, use_cache=False).logits[:, length - 1:length + count - 1]
            gap = replay.amax(-1) - replay.gather(-1, continuation.unsqueeze(-1)).squeeze(-1)
            self.assertLessEqual(gap.max().item(), 0.15)
            # Compare first-step logits as well as token choices.
            state = getattr(engine.state, "base", engine.state)
            state.position.zero_()
            actual = engine.model.forward(ids, state, prefill=True)
            expected = reference(ids, logits_to_keep=1, use_cache=False).logits[:, 0]
            torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.04)

    @torch.inference_mode()
    def test_full_width_layers_cached_logits(self):
        torch.manual_seed(71)
        config = Qwen3Config(
            hidden_size=2560, intermediate_size=9728, num_hidden_layers=2,
            num_attention_heads=32, num_key_value_heads=8, head_dim=128,
            vocab_size=2048, max_position_embeddings=4096,
            rope_theta=5000000.0, tie_word_embeddings=True,
        )
        config._attn_implementation = "sdpa"
        reference = Qwen3ForCausalLM(config).eval().to(device="cuda", dtype=torch.bfloat16)
        engine = Engine.__new__(Engine)
        engine.model, engine.state = Model(reference), None
        layer = engine.model.layers[0]
        # Force lossless projections even if tuning would choose cuBLAS.
        # Their composition is checked against teacher-forced native logits.
        weights = {"qkv": layer.qkv, "out": layer.out, "gate_up": layer.gate_up,
                   "down": layer.down, "lm_head": engine.model.lm_head}
        for layer in engine.model.layers:
            for name in ("qkv", "out", "gate_up", "down"):
                weight = getattr(layer, name)
                engine.model.packed_weights[weight.data_ptr()] = pack(weight)
        engine.model.packed_weights[engine.model.lm_head.data_ptr()] = pack(engine.model.lm_head)
        engine.model.projection_plans[4] = {
            name: PackedProjection(4, *weight.shape, (64, 128, 4), engine.model.packed_weights)
            for name, weight in weights.items()
        }
        ids = torch.randint(0, config.vocab_size, (4, 129), device="cuda")
        emitted = torch.tensor(list(engine.generate(ids.tolist(), 5)), device="cuda").T
        replay = reference(torch.cat((ids, emitted), 1), use_cache=False).logits[:, 128:133]
        gap = replay.amax(-1) - replay.gather(-1, emitted.unsqueeze(-1)).squeeze(-1)
        self.assertLessEqual(gap.max().item(), 0.3)
        state = getattr(engine.state, "base", engine.state)
        state.prefill(ids.tolist())
        native = reference(ids, logits_to_keep=1, use_cache=True)
        for _ in range(4):
            token = state.tokens.clone()
            actual = engine.model.forward(token, state)
            native = reference(token, past_key_values=native.past_key_values,
                               logits_to_keep=1, use_cache=True)
            torch.testing.assert_close(actual, native.logits[:, 0], rtol=0.04, atol=0.1)
            state.tokens.copy_(actual.argmax(-1, keepdim=True))
            state.position.add_(1)


if __name__ == "__main__":
    unittest.main()
