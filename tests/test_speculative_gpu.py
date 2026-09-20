"""Exact speculative-prefix acceptance and multi-token cache verification."""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "agent"))
sys.path.insert(0, str(root / "engine"))
from speculative import Speculative, accept, propose
from runtime import Model
from kernels.attention import decode_attention


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SpeculativeTests(unittest.TestCase):
    @torch.inference_mode()
    def test_accept_first_mismatch_divergent_positions_and_finished_sequence(self):
        history = torch.full((4, 16), -123, device="cuda", dtype=torch.int64)
        pos = torch.tensor([4, 4, 4, 9], device="cuda", dtype=torch.int32)
        draft = torch.tensor([[10, 11, 12, 13]] * 4, device="cuda")
        pred = torch.tensor([[50, 51, 52, 53], [11, 50, 51, 52], [11, 12, 13, 14], [11, 12, 13, 14]], device="cuda")
        accept[(4,)](history, pos, draft, pred, 16, 4, 9)
        self.assertEqual(pos.tolist(), [5, 6, 8, 9])
        self.assertEqual(history[0, 5:10].tolist(), [50, -123, -123, -123, -123])
        self.assertEqual(history[1, 5:10].tolist(), [11, 50, -123, -123, -123])
        self.assertEqual(history[2, 5:10].tolist(), [11, 12, 13, 14, -123])
        self.assertTrue((history[3] == -123).all().item())

    @torch.inference_mode()
    def test_lookup_ignores_future_and_matches_longest_suffix(self):
        history = torch.tensor([[1, 2, 3, 4, 5, 6, 1, 2, 3, 88, 88, 88, 88, 88, 88, 88],
                                [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 0, 0, 0]], device="cuda")
        pos = torch.tensor([8, 8], device="cuda", dtype=torch.int32)
        draft = torch.tensor([[3, 3, 3, 3], [8, 8, 8, 8]], device="cuda", dtype=torch.int64)
        propose[(2,)](history, pos, draft, 16, 4, 16)
        self.assertEqual(draft.tolist(), [[3, 4, 5, 6], [8, 8, 8, 8]])

    @torch.inference_mode()
    def test_multiquery_causal_attention_with_different_offsets(self):
        torch.manual_seed(51)
        b, nq, nk, t, d, cap = 3, 32, 8, 4, 128, 320
        q = torch.randn(b, nq, t, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(b, nk, cap, d, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        positions = [0, 64, 259]
        expected = []
        for i, start in enumerate(positions):
            length = start + t
            mask = torch.arange(length, device="cuda")[None, :] <= start + torch.arange(t, device="cuda")[:, None]
            expected.append(F.scaled_dot_product_attention(
                q[i:i + 1], k[i:i + 1, :, :length].repeat_interleave(4, 1),
                v[i:i + 1, :, :length].repeat_interleave(4, 1), attn_mask=mask))
            k[i, :, length:] = float("nan")
            v[i, :, length:] = float("nan")
        partial = torch.empty(b, nq, t, 5, d, device="cuda", dtype=torch.float32)
        lse = torch.empty(b, nq, t, 5, device="cuda", dtype=torch.float32)
        pos = torch.tensor(positions, device="cuda", dtype=torch.int32)
        actual = decode_attention(q, k, v, pos, partial, lse, 64)
        torch.testing.assert_close(actual, torch.cat(expected), rtol=0.025, atol=0.004)

    @torch.inference_mode()
    def test_teacher_forced_tokens_reset_and_output_count(self):
        for tied in (True, False):
            torch.manual_seed(71)
            config = Qwen3Config(hidden_size=256, intermediate_size=768, num_hidden_layers=4,
                                 num_attention_heads=8, num_key_value_heads=2, head_dim=128,
                                 vocab_size=1024, rope_theta=5000000.0, tie_word_embeddings=tied)
            config._attn_implementation = "sdpa"
            reference = Qwen3ForCausalLM(config).eval().to(device="cuda", dtype=torch.bfloat16)
            model = Model(reference)
            # Kernel configurations have separate tests. Avoid autotuning tiny
            # random models repeatedly during these control-flow checks.
            model.projection_plans = {n: {} for n in (1, 3, 4, 6, 12)}
            for batch, length, output, width in ((3, 17, 13, 4), (1, 65, 9, 4), (3, 17, 7, 2)):
                state = Speculative(model, (batch, length, output), width=width)
                for _ in range(2):
                    ids = torch.randint(0, config.vocab_size, (batch, length), device="cuda")
                    steps = list(state.generate(ids.tolist(), output))
                    self.assertEqual(len(steps), output)
                    self.assertTrue(all(len(step) == batch for step in steps))
                    tokens = torch.tensor(steps, device="cuda").T
                    logits = reference(torch.cat((ids, tokens), 1), use_cache=False).logits[:, length - 1:length + output - 1]
                    gap = logits.amax(-1) - logits.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
                    self.assertLessEqual(gap.max().item(), 0.15)


if __name__ == "__main__":
    unittest.main()
