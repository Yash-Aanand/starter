"""Bit-exact BF16 storage and numerically equivalent projection arithmetic."""

import sys
import unittest
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from kernels.packed_linear import pack, load_bf16, PackedProjection


@triton.jit
def restore(LOW, HIGH, META, EX, OUT, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    w = load_bf16(LOW, HIGH, META, EX, idx, idx < SIZE)
    tl.store(OUT + idx, w.to(tl.uint16, bitcast=True), idx < SIZE)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PackedTests(unittest.TestCase):
    @torch.inference_mode()
    def test_all_bf16_bits_and_chunk_boundaries(self):
        torch.manual_seed(913)
        # Sorted and shuffled patterns exercise offset and exception blocks,
        # including subnormals, signed zeros, infinities and NaN payloads.
        bits = torch.arange(65536, device="cuda").to(torch.int16)
        cases = [bits, bits[torch.randperm(65536, device="cuda")],
                 bits.repeat(17)[:1048576 + 379]]
        for original in cases:
            encoded = pack(original.view(torch.bfloat16).view(1, -1))
            actual = torch.empty_like(original)
            restore[(triton.cdiv(original.numel(), 256),)](*encoded, actual, original.numel(), 256)
            self.assertTrue(torch.equal(actual, original))

    @torch.inference_mode()
    def test_masked_tails_splits_and_exceptions(self):
        torch.manual_seed(74)
        for batch in (1, 4, 16, 31):
            x = torch.randn(batch, 1003, device="cuda").bfloat16()
            weight = (torch.randn(37, 1003, device="cuda") * .03).bfloat16()
            weight[::3, ::31] = 1.e-20
            weight[::7, ::59] = 1.0
            encoded = pack(weight)
            expected = F.linear(x.double(), weight.double()).bfloat16()
            for config in ((64,128,4), (64,128,8), (32,128,4), (64,256,4)):
                plan = PackedProjection(batch, 37, 1003, config, {weight.data_ptr(): encoded})
                torch.testing.assert_close(plan(x,weight), expected, rtol=.008, atol=.002)


if __name__ == "__main__":
    unittest.main()
