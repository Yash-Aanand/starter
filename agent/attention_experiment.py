"""Compare split attention variants; experiments stay outside engine/.

The scalar grouped variant preserves FP32 intermediates. The tensor-core
variant rounds softmax weights to BF16 before multiplying BF16 values, as in
Flash Attention; it requires separate numerical checks.
Timings use CUDA graphs on the local GPU and do not predict leaderboard TPS.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.testing import do_bench_cudagraph

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from kernels.attention import _merge, _partial


@triton.jit
def _grouped(Q, K, V, POS, OUT, LSE,
             NQ: tl.constexpr, NK: tl.constexpr, CAP: tl.constexpr,
             D: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    kv_batch, split = tl.program_id(0), tl.program_id(1)
    b, kv = kv_batch // NK, kv_batch % NK
    length = tl.load(POS) + 1
    d = tl.arange(0, D)
    n = split * BLOCK + tl.arange(0, BLOCK)
    valid = (n < length) & (n < CAP)
    offsets = ((b * NK + kv) * CAP + n[:, None]) * D + d[None, :]
    k = tl.load(K + offsets, valid[:, None], 0).to(tl.float32)
    v = tl.load(V + offsets, valid[:, None], 0).to(tl.float32)
    for h in tl.static_range(NQ // NK):
        head_batch = b * NQ + kv * (NQ // NK) + h
        q = tl.load(Q + head_batch * D + d).to(tl.float32)
        scores = tl.sum(k * q[None, :], 1) * (D ** -0.5)
        scores = tl.where(valid, scores, -float("inf"))
        maximum = tl.max(scores, 0)
        safe_max = tl.where(maximum == -float("inf"), 0.0, maximum)
        weights = tl.exp(scores - safe_max)
        denom = tl.sum(weights, 0)
        result = tl.sum(weights[:, None] * v, 0) / tl.maximum(denom, 1.0e-20)
        base = head_batch * SPLITS + split
        tl.store(OUT + base * D + d, result)
        tl.store(LSE + base, tl.where(denom > 0, safe_max + tl.log(denom), -float("inf")))


class Variant:
    def __init__(self, batch, nq, nk, cap, dim, block, warps, grouped):
        self.block, self.warps, self.grouped = block, warps, grouped
        self.splits = triton.cdiv(cap, block)
        self.partial = torch.empty(batch, nq, self.splits, dim, device="cuda", dtype=torch.float32)
        self.lse = torch.empty(batch, nq, self.splits, device="cuda", dtype=torch.float32)
        self.out = torch.empty(batch, nq, dim, device="cuda", dtype=torch.bfloat16)

    def __call__(self, q, k, v, pos):
        batch, nq, _, dim = q.shape
        nk, cap = k.shape[1:3]
        kernel = _tensor if self.grouped == "tensor" else _grouped if self.grouped else _partial
        heads = nk if self.grouped else nq
        kernel[(batch * heads, self.splits)](
            q, k, v, pos, self.partial, self.lse,
            nq, nk, cap, dim, self.splits, self.block, num_warps=self.warps,
        )
        _merge[(batch * nq,)](
            self.partial, self.lse, self.out,
            self.splits, dim, triton.next_power_of_2(self.splits), num_warps=4,
        )
        return self.out


@triton.jit
def _tensor(Q, K, V, POS, OUT, LSE,
            NQ: tl.constexpr, NK: tl.constexpr, CAP: tl.constexpr,
            D: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    kv_batch, split = tl.program_id(0), tl.program_id(1)
    b, kv = kv_batch // NK, kv_batch % NK
    length = tl.load(POS) + 1
    h, d = tl.arange(0, 16), tl.arange(0, D)
    n = split * BLOCK + tl.arange(0, BLOCK)
    valid = (n < length) & (n < CAP)
    head_batch = b * NQ + kv * (NQ // NK) + h
    q = tl.load(Q + head_batch[:, None] * D + d[None, :], h[:, None] < NQ // NK, 0)
    k = tl.load(K + ((b * NK + kv) * CAP + n[None, :]) * D + d[:, None], valid[None, :], 0)
    scores = tl.dot(q, k) * (D ** -0.5)
    scores = tl.where(valid[None, :], scores, -float("inf"))
    maximum = tl.max(scores, 1)
    safe_max = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp(scores - safe_max[:, None])
    denom = tl.sum(weights, 1)
    v = tl.load(V + ((b * NK + kv) * CAP + n[:, None]) * D + d[None, :], valid[:, None], 0)
    result = tl.dot(weights.to(v.dtype), v) / tl.maximum(denom[:, None], 1.0e-20)
    base = head_batch * SPLITS + split
    tl.store(OUT + base[:, None] * D + d[None, :], result, h[:, None] < NQ // NK)
    tl.store(LSE + base, tl.where(denom > 0, safe_max + tl.log(denom), -float("inf")), h < NQ // NK)


@torch.inference_mode()
def main():
    torch.manual_seed(71)
    print(torch.cuda.get_device_name(), flush=True)
    for batch, length in ((1, 513), (4, 2112), (16, 640)):
        nq, nk, dim, cap = 32, 8, 128, length + 17
        q = torch.randn(batch, nq, 1, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(batch, nk, cap, dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        pos = torch.tensor(length - 1, device="cuda", dtype=torch.int32)
        k[:, :, length:] = float("nan")
        v[:, :, length:] = float("nan")
        variants = [(block, warps, grouped) for grouped in (False, True)
                    for block, warps in ((64, 4), (128, 4), (256, 8))]
        if "--tensor" in sys.argv:
            variants = [(256, 8, False), (64, 4, "tensor"),
                        (128, 4, "tensor"), (256, 8, "tensor")]
        for block, warps, grouped in variants:
            variant = Variant(batch, nq, nk, cap, dim, block, warps, grouped)
            # Include empty splits and the 256-token boundary in correctness checks.
            for valid in (1, 255, 256, 257, length):
                pos.fill_(valid - 1)
                actual = variant(q, k, v, pos)
                expected = F.scaled_dot_product_attention(
                    q, k[:, :, :valid].repeat_interleave(nq // nk, dim=1),
                    v[:, :, :valid].repeat_interleave(nq // nk, dim=1),
                ).squeeze(2)
                torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.025)
            # Warm compiled kernels, then median of three separate measurements.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                timings = sorted(do_bench_cudagraph(lambda: variant(q, k, v, pos), rep=60) * 1000
                                 for _ in range(3))
            torch.cuda.current_stream().wait_stream(stream)
            print((batch, length), grouped if grouped == "tensor" else "grouped" if grouped else "per-head",
                  "block", block, "warps", warps, round(timings[1], 3), "us", flush=True)


if __name__ == "__main__":
    main()
