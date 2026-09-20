"""Lossless BF16 storage with exact reconstruction inside the matmul.

Sign and mantissa bits are stored verbatim. A block stores exponent offsets
in four bits when its range fits; exceptional blocks read the original BF16
tensor, which is also retained for prefill. No weight is rounded or clipped.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def encode(W, LOW, HIGH, META, SIZE: tl.constexpr):
    block = tl.program_id(0)
    r = tl.arange(0, 128)
    offset = block * 128 + r
    bits = tl.load(W + offset, offset < SIZE, 0).to(tl.uint16, bitcast=True).to(tl.int32)
    exponent = (bits >> 7) & 255
    base = tl.min(tl.where(offset < SIZE, exponent, 255), 0)
    maximum = tl.max(tl.where(offset < SIZE, exponent, 0), 0)
    tl.store(META + block, tl.where(maximum - base <= 15, base, -1))
    tl.store(LOW + offset, (bits & 127) | ((bits >> 8) & 128))
    delta = tl.reshape((exponent - base) & 15, (64, 2))
    high = tl.sum(delta << (tl.arange(0, 2)[None, :] * 4), 1)
    tl.store(HIGH + block * 64 + tl.arange(0, 64), high)


def pack(weight):
    weight = weight.contiguous()
    blocks = triton.cdiv(weight.numel(), 128)
    low = torch.empty(blocks * 128, dtype=torch.uint8, device=weight.device)
    high = torch.empty(blocks * 64, dtype=torch.uint8, device=weight.device)
    meta = torch.empty(blocks, dtype=torch.int16, device=weight.device)
    encode[(blocks,)](weight, low, high, meta, weight.numel(), num_warps=4)
    return low, high, meta, weight


@triton.jit
def load_bf16(LOW, HIGH, META, RAW, offset, mask):
    low = tl.load(LOW + offset, mask, 0).to(tl.int32)
    high = tl.load(HIGH + offset // 2, mask, 0).to(tl.int32)
    meta = tl.load(META + offset // 128, mask, 0).to(tl.int32)
    raw = tl.load(RAW + offset, mask & (meta < 0), 0)
    exponent = meta + ((high >> ((offset % 2) * 4)) & 15)
    bits = (low & 127) | ((low & 128) << 8) | (exponent << 7)
    return tl.where(meta < 0, raw, bits.to(tl.uint16).to(tl.bfloat16, bitcast=True))


@triton.jit
def packed_gemm(X, LOW, HIGH, META, EX, OUT,
                M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr):
    ni, mi, si = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m, n = mi * 16 + tl.arange(0, 16), ni * BN + tl.arange(0, BN)
    kk = si * BK + tl.arange(0, BK)
    acc = tl.zeros((16, BN), tl.float32)
    for i in range(tl.cdiv(K, BK * SPLIT)):
        k = kk + i * BK * SPLIT
        x = tl.load(X + m[:, None] * K + k[None, :],
                    (m[:, None] < M) & (k[None, :] < K), 0)
        w = load_bf16(LOW, HIGH, META, EX, n[None, :] * K + k[:, None],
                      (n[None, :] < N) & (k[:, None] < K))
        acc += tl.dot(x, w)
    tl.store(OUT + (si * M + m[:, None]) * N + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))


@triton.jit
def sum_parts(PART, OUT, SIZE: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    s = tl.arange(0, triton.next_power_of_2(SPLIT))
    values = tl.load(PART + s[:, None] * SIZE + r[None, :],
                     (s[:, None] < SPLIT) & (r[None, :] < SIZE), 0)
    tl.store(OUT + r, tl.sum(values, 0), r < SIZE)


class PackedProjection:
    def __init__(self, m, n, k, config, weights):
        self.m, self.n, self.k = m, n, k
        self.config, self.weights = config, weights
        self.out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        split = config[-1]
        self.part = torch.empty(split, m, n, device="cuda", dtype=torch.float32) if split > 1 else self.out

    def packed(self, x, encoded):
        bn, bk, split = self.config
        m, n, k = self.m, self.n, self.k
        # Triton 3.1's async pipeline cannot schedule these byte loads;
        # one stage uses synchronous loads and is tested on the pinned runtime.
        packed_gemm[(triton.cdiv(n, bn), triton.cdiv(m, 16), split)](
            x, *encoded, self.part, m, n, k, bn, bk, split,
            num_warps=4, num_stages=1)
        if split > 1:
            sum_parts[(triton.cdiv(m * n, 256),)](self.part, self.out, m * n, split, 256)
        return self.out

    def __call__(self, x, weight):
        return self.packed(x, self.weights[weight.data_ptr()])
