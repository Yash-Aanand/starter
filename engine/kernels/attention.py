"""Dense split-K decode attention over the valid part of a static BF16 cache."""

import torch
import triton
import triton.language as tl


@triton.jit
def _partial(Q, K, V, POS, OUT, LSE,
             NQ: tl.constexpr, NK: tl.constexpr, CAP: tl.constexpr,
             D: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    head_batch, split = tl.program_id(0), tl.program_id(1)
    b, head = head_batch // NQ, head_batch % NQ
    kv = head // (NQ // NK)
    length = tl.load(POS) + 1
    d = tl.arange(0, D)
    n = split * BLOCK + tl.arange(0, BLOCK)
    valid = (n < length) & (n < CAP)
    q = tl.load(Q + head_batch * D + d).to(tl.float32)
    offsets = ((b * NK + kv) * CAP + n[:, None]) * D + d[None, :]
    k = tl.load(K + offsets, valid[:, None], 0).to(tl.float32)
    scores = tl.sum(k * q[None, :], 1) * (D ** -0.5)
    scores = tl.where(valid, scores, -float("inf"))
    maximum = tl.max(scores, 0)
    safe_max = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp(scores - safe_max)
    denom = tl.sum(weights, 0)
    v = tl.load(V + offsets, valid[:, None], 0).to(tl.float32)
    result = tl.sum(weights[:, None] * v, 0) / tl.maximum(denom, 1.0e-20)
    base = head_batch * SPLITS + split
    tl.store(OUT + base * D + d, result)
    tl.store(LSE + base, tl.where(denom > 0, safe_max + tl.log(denom), -float("inf")))


@triton.jit
def _tensor_partial(Q, K, V, POS, OUT, LSE,
                    NQ: tl.constexpr, NK: tl.constexpr, CAP: tl.constexpr,
                    D: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr,
                    T: tl.constexpr = 1, POS_STRIDE: tl.constexpr = 0,
                    BM: tl.constexpr = 16):
    # One tile shares each KV head among its query heads. Pad the four queries
    # to a tensor-core tile; padded heads never read Q or write a result.
    kv_batch, split = tl.program_id(0), tl.program_id(1)
    b, kv = kv_batch // NK, kv_batch % NK
    position = tl.load(POS + b * POS_STRIDE)
    length = position + T
    h, d = tl.arange(0, BM), tl.arange(0, D)
    n = split * BLOCK + tl.arange(0, BLOCK)
    valid = (n < length) & (n < CAP)
    head_batch = (b * NQ + kv * (NQ // NK)) * T + h
    q = tl.load(Q + head_batch[:, None] * D + d[None, :], h[:, None] < (NQ // NK) * T, 0)
    k = tl.load(K + ((b * NK + kv) * CAP + n[None, :]) * D + d[:, None], valid[None, :], 0)
    scores = tl.dot(q, k) * (D ** -0.5)
    scores = tl.where(valid[None, :] & (n[None, :] <= position + (h % T)[:, None]), scores, -float("inf"))
    maximum = tl.max(scores, 1)
    safe_max = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp(scores - safe_max[:, None])
    denom = tl.sum(weights, 1)
    v = tl.load(V + ((b * NK + kv) * CAP + n[:, None]) * D + d[None, :], valid[:, None], 0)
    # Dense BF16 attention with FP32 dot accumulation and softmax reductions.
    result = tl.dot(weights.to(v.dtype), v) / tl.maximum(denom[:, None], 1.0e-20)
    base = head_batch * SPLITS + split
    tl.store(OUT + base[:, None] * D + d[None, :], result, h[:, None] < (NQ // NK) * T)
    tl.store(LSE + base, tl.where(denom > 0, safe_max + tl.log(denom), -float("inf")), h < (NQ // NK) * T)


@triton.jit
def _merge(PART, LSE, OUT, SPLITS: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    head_batch = tl.program_id(0)
    s, d = tl.arange(0, BLOCK), tl.arange(0, D)
    lse = tl.load(LSE + head_batch * SPLITS + s, s < SPLITS, -float("inf"))
    weights = tl.exp(lse - tl.max(lse, 0))
    part = tl.load(PART + (head_batch * SPLITS + s[:, None]) * D + d[None, :],
                   s[:, None] < SPLITS, 0)
    result = tl.sum(part * weights[:, None], 0) / tl.sum(weights, 0)
    tl.store(OUT + head_batch * D + d, result)


def decode_attention(q, key, value, position, partial, lse, block=256):
    """Q [B,Nq,1,D], cache [B,Nkv,C,D], device POS = new token's slot.

    Reads slots 0..POS only. Caller owns reusable FP32 partial storage.
    Returns a new contiguous BF16 [B,Nq,D] tensor.
    """
    batch, nq, length, dim = q.shape
    splits = partial.shape[-2]
    out = torch.empty((batch, nq, length, dim), dtype=q.dtype, device=q.device)
    nk = key.shape[1]
    _tensor_partial[(batch * nk, splits)](
        q, key, value, position, partial, lse,
        nq, nk, key.shape[2], dim, splits, block, length, int(position.numel() > 1),
        max(16, triton.next_power_of_2(nq // nk * length)),
        num_warps=4 if block <= 128 else 8,
    )
    _merge[(batch * nq * length,)](
        partial, lse, out, splits, dim, triton.next_power_of_2(splits), num_warps=4,
    )
    return out.squeeze(2) if length == 1 else out
