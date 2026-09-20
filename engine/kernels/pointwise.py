"""Fused BF16 operations with the reference's intermediate rounding points."""

import torch
import triton
import triton.language as tl


@triton.jit
def _add_norm(X, R, W, S, Y, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + cols, cols < N, 0).to(tl.float32)
    r = tl.load(R + row * N + cols, cols < N, 0).to(tl.float32)
    summed = (x + r).to(S.dtype.element_ty)
    tl.store(S + row * N + cols, summed, cols < N)
    f = summed.to(tl.float32)
    inv = tl.rsqrt(tl.sum(f * f, 0) / N + EPS)
    normalized = (f * inv).to(Y.dtype.element_ty).to(tl.float32)
    w = tl.load(W + cols, cols < N, 0).to(tl.float32)
    tl.store(Y + row * N + cols, normalized * w, cols < N)


def add_rms_norm(x, residual, weight, eps):
    """Contiguous BF16 [...,H] inputs; returns new residual and its norm."""
    width = x.shape[-1]
    summed, out = torch.empty_like(x), torch.empty_like(x)
    _add_norm[(x.numel() // width,)](
        x, residual, weight, summed, out, width, eps,
        triton.next_power_of_2(width), num_warps=4, enable_fp_fusion=False,
    )
    return summed, out


@triton.jit
def _swiglu(X, Y, N: tl.constexpr, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offsets // N, offsets % N
    gate = tl.load(X + row * (2 * N) + col, offsets < TOTAL, 0).to(tl.float32)
    up = tl.load(X + row * (2 * N) + N + col, offsets < TOTAL, 0).to(tl.float32)
    activated = (gate / (1.0 + tl.exp(-gate))).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + offsets, activated * up, offsets < TOTAL)


def swiglu(packed):
    width = packed.shape[-1] // 2
    out = torch.empty((*packed.shape[:-1], width), dtype=packed.dtype, device=packed.device)
    _swiglu[(triton.cdiv(out.numel(), 512),)](
        packed, out, width, out.numel(), 512, enable_fp_fusion=False,
    )
    return out


@triton.jit
def _qkv(QKV, QW, KW, COS, SIN, POS, Q, K, V,
         T: tl.constexpr, CAP: tl.constexpr, NQ: tl.constexpr, NK: tl.constexpr,
         D: tl.constexpr, EPS: tl.constexpr, HEADS: tl.constexpr, POS_STRIDE: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1) * HEADS + tl.arange(0, HEADS)
    b, t = row // T, row % T
    d = tl.arange(0, D)
    position = tl.load(POS + b * POS_STRIDE) + t
    src = row * (NQ + 2 * NK) * D + head[:, None] * D + d[None, :]
    valid = head < NQ + 2 * NK
    x = tl.load(QKV + src, valid[:, None], 0).to(tl.float32)
    qweight = tl.load(QW + d).to(tl.float32)
    kweight = tl.load(KW + d).to(tl.float32)
    weight = tl.where(head[:, None] < NQ, qweight[None, :], kweight[None, :])
    inv = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
    normalized = (x * inv[:, None]).to(Q.dtype.element_ty).to(tl.float32)
    scaled = (normalized * weight).to(Q.dtype.element_ty).to(tl.float32)
    # Triton 3.1 has no tl.gather. Load the opposite half and apply the
    # same norm (including both casts) instead of using a newer intrinsic.
    other = (d + D // 2) % D
    opposite = row * (NQ + 2 * NK) * D + head[:, None] * D + other[None, :]
    partner = tl.load(QKV + opposite, valid[:, None], 0).to(tl.float32)
    partner_qw = tl.load(QW + other).to(tl.float32)
    partner_kw = tl.load(KW + other).to(tl.float32)
    partner_w = tl.where(head[:, None] < NQ, partner_qw[None, :], partner_kw[None, :])
    rotated = (partner * inv[:, None]).to(Q.dtype.element_ty).to(tl.float32)
    rotated = (rotated * partner_w).to(Q.dtype.element_ty).to(tl.float32)
    rotated = tl.where(d[None, :] < D // 2, -rotated, rotated)
    cos = tl.load(COS + position * D + d).to(tl.float32)
    sin = tl.load(SIN + position * D + d).to(tl.float32)
    # Reference RoPE rounds both products, then their sum.
    left = (scaled * cos[None, :]).to(Q.dtype.element_ty).to(tl.float32)
    right = (rotated * sin[None, :]).to(Q.dtype.element_ty).to(tl.float32)
    result = left + right
    qoffset = ((b * NQ + head) * T + t)[:, None] * D + d[None, :]
    koffset = ((b * NK + head - NQ) * CAP + position)[:, None] * D + d[None, :]
    voffset = ((b * NK + head - NQ - NK) * CAP + position)[:, None] * D + d[None, :]
    tl.store(Q + qoffset, result, (head < NQ)[:, None])
    tl.store(K + koffset, result, ((head >= NQ) & (head < NQ + NK))[:, None])
    tl.store(V + voffset, x, ((head >= NQ + NK) & valid)[:, None])


def prepare_qkv(packed, qw, kw, cos, sin, position, key, value, nq, nk, dim, eps):
    """Packed [B,T,(Nq+2Nk)*D] -> Q [B,Nq,T,D]; writes K/V at POS+t.

    Position is an int32 device scalar or one absolute offset per sequence.
    Cache layout is [B,Nk,capacity,D]; returned Q owns its storage.
    """
    batch, length, _ = packed.shape
    q = torch.empty((batch, nq, length, dim), dtype=packed.dtype, device=packed.device)
    _qkv[(batch * length, triton.cdiv(nq + 2 * nk, 8))](
        packed, qw, kw, cos, sin, position, q, key, value,
        length, key.shape[2], nq, nk, dim, eps, 8, int(position.numel() > 1),
        num_warps=4, enable_fp_fusion=False,
    )
    return q
