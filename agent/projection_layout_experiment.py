"""Cold-weight BF16 layout and fused feed-forward experiments."""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from kernels.linear import Projection, bench
from kernels.pointwise import swiglu


@triton.jit
def transposed_gemv(X, W, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                    BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr):
    ni, m, si = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n = ni * BN + tl.arange(0, BN)
    kk = si * BK + tl.arange(0, BK)
    acc = tl.zeros((BK, BN), tl.float32)
    for i in range(tl.cdiv(K, BK * SPLIT)):
        k = kk + i * BK * SPLIT
        x = tl.load(X + m * K + k, k < K, 0).to(tl.float32)
        w = tl.load(W + k[:, None] * N + n[None, :], (k[:, None] < K) & (n[None, :] < N), 0).to(tl.float32)
        acc += x[:, None] * w
    tl.store(Y + (si * M + m) * N + n, tl.sum(acc, 0), n < N)


@triton.jit
def fused_mlp(X, W, Y, PART, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
              SPLIT: tl.constexpr, TRANSPOSED: tl.constexpr):
    ni, mi, si = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m, n = mi * BM + tl.arange(0, BM), ni * BN + tl.arange(0, BN)
    kk = si * BK + tl.arange(0, BK)
    gate, up = tl.zeros((BM, BN), tl.float32), tl.zeros((BM, BN), tl.float32)
    for i in range(tl.cdiv(K, BK * SPLIT)):
        k = kk + i * BK * SPLIT
        x = tl.load(X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0)
        if TRANSPOSED:
            off = k[:, None] * (2 * N) + n[None, :]
            delta = N
        else:
            off = n[None, :] * K + k[:, None]
            delta = N * K
        mask = (n[None, :] < N) & (k[:, None] < K)
        wg = tl.load(W + off, mask, 0)
        wu = tl.load(W + off + delta, mask, 0)
        gate += tl.dot(x, wg)
        up += tl.dot(x, wu)
    mask = (m[:, None] < M) & (n[None, :] < N)
    if SPLIT == 1:
        g = gate.to(tl.bfloat16).to(tl.float32)
        u = up.to(tl.bfloat16).to(tl.float32)
        silu = (g / (1. + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + m[:, None] * N + n[None, :], silu * u, mask)
    else:
        off = ((si * M + m[:, None]) * 2 * N) + n[None, :]
        tl.store(PART + off, gate, mask)
        tl.store(PART + off + N, up, mask)


@triton.jit
def reduce_swiglu(PART, Y, M: tl.constexpr, N: tl.constexpr, SPLIT: tl.constexpr,
                  BLOCK: tl.constexpr):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    s = tl.arange(0, triton.next_power_of_2(SPLIT))
    off = (s[:, None] * M + r[None, :] // N) * 2 * N + r[None, :] % N
    mask = (s[:, None] < SPLIT) & (r[None, :] < M * N)
    g = tl.sum(tl.load(PART + off, mask, 0), 0).to(tl.bfloat16).to(tl.float32)
    u = tl.sum(tl.load(PART + off + N, mask, 0), 0).to(tl.bfloat16).to(tl.float32)
    silu = (g / (1. + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + r, silu * u, r < M * N)


class Fused:
    def __init__(self, m, n, k, bm, bn, bk, split, transposed=False):
        self.m, self.n, self.k = m, n, k
        self.config = bm, bn, bk, split, transposed
        self.out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        self.part = torch.empty(split, m, 2*n, device="cuda", dtype=torch.float32) if split > 1 else self.out

    def __call__(self, x, w):
        bm, bn, bk, split, transposed = self.config
        fused_mlp[(triton.cdiv(self.n, bn), triton.cdiv(self.m, bm), split)](
            x, w, self.out, self.part, self.m, self.n, self.k,
            bm, bn, bk, split, transposed, num_warps=4)
        if split > 1:
            reduce_swiglu[(triton.cdiv(self.m * self.n, 256),)](self.part, self.out, self.m, self.n, split, 256)
        return self.out


@torch.inference_mode()
def main():
    torch.manual_seed(83)
    print(torch.cuda.get_device_name(), flush=True)
    for m in (1, 4, 16, 512, 8192):
        n, k = 9728, 2560
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = (torch.randn(2*n, k, device="cuda", dtype=torch.bfloat16) * .02).contiguous()
        wt = w.T.contiguous().T
        expected = swiglu(F.linear(x, w))
        reference = lambda a, b: swiglu(F.linear(a, b))
        baseline = bench(reference, x, [w])
        results = [(baseline, "torch"), (bench(reference, x, [wt]), "torch-transposed")]
        if m <= 16:
            for bn, bk, split in ((32,64,1),(64,64,1),(64,64,4),(64,128,4),(64,64,8)):
                old = Projection(m, 2*n, k, ("gemm",bn,bk,split))
                results.append((bench(lambda a,b:swiglu(old(a,b)), x, [w]), ("existing",bn,bk,split)))
        configs = [(16,32,64,1),(16,64,64,1),(16,64,64,4),(16,64,128,4),(16,64,64,8)] if m <= 16 else [(32,32,32,1),(32,64,32,1),(32,64,64,1),(64,32,32,1),(64,64,32,1),(64,64,64,1)]
        for bm, bn, bk, split in configs:
            for transposed in (False, True):
                candidate = Fused(m,n,k,bm,bn,bk,split,transposed)
                weight = wt if transposed else w
                actual = candidate(x,weight)
                torch.testing.assert_close(actual, expected, rtol=.025, atol=.035)
                results.append((bench(candidate,x,[weight]),candidate.config))
        print("fused",m,"baseline",round(baseline,2),"us best",sorted(results,key=lambda p:p[0])[:5],flush=True)
        del w, wt, candidate, x, expected
    for m in (1,4,16):
        for n,k in ((6144,2560),(2560,4096),(19456,2560),(2560,9728),(151936,2560)):
            x=torch.randn(m,k,device="cuda",dtype=torch.bfloat16)
            w=torch.randn(n,k,device="cuda",dtype=torch.bfloat16)*.02
            wt=w.T.contiguous().T
            copies=min(8,max(1,triton.cdiv(64*1024*1024,w.numel()*2)+1))
            weights=[w]+[w.clone() for _ in range(copies-1)]
            weights_t=[wt]+[wt.clone(memory_format=torch.preserve_format) for _ in range(copies-1)]
            baseline=bench(F.linear,x,weights)
            times=[(baseline,"torch"),(bench(F.linear,x,weights_t),"torch-transposed")]
            out=torch.empty(m,n,device="cuda",dtype=torch.bfloat16)
            for bn,bk in ((32,32),(64,32),(128,16),(128,32),(64,64)):
                def call(a,b):
                    transposed_gemv[(triton.cdiv(n,bn),m,1)](a,b,out,m,n,k,bn,bk,1,num_warps=4)
                    return out
                torch.testing.assert_close(call(x,wt), F.linear(x,w),rtol=.02,atol=.02)
                times.append((bench(call,x,weights_t),(bn,bk)))
            print("layout",(m,n,k),"torch",round(baseline,2),sorted(times,key=lambda p:p[0])[:3],flush=True)
            del weights,weights_t,w,wt


if __name__ == "__main__":
    main()
