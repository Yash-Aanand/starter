"""BF16 projections selected once during warmup on the actual GPU.

Weights and activations remain BF16; partial dot products accumulate in FP32
and round to BF16 only after their complete reduction.
"""

import statistics
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from kernels.packed_linear import pack, PackedProjection


@triton.jit
def gemm(X, W, OUT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr):
    ni, mi, si = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = mi * BM + tl.arange(0, BM)
    n = ni * BN + tl.arange(0, BN)
    kk = si * BK + tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for i in range(tl.cdiv(K, BK * SPLIT)):
        k = kk + i * BK * SPLIT
        x = tl.load(X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0)
        w = tl.load(W + n[None, :] * K + k[:, None], (n[None, :] < N) & (k[:, None] < K), 0)
        acc += tl.dot(x, w)
    tl.store(OUT + (si * M + m[:, None]) * N + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))


@triton.jit
def gemv(X, W, OUT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
         BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr):
    ni, m, si = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n = ni * BN + tl.arange(0, BN)
    kk = si * BK + tl.arange(0, BK)
    acc = tl.zeros((BN, BK), tl.float32)
    for i in range(tl.cdiv(K, BK * SPLIT)):
        k = kk + i * BK * SPLIT
        x = tl.load(X + m * K + k, k < K, 0).to(tl.float32)
        w = tl.load(W + n[:, None] * K + k[None, :], (n[:, None] < N) & (k[None, :] < K), 0).to(tl.float32)
        acc += w * x[None, :]
    result = tl.sum(acc, 1)
    tl.store(OUT + (si * M + m) * N + n, result, n < N)


@triton.jit
def reduce(Part, OUT, SIZE: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    s = tl.arange(0, triton.next_power_of_2(S))
    p = tl.load(Part + s[:, None] * SIZE + r[None, :], (s[:, None] < S) & (r[None, :] < SIZE), 0)
    tl.store(OUT + r, tl.sum(p, 0), r < SIZE)


class Projection:
    def __init__(self, m, n, k, config):
        self.config = config
        self.m, self.n, self.k = m, n, k
        self.out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        self.part = torch.empty(config[-1], m, n, device="cuda", dtype=torch.float32) if config[-1] > 1 else self.out

    def __call__(self, x, w):
        kind, bn, bk, split = self.config
        m, n, k = self.m, self.n, self.k
        if kind == "gemv":
            gemv[(triton.cdiv(n, bn), m, split)](x, w, self.part, m, n, k, bn, bk, split, num_warps=4)
        else:
            gemm[(triton.cdiv(n, bn), triton.cdiv(m, 16), split)](x, w, self.part, m, n, k, 16, bn, bk, split, num_warps=4)
        if split > 1:
            reduce[(triton.cdiv(m * n, 256),)](self.part, self.out, m * n, split, 256)
        return self.out


def bench(function, x, weights):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for w in weights:
            function(x, w)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for w in weights:
                function(x, w)
        times = []
        for _ in range(3):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(15):
                graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000 / (15 * len(weights)))
    torch.cuda.current_stream().wait_stream(stream)
    return statistics.median(times)


def choose_projections(model, batch):
    layer = model.layers[0]
    weights_by_name = {"qkv": layer.qkv, "out": layer.out,
                       "gate_up": layer.gate_up, "down": layer.down,
                       "lm_head": model.lm_head}
    plans, by_shape = {}, {}
    for name, w in weights_by_name.items():
        n, k = w.shape
        if (n, k) in by_shape:
            plans[name] = by_shape[n, k]
            continue
        x = torch.ones((batch, k), device=w.device, dtype=w.dtype)
        # A full decode scans all layer weights. Rotate buffers beyond H100's
        # L2 capacity so tuning doesn't select a kernel on unrealistically hot
        # weights. Copies are temporary and never replace model parameters.
        copies = min(8, max(1, triton.cdiv(64 * 1024 * 1024, w.numel() * w.element_size()) + 1))
        weights = [w] + [w.clone() for _ in range(copies - 1)]
        best, best_time = F.linear, bench(F.linear, x, weights)
        configs = [("gemm", bn, bk, split) for bn, bk, split in
                   ((32, 64, 1), (64, 64, 1), (64, 64, 4), (64, 128, 4), (64, 64, 8))]
        if batch == 1:
            configs += [("gemv", bn, bk, split) for bn, bk, split in
                        ((4, 512, 1), (4, 1024, 1), (8, 512, 1), (4, 512, 4))]
        for config in configs:
            candidate = Projection(batch, n, k, config)
            elapsed = bench(candidate, x, weights)
            # Keep cuBLAS unless a candidate wins beyond small timing noise.
            if elapsed < best_time * 0.97:
                best, best_time = candidate, elapsed
        # Lossless storage is useful only when its decode work costs less than
        # the memory traffic it saves. Measure it on this GPU during warmup.
        if batch <= 32 and n * k >= 1048576:
            encoded = model.packed_weights.get(w.data_ptr())
            if encoded is None:
                encoded = pack(w)
            packed_copies = min(8, max(1, triton.cdiv(64 * 1024 * 1024,
                                sum(t.numel() * t.element_size() for t in encoded[:3])) + 1))
            packed_weights = [encoded] + [tuple(t.clone() for t in encoded)
                                          for _ in range(packed_copies - 1)]
            for config in ((64, 128, 4), (64, 128, 8), (32, 128, 4), (64, 256, 4)):
                candidate = PackedProjection(batch, n, k, config, model.packed_weights)
                elapsed = bench(candidate.packed, x, packed_weights)
                if elapsed < best_time * 0.97:
                    best, best_time = candidate, elapsed
            if isinstance(best, PackedProjection):
                model.packed_weights[w.data_ptr()] = encoded
            del packed_weights, encoded
        plans[name] = by_shape[n, k] = best
        del weights
    # Prepare selected representations for every layer before graph capture.
    # BF16 originals are retained for the large prefill GEMMs.
    for name, plan in plans.items():
        if isinstance(plan, PackedProjection):
            tensors = [model.lm_head] if name == "lm_head" else [getattr(layer, name) for layer in model.layers]
            for weight in tensors:
                if weight.data_ptr() not in model.packed_weights:
                    model.packed_weights[weight.data_ptr()] = pack(weight)
    return plans


def project(x, weight, plan):
    if plan is None:
        return F.linear(x, weight)
    result = plan(x.reshape(-1, x.shape[-1]), weight)
    return result.view(*x.shape[:-1], weight.shape[0])
