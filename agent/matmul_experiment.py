"""BF16 small-batch projection experiments, including cold-weight graph timing."""

import statistics
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


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


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    print(torch.cuda.get_device_name(), flush=True)
    for m in (1, 4, 16):
        for n, k in ((6144, 2560), (2560, 4096), (19456, 2560), (2560, 9728), (151936, 2560)):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            w = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()
            # Rotate actual weight buffers beyond H100's L2 capacity, avoiding
            # a misleading warm-cache win that disappears in the full model.
            copies = min(8, max(1, triton.cdiv(64 * 1024 * 1024, w.numel() * w.element_size()) + 1))
            weights = [w] + [w.clone() for _ in range(copies - 1)]
            expected = F.linear(x, w)
            baseline = bench(F.linear, x, weights)
            values = [(baseline, "torch")]
            configs = [("gemm", bn, bk, split) for bn, bk, split in
                       ((32, 64, 1), (64, 64, 1), (64, 64, 4), (64, 128, 4), (64, 64, 8))]
            if m == 1:
                configs += [("gemv", bn, bk, split) for bn, bk, split in
                            ((4, 512, 1), (4, 1024, 1), (8, 512, 1), (4, 512, 4))]
            for config in configs:
                projection = Projection(m, n, k, config)
                actual = projection(x, w)
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
                values.append((bench(projection, x, weights), config))
            values.sort(key=lambda value: value[0])
            print((m, n, k), "torch", round(baseline, 2), "us; best",
                  [(round(t, 2), c) for t, c in values[:3]], flush=True)
            del weights, w


if __name__ == "__main__":
    main()
