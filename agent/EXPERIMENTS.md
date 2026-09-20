# Qwen3 BF16 optimization log

Current workflow: connected GitHub pushes start official runs. Public runs have
been retired; six hidden workloads determine the geometric-mean tok/s score.
The local contract and older agent client still describe the retired workflow.

## Baseline

- Commit: `94cbb4137333f1d342775054a79ea1aa6635da32` (unchanged engine).
- Run: `a1e06cc4-b6c1-4999-b772-b71dfb0c5256`.
- Submission: `5bb412f1-b775-4895-88d0-dae549a71b5f`.
- Official score: **166.3961 tok/s**, succeeded and ranked.
- Public throughput: 34.47 / 117.54 / 503.60 tok/s.
- Public TTFT: 32.74 / 201.56 / 191.54 ms.
- Public TPOT: 28.89 / 28.64 / 30.51 ms.
- Result JSON is stored in `agent/results/`.

## Candidate 1: captured dense BF16 decode

Packed QKV and gate/up weights, static BF16 KV storage, native Flash SDPA with
grouped query heads for causal prefill, split-K dense decode attention, CUDA
graph replay, fused residual/RMSNorm, head RMSNorm/RoPE/cache writes, and SwiGLU.
Rounding boundaries are preserved for RMSNorm, RoPE, SiLU, and residual sums.
No quantization, sparsity, cache eviction, or prompt-dependent fast paths.

Local validation: all five GPU tests passed on an RTX 4070 Laptop GPU with
PyTorch 2.5.1, Triton 3.1.0, and Transformers 4.51.3. These cover individual
kernels, poisoned cache tails, graph replay, prompt resets, shape changes,
teacher-forced random-model continuations, and full-width cached layer logits.
Archive validation also passed. A first local test caught use of `tl.gather`,
which is unavailable in Triton 3.1; it was replaced before submission.

Reduced-depth timing smoke test (random weights, **two layers**, RTX 4070;
these are not full-model results):

| Batch / prompt / output | Native TTFT / TPOT ms | Candidate TTFT / TPOT ms |
| --- | --- | --- |
| 1 / 512 / 32 | 11.665 / 7.245 | 10.378 / 5.729 |
| 4 / 2048 / 32 | 159.648 / 10.241 | 106.530 / 5.850 |
| 16 / 512 / 128 | 195.148 / 11.097 | 100.962 / 6.070 |

H100 run `507b996b-20ce-4031-ae29-7cc1d9f5c265` succeeded and ranked at
**729.5392 tok/s**. Public workloads: 202.83 / 411.61 / 2196.07 tok/s;
TTFT 11.46 / 120.25 / 111.90 ms; TPOT 4.72 / 6.14 / 6.46 ms.
The earlier leased run timed out in infrastructure; this was a rerun of the
same `bfa2659` submission. Current target: **at least 1,400 tok/s**.

## Candidate 2: final-position prefill, grouped attention, tuned projections

- Final layer still fills the entire prompt KV cache, but computes attention,
  output projection and MLP only for its final query. Its attention is noncausal
  because the final query can see every preceding prompt key.
- Decode attention groups query heads on BF16 tensor cores with FP32 softmax
  and accumulation. Scalar grouping alone was slower and was rejected.
- Small-batch BF16 GEMM/GEMV kernels offer split-K parallelism. Warmup compares
  these against cuBLAS on the actual device, separately per projection shape.
  Rotating weight copies exceed L2 capacity to avoid misleading hot-cache wins.
  Choices and buffers remain fixed for all measured samples.
- Full-width cached logits, teacher-forced continuations, poisoned cache tails,
  and split-K arithmetic are covered by local GPU tests.

Commit `a9b6b178d0781b83c2bfc792d999067219913689`, run
`41cc7664-0d0d-461d-970d-005d3c973792`: **878.0587 tok/s**, ranked, all cases
passed. Public throughput 229.23 / 467.64 / 2725.99 tok/s; TTFT
13.87 / 119.69 / 107.76 ms; TPOT 4.06 / 4.92 / 5.09 ms. Run took nine minutes.
The user raised the target to **1,500 tok/s**.

Local two-layer A/B before projection tuning (same weights, alternating order):

| Batch / prompt / output | Before TTFT / TPOT ms | After TTFT / TPOT ms |
| --- | --- | --- |
| 1 / 512 / 32 | 9.157 / 4.982 | 7.604 / 4.979 |
| 4 / 2048 / 32 | 98.204 / 5.351 | 59.516 / 5.320 |
| 16 / 512 / 128 | 93.165 / 5.489 | 56.993 / 5.443 |

The large two-layer prefill gain must not be extrapolated to 36 layers: only
the final layer is pruned. Projection microbenchmarks found the down projection
improved from 282 to 209 us (B4) and 285 to 211 us (B16) on the laptop. Other
projections were mostly bandwidth-limited there; H100 tuning may differ.

References: [PyTorch GPT-fast](https://pytorch.org/blog/accelerating-generative-ai-2/),
[Triton matmul](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html),
[Triton attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).

## Candidate 3: verified multi-token lookup with single-token fallback

Lookup proposes a continuation from a matching suffix within the current
sequence; otherwise, carried predictions from the previous verification pass
act as guesses. A dense, causal full-model pass checks every proposed prefix.
Only greedy tokens through the first mismatch are emitted. Sequences maintain
independent cache positions; unwritten/future cache slots stay masked.

Four-token verification for batches up to four, two-token verification for
larger batches. Every four passes, insufficient acceptance compared with the
GPU's measured verification cost switches to a captured one-token path.
History, guesses and acceptance tracking reset for every generation.

This is an H100 experiment, not a claimed speedup: random-model local runs had
poor acceptance and switched to ordinary decoding, with roughly 0-5% overhead.
The real corpus and trained model determine whether it helps. Fresh prompts
across the five samples also make the timing-spread gate a material concern.
Earlier 878.1 tok/s commit remains available for rollback.

Additional tests cover first-mismatch acceptance, divergent sequence positions,
finished sequences, two- and four-token blocks, causal multi-query attention,
teacher-forced continuations, and complete resets.

References: [Prompt Lookup Decoding](https://github.com/apoorvumang/prompt-lookup-decoding),
[LMSYS lookahead decoding](https://lmsys.org/blog/2023-11-21-lookahead-decoding/).

Local Python lives at `/home/yasha/.cache/dryft-starter-venv/bin/python` in WSL,
outside the Windows editor's environment-discovery path. From PowerShell:

```powershell
wsl -d Ubuntu -- /home/yasha/.cache/dryft-starter-venv/bin/python -m unittest discover -s /mnt/c/Users/yasha/Desktop/GitHub/starter/tests -p test_engine_gpu.py -v
```
