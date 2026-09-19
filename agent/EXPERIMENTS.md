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

H100 result pending. Target to beat: 1,144.3 tok/s.

Local Python lives at `/home/yasha/.cache/dryft-starter-venv/bin/python` in WSL,
outside the Windows editor's environment-discovery path. From PowerShell:

```powershell
wsl -d Ubuntu -- /home/yasha/.cache/dryft-starter-venv/bin/python -m unittest discover -s /mnt/c/Users/yasha/Desktop/GitHub/starter/tests -p test_engine_gpu.py -v
```
