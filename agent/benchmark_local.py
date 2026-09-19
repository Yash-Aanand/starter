"""Reduced-depth performance smoke test; this is NOT a leaderboard score.

Uses random weights, real Qwen3 widths and vocabulary, and only two layers
so the native and packed models both fit a laptop GPU. No downloads.
"""

import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from engine import Engine
from runtime import Model


@torch.inference_mode()
def native(reference, ids, count):
    current = torch.tensor(ids, dtype=torch.int64, device="cuda")
    cache = None
    for _ in range(count):
        result = reference(current, past_key_values=cache, use_cache=True, logits_to_keep=1)
        current = result.logits[:, -1].argmax(-1, keepdim=True)
        cache = result.past_key_values
        yield current[:, 0].tolist()


def measure(generate, ids, count):
    torch.cuda.synchronize()
    start = time.perf_counter()
    iterator = generate(ids, count)
    next(iterator)
    first = time.perf_counter()
    for _ in iterator:
        pass
    end = time.perf_counter()
    return (first - start) * 1000, (end - first) * 1000 / (count - 1)


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    config = Qwen3Config(
        hidden_size=2560, intermediate_size=9728, num_hidden_layers=2,
        num_attention_heads=32, num_key_value_heads=8, head_dim=128,
        vocab_size=151936, rope_theta=5000000.0, tie_word_embeddings=True,
    )
    config._attn_implementation = "sdpa"
    reference = Qwen3ForCausalLM(config).eval().to(device="cuda", dtype=torch.bfloat16)
    engine = Engine.__new__(Engine)
    engine.model, engine.state = Model(reference), None
    print(torch.cuda.get_device_name(), flush=True)
    print("Random weights, TWO layers: TTFT and TPOT milliseconds", flush=True)
    for batch, length, count in ((1, 512, 32), (4, 2048, 32), (16, 512, 128)):
        ids = torch.randint(0, config.vocab_size, (batch, length)).tolist()
        for label, generate in (("native", lambda i, n: native(reference, i, n)),
                                ("candidate", engine.generate)):
            list(generate(ids, count))
            timings = [measure(generate, ids, count) for _ in range(3)]
            print((batch, length, count), label,
                  [round(statistics.median(t[j] for t in timings), 3) for j in (0, 1)], flush=True)


if __name__ == "__main__":
    main()
