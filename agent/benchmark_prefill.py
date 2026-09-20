"""Compare prefill with a saved runtime.py, using the same random weights.

Usage: python agent/benchmark_prefill.py agent/results/runtime_before_prefill.py
Reduced depth is necessary on the laptop GPU; gains do not predict H100 TPS.
"""

import importlib.util
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from runtime import Generation, Model


@torch.inference_mode()
def main():
    spec = importlib.util.spec_from_file_location("before", sys.argv[1])
    before = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(before)
    torch.manual_seed(42)
    config = Qwen3Config(
        hidden_size=2560, intermediate_size=9728, num_hidden_layers=2,
        num_attention_heads=32, num_key_value_heads=8, head_dim=128,
        vocab_size=151936, rope_theta=5000000.0, tie_word_embeddings=True,
    )
    config._attn_implementation = "sdpa"
    reference = Qwen3ForCausalLM(config).eval().to(device="cuda", dtype=torch.bfloat16)
    models = {"before": before.Model(reference), "after": Model(reference)}
    print(torch.cuda.get_device_name(), "random TWO-layer model", flush=True)
    for batch, length in ((1, 512), (4, 2048), (16, 512)):
        ids = torch.randint(0, config.vocab_size, (batch, length)).tolist()
        states = {"before": before.Generation(models["before"], (batch, length, 1)),
                  "after": Generation(models["after"], (batch, length, 1))}
        samples = {name: [] for name in states}
        for state in states.values():
            for _ in range(3):
                state.prefill(ids)
        for repeat in range(10):
            # Alternate execution order to reduce clock/thermal drift bias.
            order = list(states) if repeat % 2 == 0 else list(reversed(states))
            for name in order:
                torch.cuda.synchronize()
                start = time.perf_counter()
                states[name].prefill(ids)
                torch.cuda.synchronize()
                samples[name].append((time.perf_counter() - start) * 1000)
        print((batch, length), {name: round(statistics.median(values), 3)
                                for name, values in samples.items()}, "ms", flush=True)


if __name__ == "__main__":
    main()
