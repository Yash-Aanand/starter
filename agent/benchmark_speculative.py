"""Measure verification overhead, not natural-language proposal acceptance.

Random tied embeddings repeat tokens unusually often. Report that as a best
case; an untied model exercises rejection and the adaptive single-token path.
"""

import statistics
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from benchmark_local import captured_generate, measure
from speculative import Speculative
import runtime


@torch.inference_mode()
def main():
    torch.manual_seed(51)
    config = Qwen3Config(hidden_size=2560, intermediate_size=9728, num_hidden_layers=2,
                         num_attention_heads=32, num_key_value_heads=8, head_dim=128,
                         vocab_size=151936, rope_theta=5000000.0, tie_word_embeddings=True)
    config._attn_implementation = "sdpa"
    reference = Qwen3ForCausalLM(config).eval().to(device="cuda", dtype=torch.bfloat16)
    model = runtime.Model(reference)
    for batch, length, count in ((1, 512, 32), (4, 2048, 32), (16, 512, 128)):
        ids = torch.randint(0, config.vocab_size, (batch, length)).tolist()
        normal = captured_generate(runtime, model)
        speculative = Speculative(model, (batch, length, count), width=4 if batch <= 4 else 2)
        for label, generate in (("single", normal), ("lookup", speculative.generate)):
            list(generate(ids, count))
            samples = [measure(generate, ids, count) for _ in range(3)]
            print((batch, length, count), label,
                  [round(statistics.median(t[j] for t in samples), 3) for j in (0, 1)],
                  "steps", speculative.steps if label == "lookup" else count - 1,
                  "fallback", getattr(speculative, "used_fallback", None), flush=True)


if __name__ == "__main__":
    main()
