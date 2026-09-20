"""A/B against saved 878 tok/s runtime; local random two-layer model only."""

import statistics
from pathlib import Path
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from benchmark_local import load_module, captured_generate, measure
import runtime


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    config = Qwen3Config(hidden_size=2560, intermediate_size=9728, num_hidden_layers=2,
                         num_attention_heads=32, num_key_value_heads=8, head_dim=128,
                         vocab_size=151936, rope_theta=5000000.0, tie_word_embeddings=True)
    config._attn_implementation = "sdpa"
    reference = Qwen3ForCausalLM(config).eval().to(device="cuda",dtype=torch.bfloat16)
    results = Path(__file__).resolve().parent / "results"
    before = load_module("best_runtime",results / "best_runtime.py")
    before.choose_projections = load_module("best_linear",results / "best_linear.py").choose_projections
    before.decode_attention = load_module("best_attention",results / "best_attention.py").decode_attention
    old, new = before.Model(reference), runtime.Model(reference)
    generators = {"878-path": captured_generate(before,old), "packed-graphs": captured_generate(runtime,new)}
    print(torch.cuda.get_device_name(),"TWO random layers; timings are not H100 estimates",flush=True)
    for batch, prompt, count in ((1,512,32),(4,2048,32),(16,512,128)):
        ids = torch.randint(0,config.vocab_size,(batch,prompt)).tolist()
        for generate in generators.values():
            list(generate(ids,count))
        samples = {name:[] for name in generators}
        for repeat in range(6):
            order = list(generators) if repeat % 2 == 0 else list(reversed(generators))
            for name in order:
                samples[name].append(measure(generators[name],ids,count))
        print((batch,prompt,count),{name:[round(statistics.median(t[j] for t in values),3) for j in (0,1)] for name,values in samples.items()},"TTFT/TPOT ms",flush=True)
        print("plans",{key:(type(plan).__name__,getattr(plan,"config",None)) for key,plan in new.projection_plans[batch].items()},flush=True)


if __name__ == "__main__":
    main()
