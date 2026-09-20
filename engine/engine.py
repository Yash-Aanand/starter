"""BF16 Qwen3 inference with a static cache and captured decode steps."""

import torch
from transformers import AutoModelForCausalLM

from runtime import Model, Generation
from speculative import Speculative


class Engine:
    def __init__(self, model_path: str) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode():
            reference = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            ).eval().to("cuda:0")
            self.model = Model(reference)
            del reference
        self.state = None

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        """Yield exactly one greedy token per sequence per requested step."""
        if max_new_tokens <= 0:
            return
        shape = (len(input_ids), len(input_ids[0]), max_new_tokens)
        with torch.inference_mode():
            if self.state is None or self.state.shape != shape:
                # Only one workload is retained. Setup occurs during warmup.
                self.state = None
                if max_new_tokens > 1:
                    self.state = Speculative(self.model, shape, width=4 if shape[0] <= 4 else 2)
                else:
                    self.state = Generation(self.model, shape)
            state = self.state
            if max_new_tokens > 1:
                yield from state.generate(input_ids, max_new_tokens)
                return
            state.prefill(input_ids)
            yield state.tokens[:, 0].tolist()
            for _ in range(max_new_tokens - 1):
                state.graph.replay()
                yield state.tokens[:, 0].tolist()
