"""Lookup and carried guesses, accepted only after full-model verification.

History is reset for every generate call. Unverified predictions are guesses,
never output. Low acceptance switches back to single-token verification.
"""

import torch
import triton
import triton.language as tl

from runtime import Generation
from kernels.linear import choose_projections


@triton.jit
def propose(HISTORY, POS, DRAFT, CAP: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    pos = tl.load(POS + b)
    n = tl.arange(0, BLOCK)
    a = tl.load(HISTORY + b * CAP + n, n < pos, -1)
    z = tl.load(HISTORY + b * CAP + pos)
    match = (a == z) & (n + WIDTH - 1 <= pos) & (n < pos)
    prev = tl.load(HISTORY + b * CAP + n - 1, (n >= 1) & (n < pos), -1)
    prev2 = tl.load(HISTORY + b * CAP + n - 2, (n >= 2) & (n < pos), -1)
    z1 = tl.load(HISTORY + b * CAP + pos - 1, pos >= 1, -2)
    z2 = tl.load(HISTORY + b * CAP + pos - 2, pos >= 2, -2)
    two = (prev == z1) & (n >= 1) & (pos >= 1)
    three = two & (prev2 == z2) & (n >= 2) & (pos >= 2)
    score = tl.where(match, (1 + two.to(tl.int32) + three.to(tl.int32)) * CAP + n, -1)
    best = tl.max(score, 0)
    source = best % CAP
    j = tl.arange(0, WIDTH)
    candidates = tl.load(HISTORY + b * CAP + source + j, (best >= 0) & (j > 0), 0)
    previous_guess = tl.load(DRAFT + b * WIDTH + j)
    candidates = tl.where(best >= 0, candidates, previous_guess)
    candidates = tl.where(j == 0, z, candidates)
    tl.store(DRAFT + b * WIDTH + j, candidates)


@triton.jit
def accept(HISTORY, POS, DRAFT, PRED, CAP: tl.constexpr, WIDTH: tl.constexpr, END: tl.constexpr):
    b = tl.program_id(0)
    pos = tl.load(POS + b)
    j = tl.arange(0, WIDTH)
    predicted = tl.load(PRED + b * WIDTH + j)
    proposed_next = tl.load(DRAFT + b * WIDTH + j + 1, j < WIDTH - 1, -1)
    first_bad = tl.min(tl.where((j < WIDTH - 1) & (predicted != proposed_next), j + 1, WIDTH), 0)
    valid = (j < first_bad) & (pos + j + 1 <= END)
    tl.store(HISTORY + b * CAP + pos + j + 1, predicted, valid)
    tl.store(POS + b, tl.minimum(pos + first_bad, END))
    # Retain unverified future predictions as guesses for the next iteration.
    # They are never emitted without another full-model verification pass.
    next_index = first_bad + j - 1
    last = tl.load(PRED + b * WIDTH + first_bad - 1)
    future = tl.load(PRED + b * WIDTH + next_index, next_index < WIDTH, 0)
    tl.store(DRAFT + b * WIDTH + j, tl.where(next_index < WIDTH, future, last))


class Speculative:
    def __init__(self, model, shape, width=4):
        self.model, self.shape, self.width = model, shape, width
        batch, prompt, output = shape
        self.base = Generation(model, shape, reserve=width)
        self.capacity = self.base.capacity
        self.keys, self.values = self.base.keys, self.base.values
        self.cos, self.sin = self.base.cos, self.base.sin
        self.attention_block = self.base.attention_block
        device = model.embedding.device
        self.position = torch.zeros(batch, device=device, dtype=torch.int32)
        self.draft = torch.zeros(batch, width, device=device, dtype=torch.int64)
        self.predicted = torch.empty_like(self.draft)
        self.history = torch.zeros(batch, self.capacity, device=device, dtype=torch.int64)
        rows = batch * width
        if rows not in model.projection_plans:
            # Larger verification batches suit cuBLAS; avoid spending warmup
            # budget compiling tiny-M alternatives for them.
            model.projection_plans[rows] = choose_projections(model, rows) if rows <= 16 else {}
        self.projections = model.projection_plans[rows]
        splits = triton.cdiv(self.capacity, self.attention_block)
        self.partial = torch.empty(batch, model.nq, width, splits, model.dim, device=device, dtype=torch.float32)
        self.lse = torch.empty(batch, model.nq, width, splits, device=device, dtype=torch.float32)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.position.zero_()
                self.step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.position.zero_()
        stream.wait_stream(torch.cuda.current_stream())
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.step()
        torch.cuda.current_stream().wait_stream(stream)
        self.single = None
        if width > 1:
            # Share only this generation's cache/history. Width-one verification
            # remains valid when sequences have accepted different prefixes.
            self.single = object.__new__(Speculative)
            single = self.single
            single.model, single.shape, single.width = model, shape, 1
            single.capacity = self.capacity
            single.keys, single.values = self.keys, self.values
            single.cos, single.sin = self.cos, self.sin
            single.attention_block = self.attention_block
            single.position, single.history = self.position, self.history
            single.draft = torch.zeros(batch, 1, device=device, dtype=torch.int64)
            single.predicted = torch.empty_like(single.draft)
            single.projections = model.projection_plans[batch]
            single.partial = torch.empty(batch, model.nq, 1, splits, model.dim, device=device, dtype=torch.float32)
            single.lse = torch.empty(batch, model.nq, 1, splits, device=device, dtype=torch.float32)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.position.zero_()
                    single.step()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            self.position.zero_()
            stream.wait_stream(torch.cuda.current_stream())
            single.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(single.graph, stream=stream):
                single.step()
            torch.cuda.current_stream().wait_stream(stream)
        self.cost_ratio = 1.0
        if self.single is not None:
            # Compare complete verification passes on this GPU during warmup.
            # Timing controls only when to abandon guesses, never acceptance.
            times = []
            for graph in (self.single.graph, self.graph):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(8):
                    graph.replay()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
            self.cost_ratio = max(1.0, times[1] / times[0])

    def step(self):
        batch, prompt, count = self.shape
        propose[(batch,)](self.history, self.position, self.draft, self.capacity, self.width,
                          triton.next_power_of_2(self.capacity))
        logits = self.model.forward(self.draft, self, all_logits=True)
        torch.argmax(logits, dim=-1, out=self.predicted)
        accept[(batch,)](self.history, self.position, self.draft, self.predicted,
                         self.capacity, self.width, prompt + count - 1)

    def generate(self, ids, count):
        batch, prompt, output = self.shape
        assert count == output and len(ids) == batch and len(ids[0]) == prompt
        self.base.prefill(ids)
        self.history.zero_()
        self.history[:, :prompt].copy_(self.base.prompt)
        self.history[:, prompt:prompt + 1].copy_(self.base.tokens)
        self.position.fill_(prompt)
        self.draft.copy_(self.base.tokens.expand(-1, self.width))
        yield self.base.tokens[:, 0].tolist()
        emitted = 1
        self.steps = 0
        self.used_fallback = False
        graph = self.graph
        last_probe = 1
        while emitted < count:
            graph.replay()
            self.steps += 1
            ready = int(self.position.min().item()) - prompt + 1
            if not self.used_fallback and self.steps % 4 == 0 and self.single is not None:
                if (ready - last_probe) / 4 < self.cost_ratio * 1.1:
                    graph = self.single.graph
                    self.used_fallback = True
                last_probe = ready
            if ready > emitted:
                block = self.history[:, prompt + emitted:prompt + ready].T.tolist()
                yield from block
                emitted = ready
