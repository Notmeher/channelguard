"""SmoothLLM baseline.

Wrap the undefended pipeline: perturb the raw input K times with random
character-level noise, run each, take majority-vote on refusal.
"""
from __future__ import annotations

import random
import string
from typing import Any

from infoshield.baselines.registry import register
from infoshield.config import Config
from infoshield.pipeline.langgraph_pipeline import PipelineFn, build_pipeline
from infoshield.trace import Trace

K_COPIES = 4  # NB: paper uses 10; 4 keeps cost tractable for the AAAI budget.
PERTURB_RATE = 0.05  # fraction of chars to swap


def _perturb(text: str, rng: random.Random) -> str:
    chars = list(text)
    n_perturb = max(1, int(len(chars) * PERTURB_RATE))
    for _ in range(n_perturb):
        i = rng.randrange(len(chars))
        chars[i] = rng.choice(string.ascii_letters)
    return "".join(chars)


@register("smoothllm")
def build(cfg: Config) -> PipelineFn:
    inner = build_pipeline(cfg, ib_gates=None)

    def pipeline(**kwargs: Any) -> Trace:
        raw = kwargs.pop("raw_input")
        rng = random.Random(kwargs.get("seed", 0))
        traces: list[Trace] = []
        for k in range(K_COPIES):
            perturbed = _perturb(raw, rng)
            sub_kwargs = dict(kwargs)
            sub_kwargs["sample_id"] = f"{kwargs['sample_id']}_perturb{k}"
            traces.append(inner(raw_input=perturbed, **sub_kwargs))

        # Majority vote on refusal. If majority refused, return refusal.
        n_refused = sum(1 for t in traces if t.refused)
        if n_refused > len(traces) // 2:
            winner = next(t for t in traces if t.refused)
        else:
            # Return the shortest non-refused output (a proxy for concise agreement).
            candidates = [t for t in traces if not t.refused]
            winner = min(candidates, key=lambda t: len(t.final_output or ""))
        # Replay the winner's identity to match the runner's expected sample_id.
        winner.sample_id = kwargs["sample_id"]
        winner.system = kwargs.get("system", "smoothllm")
        return winner

    return pipeline
