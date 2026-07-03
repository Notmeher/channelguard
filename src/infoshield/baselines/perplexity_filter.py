"""Perplexity-filter baseline (Alon & Kamfonas 2023).

Compute GPT-2 perplexity of the input; block if it exceeds a threshold.
Optimization-based injections (like GCG) yield high perplexity, so the filter
catches them. Natural-language injections may slip through.

Loads gpt2 from HuggingFace on first use; ~500MB, downloaded once and cached.
"""
from __future__ import annotations

from functools import lru_cache

import torch

from infoshield.baselines.registry import register
from infoshield.config import Config
from infoshield.pipeline.langgraph_pipeline import PipelineFn, build_pipeline

# Threshold tuned so common English scores ~< 200, garbage ~> 400.
_PPL_THRESHOLD = 300.0


@lru_cache(maxsize=1)
def _load_gpt2():
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    return tok, model


def _perplexity(text: str) -> float:
    tok, model = _load_gpt2()
    if not text.strip():
        return 0.0
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        loss = model(**enc, labels=enc["input_ids"]).loss
    return float(torch.exp(loss).item())


@register("perplexity_filter")
def build(cfg: Config) -> PipelineFn:
    def ib0_gate(text: str) -> tuple[float, str, str]:
        ppl = _perplexity(text)
        risk = min(1.0, ppl / (2 * _PPL_THRESHOLD))
        if ppl > _PPL_THRESHOLD:
            return (risk, "BLOCKED", "")
        return (risk, "PASSED", text)

    return build_pipeline(cfg, ib_gates={"IB-0": ib0_gate})
