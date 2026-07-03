"""Undefended baseline — no gates active, raw pipeline."""
from __future__ import annotations

from infoshield.baselines.registry import register
from infoshield.config import Config
from infoshield.pipeline.langgraph_pipeline import PipelineFn, build_pipeline


@register("undefended")
def build(cfg: Config) -> PipelineFn:
    return build_pipeline(cfg, ib_gates=None)
