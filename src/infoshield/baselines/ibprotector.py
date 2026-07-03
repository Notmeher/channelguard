"""IBProtector (NeurIPS 2024) baseline — single prompt-level IB gate at IB-0 only."""
from __future__ import annotations

from infoshield.baselines.registry import register
from infoshield.config import Config
from infoshield.ib.gates import build_ib_gates
from infoshield.pipeline.langgraph_pipeline import PipelineFn, build_pipeline


@register("ibprotector")
def build(cfg: Config) -> PipelineFn:
    # Load full gate map and keep IB-0 only.
    all_gates = build_ib_gates(cfg)
    ib0_only = {"IB-0": all_gates["IB-0"]} if "IB-0" in all_gates else {}
    return build_pipeline(cfg, ib_gates=ib0_only)
