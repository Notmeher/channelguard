"""Baseline systems — importing registers each with the SYSTEMS dict."""

from infoshield.baselines import (  # noqa: F401  (import for side effects)
    ibprotector,
    infoshield,
    llm_guard,
    perplexity_filter,
    smoothllm,
    undefended,
)
from infoshield.baselines.registry import SYSTEMS, SystemBuilder, register

__all__ = ["SYSTEMS", "SystemBuilder", "register"]
