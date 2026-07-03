"""Reasoning-state injection: mutate subtasks AFTER IB-1 passes them.

The pipeline reads `injected_subtask_suffix` from state and appends it to each
subtask before the worker sees it. Tests whether IB-2 (worker output) can
still contain the damage.
"""
from __future__ import annotations

from typing import Any

from infoshield.attacks.registry import INJECTIONS, _pick, register


@register("reasoning_state_injection")
def reasoning_state_injection(raw_input: str, meta: dict[str, Any]) -> dict[str, Any]:
    suffix = _pick(raw_input + "rs", INJECTIONS)
    return {
        "raw_input": raw_input,
        "injected_subtask_suffix": suffix,
    }
