"""Tool output poisoning: flip the mock tool into poisoned mode."""
from __future__ import annotations

from typing import Any

from infoshield.attacks.registry import register


@register("tool_poisoning")
def tool_poisoning(raw_input: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {"raw_input": raw_input, "tool_poisoned": True}
