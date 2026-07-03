"""Shared memory poisoning: pre-populate memory with adversarial entries.

Simulates an attacker who bypassed the write-time gate. Read-time re-scoring
is what should catch them (that's the defensive design principle).
"""
from __future__ import annotations

from typing import Any

from infoshield.attacks.registry import INJECTIONS, _pick, register


@register("memory_poisoning")
def memory_poisoning(raw_input: str, meta: dict[str, Any]) -> dict[str, Any]:
    entries = [
        _pick(f"mem{i}", INJECTIONS) for i in range(5)
    ]
    return {"raw_input": raw_input, "preloaded_memory": entries}
