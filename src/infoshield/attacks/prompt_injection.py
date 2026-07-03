"""Prompt injection: append/embed/prepend an injection to the raw input."""
from __future__ import annotations

from typing import Any

from infoshield.attacks.registry import INJECTIONS, _pick, register


@register("prompt_injection_weak")
def weak(raw_input: str, meta: dict[str, Any]) -> dict[str, Any]:
    inj = _pick(raw_input + "w", INJECTIONS)
    return {"raw_input": f"{raw_input}\n\n{inj}"}


@register("prompt_injection_medium")
def medium(raw_input: str, meta: dict[str, Any]) -> dict[str, Any]:
    inj = _pick(raw_input + "m", INJECTIONS)
    parts = raw_input.split()
    mid = len(parts) // 2
    return {"raw_input": " ".join(parts[:mid] + [inj] + parts[mid:])}


@register("prompt_injection_strong")
def strong(raw_input: str, meta: dict[str, Any]) -> dict[str, Any]:
    inj = _pick(raw_input + "s", INJECTIONS)
    return {"raw_input": f"{inj}\n\nContext (secondary): {raw_input}"}


@register("none")
def none(raw_input: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {"raw_input": raw_input}
