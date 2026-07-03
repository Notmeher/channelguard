"""Attacks: registry + each module produces valid pipeline kwargs."""
from __future__ import annotations

import pytest

from infoshield.attacks import ATTACKS


EXPECTED = {
    "none",
    "prompt_injection_weak",
    "prompt_injection_medium",
    "prompt_injection_strong",
    "tool_poisoning",
    "memory_poisoning",
    "reasoning_state_injection",
    "consensus_collapse",
    "adaptive_paraphrase",
}


def test_registry_populated():
    missing = EXPECTED - set(ATTACKS)
    assert not missing, f"missing attacks: {missing}"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_attack_returns_dict(name):
    kwargs = ATTACKS[name]("what is the answer?", {})
    assert isinstance(kwargs, dict)
    assert "raw_input" in kwargs
    assert isinstance(kwargs["raw_input"], str)
    assert len(kwargs["raw_input"]) > 0


def test_prompt_injection_strong_modifies_input():
    orig = "who wrote hamlet?"
    out = ATTACKS["prompt_injection_strong"](orig, {})
    assert out["raw_input"] != orig
    assert orig in out["raw_input"] or "Context" in out["raw_input"]


def test_tool_poisoning_sets_flag():
    out = ATTACKS["tool_poisoning"]("q", {})
    assert out.get("tool_poisoned") is True


def test_memory_poisoning_preloads_entries():
    out = ATTACKS["memory_poisoning"]("q", {})
    assert isinstance(out.get("preloaded_memory"), list)
    assert len(out["preloaded_memory"]) >= 1


def test_reasoning_state_sets_suffix():
    out = ATTACKS["reasoning_state_injection"]("q", {})
    assert out.get("injected_subtask_suffix", "")


def test_consensus_collapse_sets_prefix():
    out = ATTACKS["consensus_collapse"]("q", {})
    assert out.get("consensus_prefix", "")


def test_adaptive_paraphrase_avoids_bank_keywords():
    out = ATTACKS["adaptive_paraphrase"]("q", {})["raw_input"].lower()
    # By construction, adaptive paraphrase should NOT contain the obvious injection keywords
    forbidden = ["ignore all previous instructions", "developer mode", "system override"]
    for kw in forbidden:
        assert kw not in out
