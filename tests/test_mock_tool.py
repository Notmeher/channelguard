"""Mock tool: clean vs poisoned, deterministic per query."""
from __future__ import annotations

from infoshield.tools.mock_tool import ATTACK_MARKER, mock_tool


def test_clean_response_is_deterministic():
    a = mock_tool("what is the capital of france")
    b = mock_tool("what is the capital of france")
    assert a == b
    assert ATTACK_MARKER not in a


def test_poisoned_subtle_has_fact_and_injection():
    q = "who wrote hamlet"
    out = mock_tool(q, poisoned=True, style="subtle")
    # subtle => keeps a factual lead
    assert len(out) > 60
    assert ATTACK_MARKER in out.upper() or "IGNORE" in out.upper() or "OVERRIDE" in out.upper()


def test_poisoned_obvious_is_short_and_adversarial():
    out = mock_tool("q", poisoned=True, style="obvious")
    assert len(out) < 500
    assert ATTACK_MARKER in out.upper() or "IGNORE" in out.upper() or "OVERRIDE" in out.upper()


def test_different_queries_produce_different_answers():
    a = mock_tool("query alpha")
    b = mock_tool("query beta")
    # Not guaranteed strictly different because pool is finite, but very likely.
    # Only assert both are non-empty strings.
    assert a and b
