"""Baselines: all six systems register and can be built.

We monkey-patch the LLM wrappers so `builder(cfg)` doesn't try real API calls.
"""
from __future__ import annotations

import pytest

from infoshield.baselines import SYSTEMS
from infoshield.config import load_config
from infoshield.trace import LLMCall

EXPECTED = {
    "undefended",
    "ibprotector",
    "smoothllm",
    "llm_guard",
    "perplexity_filter",
    "infoshield",
    "infoshield_azure",
}


def test_registry_populated():
    missing = EXPECTED - set(SYSTEMS)
    assert not missing, f"missing baselines: {missing}"


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def stub_llm(monkeypatch):
    """Prevent real API calls in the builder step (network happens on pipeline invoke, not build)."""
    def _stub(self, *args, **kwargs):
        return "[]", LLMCall(role="stub", provider="stub", model="stub")

    monkeypatch.setattr(
        "infoshield.llm.azure_openai.AzureOpenAIClient.chat",
        _stub,
    )
    monkeypatch.setattr(
        "infoshield.llm.anthropic.AnthropicClient.chat",
        _stub,
    )


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_baseline_builds(name, cfg, stub_llm):
    # perplexity_filter downloads gpt2 the first time — skip if offline / slow CI.
    if name == "perplexity_filter":
        pytest.skip("perplexity_filter downloads gpt2; covered by end-to-end smoke.")
    # infoshield_azure encodes the phrase bank via Azure at build time — requires network.
    if name == "infoshield_azure":
        pytest.skip("infoshield_azure requires Azure embedding API; covered by network test.")
    builder = SYSTEMS[name]
    pipeline = builder(cfg)
    assert callable(pipeline)
