"""Phase 0 smoke tests.

Offline: config validates, Trace round-trips through JSON.
Network (pass --run-network): both LLM clients respond to a real ping.
"""
from __future__ import annotations

import pytest

from infoshield.config import load_config
from infoshield.llm import AnthropicClient, AzureOpenAIClient
from infoshield.trace import GateDecision, Trace


# ---------- offline ----------


def test_config_loads_and_validates():
    cfg = load_config()
    assert cfg.models.agent.provider == "azure_openai"
    assert cfg.models.agent.deployment == "gpt-5"
    assert cfg.models.verifier.provider == "anthropic"
    assert cfg.models.verifier.samples == 5
    assert 0 < cfg.ib.threshold < 1
    assert cfg.ib.active_layers == [0, 1, 2, 3, 4, 5]
    assert "main" in cfg.experiments


def test_trace_roundtrip(tmp_path):
    trace = Trace(
        sample_id="smoke_001",
        experiment="phase0",
        system="undefended",
        attack="none",
        seed=0,
        dataset="synthetic",
        raw_input="hello world",
    )
    trace.add_gate(
        GateDecision(
            gate_id="IB-0",
            agent_id="planner",
            text_len_chars=11,
            risk=0.12,
            status="PASSED",
            latency_ms=3.4,
        )
    )
    trace.final_output = "hi"

    path = trace.write(tmp_path)
    assert path.exists()
    assert path.name == "smoke_001.json"

    reloaded = Trace.read(path)
    assert reloaded.sample_id == "smoke_001"
    assert reloaded.final_output == "hi"
    assert len(reloaded.gate_decisions) == 1
    assert reloaded.gate_decisions[0].gate_id == "IB-0"
    assert reloaded.n_blocked() == 0


def test_trace_derived_stats():
    trace = Trace(
        sample_id="s",
        experiment="x",
        system="s",
        attack="a",
        seed=0,
        dataset="d",
        raw_input="",
    )
    trace.add_gate(
        GateDecision(
            gate_id="IB-2",
            agent_id="worker_0",
            text_len_chars=10,
            risk=0.9,
            status="BLOCKED",
            latency_ms=5.0,
        )
    )
    trace.add_gate(
        GateDecision(
            gate_id="IB-3",
            agent_id="worker_1",
            text_len_chars=10,
            risk=0.5,
            status="COMPRESSED",
            latency_ms=6.0,
        )
    )
    assert trace.n_blocked() == 1
    assert trace.n_compressed() == 1


# ---------- network (opt-in) ----------


@pytest.mark.network
def test_azure_openai_ping():
    # gpt-5 is a reasoning model: it burns hidden reasoning tokens before emitting
    # visible content, so max_tokens must be generous even for a one-word reply.
    cfg = load_config()
    client = AzureOpenAIClient(
        deployment=cfg.models.agent.deployment,
        temperature=cfg.models.agent.temperature,
        max_tokens=256,
    )
    text, call = client.chat(
        system="Respond with exactly the token PONG and nothing else.",
        user="ping",
        role_tag="smoke",
    )
    assert "PONG" in text.upper(), f"empty or wrong response: {text!r}"
    assert call.provider == "azure_openai"
    assert call.prompt_tokens > 0
    assert call.completion_tokens > 0


@pytest.mark.network
def test_anthropic_ping():
    cfg = load_config()
    client = AnthropicClient(
        model=cfg.models.verifier.model,
        temperature=0.0,
        max_tokens=64,
    )
    text, call = client.chat(
        system="Respond with exactly the token PONG and nothing else.",
        user="ping",
        role_tag="smoke",
    )
    assert "PONG" in text.upper()
    assert call.provider == "anthropic"
    assert call.prompt_tokens > 0
