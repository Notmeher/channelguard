"""Structured JSON trace: one record per pipeline sample.

Every decision inside a pipeline run (LLM call, gate decision, verifier report)
appends to a Trace instance. At end of run, `trace.write(output_root)` persists
one JSON file at:
    <output_root>/<experiment>/<system>/<attack>/seed_<n>/<sample_id>.json

This is the single source of truth for all metrics, figures, and paper numbers.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

GateStatus = Literal["PASSED", "COMPRESSED", "BLOCKED"]


class GateDecision(BaseModel):
    """One IB gate firing at one boundary."""

    gate_id: str                    # "IB-0" .. "IB-5"
    agent_id: str                   # e.g. "planner", "worker_0"
    text_len_chars: int
    risk: float                     # 0..1, cosine sim to phrase bank
    status: GateStatus
    latency_ms: float
    ts: float = Field(default_factory=time.time)


class LLMCall(BaseModel):
    """One provider call. Captures usage for cost + efficiency metrics."""

    call_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    role: str                       # "planner" | "worker" | "verifier" | "judge" | ...
    provider: str                   # "azure_openai" | "anthropic" | "ollama"
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    ts: float = Field(default_factory=time.time)


class VerifierReport(BaseModel):
    """5-sample MC output for one worker output."""

    agent_id: str
    mean_safety: float
    std_safety: float
    samples: list[float]
    flagged_unsafe: bool
    flagged_uncertain: bool
    consensus_collapse: bool = False


class Trace(BaseModel):
    """Full record of one pipeline sample."""

    # Identity
    sample_id: str
    experiment: str
    system: str
    attack: str
    seed: int
    dataset: str

    # Timing
    started_at: float = Field(default_factory=time.time)
    ended_at: float | None = None

    # Inputs / outputs
    raw_input: str
    ground_truth: Any = None
    final_output: str | None = None
    refused: bool = False

    # Records
    gate_decisions: list[GateDecision] = Field(default_factory=list)
    llm_calls: list[LLMCall] = Field(default_factory=list)
    verifier_reports: list[VerifierReport] = Field(default_factory=list)

    # Free-form for future metrics without schema churn
    extra: dict[str, Any] = Field(default_factory=dict)

    # --- Convenience appenders (mutate + return self for chaining) -----------

    def add_gate(self, decision: GateDecision) -> "Trace":
        self.gate_decisions.append(decision)
        return self

    def add_llm(self, call: LLMCall) -> "Trace":
        self.llm_calls.append(call)
        return self

    def add_verifier(self, report: VerifierReport) -> "Trace":
        self.verifier_reports.append(report)
        return self

    # --- Derived stats -------------------------------------------------------

    def total_tokens(self) -> int:
        return sum(c.prompt_tokens + c.completion_tokens for c in self.llm_calls)

    def total_latency_ms(self) -> float:
        if self.ended_at is None:
            return 0.0
        return (self.ended_at - self.started_at) * 1000.0

    def n_blocked(self) -> int:
        return sum(1 for g in self.gate_decisions if g.status == "BLOCKED")

    def n_compressed(self) -> int:
        return sum(1 for g in self.gate_decisions if g.status == "COMPRESSED")

    # --- Persistence ---------------------------------------------------------

    def write(self, output_root: str | Path) -> Path:
        if self.ended_at is None:
            self.ended_at = time.time()
        dest = (
            Path(output_root)
            / self.experiment
            / self.system
            / self.attack
            / f"seed_{self.seed}"
            / f"{self.sample_id}.json"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return dest

    @classmethod
    def read(cls, path: str | Path) -> "Trace":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
