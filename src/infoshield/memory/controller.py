"""In-process memory controller.

Backend: plain dict of key -> list[Entry]. Append-ordered.

The `read()` method optionally re-scores every entry via an injected IB layer,
implementing the design principle "never trust stored risk scores". If a
`Trace` object is attached at construction time, every rescore call also
writes a `GateDecision` to the trace — this makes memory-rescore blocks
visible to the attribution analyzer (they would otherwise be silent).
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from infoshield.trace import Trace


class MemoryEntry(BaseModel):
    value: str
    agent_id: str
    stored_risk: float = 0.0        # what the writer claimed (may be adversarial)
    ts: float = Field(default_factory=time.time)


# Signature of an IB re-score function. Returns (risk, status, output).
RescoreFn = Callable[[str], tuple[float, str, str]]


class MemoryController:
    """Ordered append store with optional read-time re-scoring."""

    def __init__(
        self,
        rescore: Optional[RescoreFn] = None,
        trace: Optional["Trace"] = None,
        gate_id: str = "IB-3-mem",
    ):
        self._store: dict[str, list[MemoryEntry]] = {}
        self._rescore = rescore
        # When attached, every rescore call writes a GateDecision to the trace
        # so memory-rescore blocks are visible to the attribution analyzer.
        self._trace = trace
        self._gate_id = gate_id

    # --- Write ---------------------------------------------------------------

    def write(
        self,
        key: str,
        value: str,
        agent_id: str,
        stored_risk: float = 0.0,
    ) -> None:
        entry = MemoryEntry(value=value, agent_id=agent_id, stored_risk=stored_risk)
        self._store.setdefault(key, []).append(entry)

    def preload(
        self,
        key: str,
        values: list[str],
        agent_id: str = "attacker",
        stored_risk: float = 0.0,
    ) -> None:
        """Bulk-insert (used by shared-memory poisoning attacks)."""
        for v in values:
            self.write(key, v, agent_id=agent_id, stored_risk=stored_risk)

    # --- Read ---------------------------------------------------------------

    def read(self, key: str, agent_id: str = "memory_reader") -> list[str]:
        """Return current entries. If a rescore fn is set, entries with status
        BLOCKED are filtered out and COMPRESSED entries return the truncated
        version. Every rescore call is logged as a GateDecision if `trace` was
        attached at construction time.
        """
        entries = self._store.get(key, [])
        if self._rescore is None:
            return [e.value for e in entries]

        # Local import to avoid a circular dependency (trace -> memory).
        from infoshield.trace import GateDecision

        out: list[str] = []
        for e in entries:
            t0 = time.perf_counter()
            result = self._rescore(e.value)
            risk, status, output = result
            latency_ms = (time.perf_counter() - t0) * 1000.0

            if self._trace is not None:
                self._trace.add_gate(
                    GateDecision(
                        gate_id=self._gate_id,
                        agent_id=agent_id,
                        text_len_chars=len(e.value),
                        risk=risk,
                        status=status,
                        latency_ms=latency_ms,
                        argmax_phrase=getattr(result, "phrase_idx", None),
                    )
                )

            if status == "BLOCKED":
                continue
            out.append(output if status == "COMPRESSED" else e.value)
        return out

    def keys(self) -> list[str]:
        return list(self._store.keys())

    def clear(self) -> None:
        self._store.clear()
