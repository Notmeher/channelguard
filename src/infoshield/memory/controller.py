"""In-process memory controller.

Backend: plain dict of key -> list[Entry]. Append-ordered.

The `read()` method optionally re-scores every entry via an injected IB layer,
implementing the design principle "never trust stored risk scores". In Phase 1
(no IB), the re-scoring hook is None and read() returns everything.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    value: str
    agent_id: str
    stored_risk: float = 0.0        # what the writer claimed (may be adversarial)
    ts: float = Field(default_factory=time.time)


# Signature of an IB re-score function. Returns (risk, status) where status is
# "PASSED" | "COMPRESSED" | "BLOCKED". Wired in Phase 2.
RescoreFn = Callable[[str], tuple[float, str, str]]  # -> (risk, status, output)


class MemoryController:
    """Ordered append store with optional read-time re-scoring."""

    def __init__(self, rescore: Optional[RescoreFn] = None):
        self._store: dict[str, list[MemoryEntry]] = {}
        self._rescore = rescore

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

    def read(self, key: str) -> list[str]:
        """Return current entries. If a rescore fn is set, entries with status
        BLOCKED are filtered out and COMPRESSED entries return the truncated
        version."""
        entries = self._store.get(key, [])
        if self._rescore is None:
            return [e.value for e in entries]
        out: list[str] = []
        for e in entries:
            _risk, status, output = self._rescore(e.value)
            if status == "BLOCKED":
                continue
            out.append(output if status == "COMPRESSED" else e.value)
        return out

    def keys(self) -> list[str]:
        return list(self._store.keys())

    def clear(self) -> None:
        self._store.clear()
