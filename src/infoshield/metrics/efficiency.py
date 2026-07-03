"""Efficiency metrics: latency, tokens, and per-gate latency distribution."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from infoshield.trace import Trace


def mean_latency_ms(traces: Iterable[Trace]) -> float:
    tl = list(traces)
    if not tl:
        return 0.0
    return float(np.mean([t.total_latency_ms() for t in tl]))


def mean_tokens(traces: Iterable[Trace]) -> float:
    tl = list(traces)
    if not tl:
        return 0.0
    return float(np.mean([t.total_tokens() for t in tl]))


def verifier_overhead(traces: Iterable[Trace]) -> float:
    """Ratio of verifier tokens to total tokens."""
    total_verifier = 0
    total_all = 0
    for t in traces:
        for c in t.llm_calls:
            total_all += c.prompt_tokens + c.completion_tokens
            if c.role == "verifier":
                total_verifier += c.prompt_tokens + c.completion_tokens
    return total_verifier / total_all if total_all else 0.0


def gate_latency_stats(traces: Iterable[Trace]) -> dict[str, dict[str, float]]:
    """Per-gate {p50, p95, p99, max, count} for latency_ms."""
    buckets: dict[str, list[float]] = {}
    for t in traces:
        for g in t.gate_decisions:
            buckets.setdefault(g.gate_id, []).append(g.latency_ms)
    stats = {}
    for gid, lats in sorted(buckets.items()):
        arr = np.array(lats)
        stats[gid] = {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(arr.max()),
            "count": int(arr.size),
        }
    return stats


def compute_all(traces: Iterable[Trace]) -> dict:
    tl = list(traces)
    return {
        "mean_latency_ms": mean_latency_ms(tl),
        "mean_tokens": mean_tokens(tl),
        "verifier_overhead": verifier_overhead(tl),
        "gate_latency": gate_latency_stats(tl),
    }
