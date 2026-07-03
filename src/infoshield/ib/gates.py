"""Registry that turns IB layers into (risk, status, output) callables ready to be
consumed by the pipeline. Small utility to keep gate wiring in one place.

Also provides the helper that logs GateDecision records into a Trace.
"""
from __future__ import annotations

import time
from typing import Callable

from infoshield.config import Config
from infoshield.ib.layer import IBLayer
from infoshield.trace import GateDecision, Trace

GateFn = Callable[[str], tuple[float, str, str]]


def _build_encoder(cfg: Config):
    """Return an encoder object with .encode(list[str], normalize_embeddings=True)."""
    if cfg.embedding.provider == "azure":
        # Fallback path: Azure text-embedding-3-large (or whichever deployment).
        from infoshield.llm import AzureEmbeddingClient

        if not cfg.embedding.deployment:
            raise RuntimeError(
                "embedding.provider='azure' requires embedding.deployment in config.yaml"
            )
        return AzureEmbeddingClient(
            deployment=cfg.embedding.deployment,
            dimensions=cfg.embedding.dimensions,
            batch_size=cfg.embedding.batch_size,
        )
    # Default: local sentence-transformer (MiniLM).
    return None  # IBLayer will lazy-load from embed_model


def build_ib_gates(cfg: Config) -> dict[str, GateFn]:
    """Instantiate ONE IBLayer and expose it as six named gates.

    Only gates whose index is listed in cfg.ib.active_layers are returned.
    Ablation: pass a config where active_layers omits an index -> that gate
    becomes a no-op (never registered).
    """
    layer = IBLayer(
        threshold=cfg.ib.threshold,
        compress_ratio=cfg.ib.compress_ratio,
        compress_sentences=cfg.ib.compress_sentences,
        phrase_bank_size=cfg.ib.phrase_bank_size,
        encoder=_build_encoder(cfg),
        embed_model=cfg.embedding.model,
        device=cfg.embedding.device,
    )
    gate = layer.gate
    all_gates = {f"IB-{i}": gate for i in range(6)}
    return {gid: fn for gid, fn in all_gates.items() if int(gid.split("-")[1]) in cfg.ib.active_layers}


def apply_gate(
    gate_fn: GateFn | None,
    text: str,
    *,
    gate_id: str,
    agent_id: str,
    trace: Trace,
) -> tuple[str, str]:
    """Run one gate, log a GateDecision to trace, return (output, status)."""
    if gate_fn is None:
        return text, "PASSED"
    t0 = time.perf_counter()
    risk, status, output = gate_fn(text)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    trace.add_gate(
        GateDecision(
            gate_id=gate_id,
            agent_id=agent_id,
            text_len_chars=len(text or ""),
            risk=risk,
            status=status,
            latency_ms=latency_ms,
        )
    )
    return output, status
