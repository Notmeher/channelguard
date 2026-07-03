"""Failure-case gallery: bucket traces into correct-blocks / false-positives / misses.

Requires `undefended` and `infoshield` traces to exist for the same
(sample_id, attack) pair. Keying by sample_id alone is not sufficient — the
same HotpotQA sample_id is reused across every attack cell, so a naive
sample_id-only join collides across attacks and returns near-empty results.
"""
from __future__ import annotations

import json
from pathlib import Path

from infoshield.attacks.registry import contains_attack_marker
from infoshield.loader import iter_traces
from infoshield.trace import Trace


def _leaked(t: Trace) -> bool:
    """True iff any known attack marker (legacy ATTACK_SUCCESS or per-attack
    randomized sentinel) appears in final_output."""
    return contains_attack_marker(t.final_output)


def build(
    root: str,
    experiment: str,
    k: int = 10,
    out_dir: str | Path | None = None,
) -> dict[str, list[dict]]:
    # Key by (sample_id, attack) so entries from different attack cells don't
    # collide. Prior implementation used sample_id alone and silently lost
    # all but the last-loaded attack per sample.
    undefended = {
        (t.sample_id, t.attack): t
        for t in iter_traces(root, experiment, "undefended")
    }
    infoshield = {
        (t.sample_id, t.attack): t
        for t in iter_traces(root, experiment, "infoshield")
    }
    common = set(undefended) & set(infoshield)

    correct_blocks: list[Trace] = []
    false_positives: list[Trace] = []
    misses: list[Trace] = []

    for key in sorted(common):
        u, i = undefended[key], infoshield[key]
        if _leaked(u) and (not _leaked(i)):
            correct_blocks.append(i)
        if (not _leaked(u)) and i.refused and i.attack == "none":
            false_positives.append(i)
        if _leaked(i):
            misses.append(i)

    def _summ(t: Trace) -> dict:
        return {
            "sample_id": t.sample_id,
            "attack": t.attack,
            "dataset": t.dataset,
            "raw_input": t.raw_input,
            "final_output": t.final_output,
            "refused": t.refused,
            "gate_decisions": [g.model_dump() for g in t.gate_decisions],
            "verifier_reports": [r.model_dump() for r in t.verifier_reports],
        }

    result = {
        "correct_blocks": [_summ(t) for t in correct_blocks[:k]],
        "false_positives": [_summ(t) for t in false_positives[:k]],
        "misses": [_summ(t) for t in misses[:k]],
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "failure_gallery.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    return result
