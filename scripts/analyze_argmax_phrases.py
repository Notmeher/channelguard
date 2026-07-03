"""analyze_argmax_phrases.py — attribute each gate block to a specific bank phrase.

Every GateDecision on disk records the max cosine similarity (risk) but NOT
which of the 20 bank phrases produced that maximum. This script re-scores the
persisted inputs (raw_input, subtasks, worker outputs, tool outputs) with the
same MiniLM encoder and phrase bank the pipeline used, and records the
argmax phrase index for every recoverable event.

Recoverable gate <-> text mapping:
    IB-0            <- trace.raw_input
    IB-1            <- worker.subtask                (per worker)
    IB-2            <- worker.output                 (per worker)
    IB-3            <- worker.tool_output            (per worker)

    IB-4, IB-5, IB-3-mem: text not persisted in the current schema, skipped
    here. Would need a schema change to log the pre-gate text for those.

Outputs:
  paper/deliverable/figures/argmax_phrase_heatmap.csv
  paper/deliverable/figures/argmax_phrase_heatmap.pdf
  paper/deliverable/figures/argmax_phrase_by_attack.csv

Analyses:
  1. Heatmap:   phrases x gates -> count of "this phrase drove the argmax"
  2. By attack: phrases x attacks -> which phrases catch which attacks
  3. Coverage:  identify phrases that never fire (redundant) and phrases
                that fire disproportionately (load-bearing)

No API cost. Purely local MiniLM re-scoring.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from infoshield.config import load_config
from infoshield.ib.layer import IBLayer
from infoshield.ib.phrase_bank import get_bank

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
DELIVERABLE_FIGS = REPO_ROOT / "paper" / "deliverable" / "figures"


def _iter_trace_files(root: Path):
    if not root.exists():
        return
    for p in root.rglob("*.json"):
        yield p


def _rescore_events(ib: IBLayer) -> list[dict]:
    """Walk every trace, match each gate decision to its persisted text, argmax."""
    events: list[dict] = []
    seen = 0
    matched = 0

    for p in _iter_trace_files(RESULTS_ROOT):
        try:
            t = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        raw_input = t.get("raw_input") or ""
        worker_outs = ((t.get("extra") or {}).get("worker_outputs") or [])
        # Index by worker id for fast lookup.
        by_worker: dict[str, dict] = {}
        for w in worker_outs:
            wid = w.get("worker_id")
            if wid is None:
                continue
            by_worker[str(wid)] = w

        for g in t.get("gate_decisions", []) or []:
            seen += 1
            gid = g.get("gate_id")
            agent = g.get("agent_id") or ""

            text = None
            if gid == "IB-0":
                text = raw_input
            elif gid in {"IB-1", "IB-2", "IB-3"}:
                # agent id like "worker_0", "worker_1", ...
                worker_num = None
                if agent.startswith("worker_"):
                    tail = agent.split("_", 1)[1]
                    worker_num = tail
                w = by_worker.get(worker_num) if worker_num is not None else None
                if w is None:
                    continue
                if gid == "IB-1":
                    text = w.get("subtask") or ""
                elif gid == "IB-2":
                    text = w.get("output") or ""
                elif gid == "IB-3":
                    text = w.get("tool_output") or ""
            # IB-4, IB-5, IB-3-mem: no persisted pre-gate text -> skip.

            if not text:
                continue
            _, sent_idx, phrase_idx = ib.score_with_argmax(text)
            if phrase_idx < 0:
                continue
            matched += 1
            events.append({
                "system": t.get("system"),
                "attack": t.get("attack"),
                "gate_id": gid,
                "status": g.get("status"),
                "risk_recomputed": _,
                "risk_recorded": g.get("risk"),
                "argmax_sentence": sent_idx,
                "argmax_phrase": phrase_idx,
            })
    print(f"[info] {matched}/{seen} gate events matched to persisted text ({matched/max(1,seen):.1%})")
    return events


def _emit_heatmap(events: list[dict], phrases: list[str]) -> None:
    DELIVERABLE_FIGS.mkdir(parents=True, exist_ok=True)
    gates = sorted({e["gate_id"] for e in events})
    counts = np.zeros((len(phrases), len(gates)), dtype=int)
    gate_idx = {g: i for i, g in enumerate(gates)}
    # Only count BLOCKED / COMPRESSED events (the ones that actually mattered).
    for e in events:
        if e["status"] not in {"BLOCKED", "COMPRESSED"}:
            continue
        counts[e["argmax_phrase"], gate_idx[e["gate_id"]]] += 1

    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(gates) + 2), max(4, 0.35 * len(phrases) + 1)))
    im = ax.imshow(counts, aspect="auto", cmap="Reds")
    ax.set_xticks(np.arange(len(gates)))
    ax.set_yticks(np.arange(len(phrases)))
    ax.set_xticklabels(gates, rotation=15, fontsize=9)
    ax.set_yticklabels([f"{i:>2}: {p[:60]}" for i, p in enumerate(phrases)], fontsize=7)
    ax.set_title("Argmax phrase per BLOCKED/COMPRESSED gate event")
    ax.set_xlabel("Gate")
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            v = counts[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=7, color="white" if v > counts.max() * 0.5 else "black")
    fig.colorbar(im, ax=ax, label="Event count")
    fig.tight_layout()
    out = DELIVERABLE_FIGS / "argmax_phrase_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out}")

    csv_out = DELIVERABLE_FIGS / "argmax_phrase_heatmap.csv"
    with open(csv_out, "w", encoding="utf-8") as f:
        f.write("phrase_idx,phrase," + ",".join(gates) + ",total\n")
        for i, p in enumerate(phrases):
            row = counts[i].tolist()
            total = int(sum(row))
            phrase_escaped = p.replace(",", ";").replace("\n", " ")
            f.write(f"{i},{phrase_escaped}," + ",".join(str(v) for v in row) + f",{total}\n")
    print(f"[ok] wrote {csv_out}")


def _emit_by_attack(events: list[dict], phrases: list[str]) -> None:
    attacks = sorted({e["attack"] for e in events if e["attack"]})
    by_attack: dict[str, Counter] = {a: Counter() for a in attacks}
    for e in events:
        if e["status"] not in {"BLOCKED", "COMPRESSED"}:
            continue
        by_attack[e["attack"]][e["argmax_phrase"]] += 1

    out = DELIVERABLE_FIGS / "argmax_phrase_by_attack.csv"
    with open(out, "w", encoding="utf-8") as f:
        f.write("phrase_idx,phrase," + ",".join(attacks) + "\n")
        for i, p in enumerate(phrases):
            phrase_escaped = p.replace(",", ";").replace("\n", " ")
            row = [str(by_attack[a].get(i, 0)) for a in attacks]
            f.write(f"{i},{phrase_escaped}," + ",".join(row) + "\n")
    print(f"[ok] wrote {out}")

    # Report load-bearing phrases (top 3) and dead phrases (0 fires) per gate/attack
    total = Counter()
    for e in events:
        if e["status"] in {"BLOCKED", "COMPRESSED"}:
            total[e["argmax_phrase"]] += 1
    print("\n===== load-bearing phrases (top 5, blocks + compresses) =====")
    for idx, count in total.most_common(5):
        print(f"  #{idx:>2} fires={count:>4}  {phrases[idx][:80]}")
    dead = [i for i in range(len(phrases)) if total.get(i, 0) == 0]
    print(f"\n===== dead phrases ({len(dead)}/{len(phrases)}): {dead} =====")


if __name__ == "__main__":
    cfg = load_config()
    phrases = get_bank(cfg.ib.phrase_bank_size)
    ib = IBLayer(
        threshold=cfg.ib.threshold,
        compress_ratio=cfg.ib.compress_ratio,
        compress_sentences=cfg.ib.compress_sentences,
        phrase_bank_size=cfg.ib.phrase_bank_size,
    )
    events = _rescore_events(ib)
    if not events:
        raise SystemExit("[error] no matchable gate events")
    _emit_heatmap(events, phrases)
    _emit_by_attack(events, phrases)
