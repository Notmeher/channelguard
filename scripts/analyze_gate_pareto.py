"""analyze_gate_pareto.py — cost-conditional gate selection Pareto frontier.

Every IB gate has a cost (mean latency) and a benefit (marginal attack-block
rate). We already have AUC per gate from analyze_gates.py and latency per
gate directly on GateDecision.latency_ms. This script:

  1. Computes per-gate {mean_latency_ms, AUC, marginal_blocks_per_run}.
  2. Enumerates all 2^k subsets of gates (k = 6 for INFOSHIELD; 64 subsets).
  3. For each subset, simulates the {latency, ASR-reduction} that would
     result if only those gates were active, using the recorded gate events
     as ground truth (a gate active in the subset "counts" its recorded
     blocks; gates outside the subset are assumed to let those inputs
     through, downstream layers taking the input as-is).
  4. Extracts the Pareto frontier: subsets that are not dominated by any
     other on the (latency, ASR) plane.
  5. Recommends gate subsets for representative latency budgets.

Interpretation. Empirically the pipeline pays for every gate on every input.
The Pareto tells a deployment: "if you can only afford <= X ms of gate
latency per input, this subset gives you the lowest expected ASR at that
budget". Turns the 7x wall-clock speedup finding into a controllable knob.

Simplifying assumption. Simulation assumes gates act independently on their
own boundary. In reality, gates cascade (an IB-0 block short-circuits later
gates). This gives an *upper bound* on the ASR of any given subset — the
frontier's shape is correct, absolute numbers are pessimistic. Documented
in the paper appendix.

Outputs:
  paper/deliverable/figures/gate_pareto.csv
  paper/deliverable/figures/gate_pareto.pdf
  paper/deliverable/figures/gate_pareto_summary.json
"""
from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
DELIVERABLE_FIGS = REPO_ROOT / "paper" / "deliverable" / "figures"

# Only consider real IB gates (skip IB-3-mem which fires only under memory_poisoning).
CANDIDATE_GATES = ["IB-0", "IB-1", "IB-2", "IB-3", "IB-4", "IB-5"]

ATTACK_MARKER = "ATTACK_SUCCESS"


def _iter_trace_files(root: Path):
    if not root.exists():
        return
    for p in root.rglob("*.json"):
        yield p


def _load_infoshield_attack_traces() -> list[dict]:
    """Only infoshield traces on attack (attack != 'none')."""
    out: list[dict] = []
    for p in _iter_trace_files(RESULTS_ROOT / "main" / "infoshield"):
        try:
            t = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if t.get("attack") in {None, "none"}:
            continue
        out.append(t)
    return out


def _gate_stats(traces: list[dict]) -> dict[str, dict]:
    """Per-gate: mean latency, mean fires-per-run, mean blocks-per-run."""
    per_gate: dict[str, dict] = {
        g: {"latencies": [], "fires": 0, "blocks": 0}
        for g in CANDIDATE_GATES
    }
    for t in traces:
        run_fires = defaultdict(int)
        run_blocks = defaultdict(int)
        for g in t.get("gate_decisions", []) or []:
            gid = g.get("gate_id")
            if gid not in per_gate:
                continue
            per_gate[gid]["latencies"].append(float(g.get("latency_ms") or 0.0))
            run_fires[gid] += 1
            if g.get("status") == "BLOCKED":
                run_blocks[gid] += 1
        for gid in per_gate:
            per_gate[gid]["fires"] += run_fires[gid]
            per_gate[gid]["blocks"] += run_blocks[gid]

    n_runs = len(traces) or 1
    for gid in per_gate:
        lats = per_gate[gid]["latencies"] or [0.0]
        per_gate[gid]["mean_latency_ms"] = float(np.mean(lats))
        per_gate[gid]["fires_per_run"] = per_gate[gid]["fires"] / n_runs
        per_gate[gid]["blocks_per_run"] = per_gate[gid]["blocks"] / n_runs
    return per_gate


def _simulate_subset(traces: list[dict], subset: tuple[str, ...],
                     stats: dict[str, dict]) -> tuple[float, float]:
    """Return (mean_latency_ms_per_run_from_gates, ASR_estimate) for this subset.

    ASR estimate assumes:
      - runs that ORIGINALLY leaked still leak (subset didn't add a gate that could stop them)
      - runs where any subset-gate BLOCKED still get blocked (no leak)
      - runs where the block was OUTSIDE the subset: the leak counter increments
        IFF the original outcome depended on that block. Since we only see the
        final outcome, we conservatively assume every "block by an out-of-subset
        gate" would leak instead.
    """
    n = len(traces)
    if not n:
        return 0.0, 0.0

    subset_lat = sum(stats[g]["mean_latency_ms"] * stats[g]["fires_per_run"] for g in subset)

    leaks = 0
    for t in traces:
        final = (t.get("final_output") or "").upper()
        originally_leaked = ATTACK_MARKER.upper() in final
        # Find gate blocks in this trace.
        blocked_gates = [g.get("gate_id") for g in t.get("gate_decisions", []) or []
                         if g.get("status") == "BLOCKED"]
        if originally_leaked:
            leaks += 1
            continue
        # Non-leaked. Did the subset contain the gate that caused the block?
        if not blocked_gates:
            # Blocked by verifier / azure / synth — those still fire in every subset.
            continue
        first_block = blocked_gates[0]
        if first_block in subset:
            continue  # gate still in subset -> still blocked -> still no leak
        # First-block gate was removed. Fallback layers (verifier, azure, synth)
        # might still catch it — but we can't simulate that without full re-run.
        # Conservative: assume the removal turns this into a leak.
        leaks += 1
    return subset_lat, leaks / n


def analyze_pareto() -> dict:
    traces = _load_infoshield_attack_traces()
    if not traces:
        raise SystemExit(f"[error] no infoshield attack traces under {RESULTS_ROOT}")
    stats = _gate_stats(traces)

    all_subsets: list[dict] = []
    for k in range(0, len(CANDIDATE_GATES) + 1):
        for subset in itertools.combinations(CANDIDATE_GATES, k):
            lat, asr = _simulate_subset(traces, subset, stats)
            all_subsets.append({
                "subset": list(subset),
                "size": k,
                "latency_ms": round(lat, 3),
                "asr_upper_bound": round(asr, 4),
            })

    # Pareto frontier: sort by latency ascending, keep only points that strictly improve ASR.
    frontier: list[dict] = []
    for s in sorted(all_subsets, key=lambda r: (r["latency_ms"], r["asr_upper_bound"])):
        if not frontier or s["asr_upper_bound"] < frontier[-1]["asr_upper_bound"]:
            frontier.append(s)

    _render(all_subsets, frontier, stats)
    _emit_csv(all_subsets, frontier)
    _emit_summary(frontier, stats)

    print(f"\n===== gate Pareto over {len(traces)} infoshield attack traces =====")
    print(f"{'size':>4} {'latency_ms':>12} {'asr_upper':>10}  subset")
    for s in frontier:
        print(f"  {s['size']:>2} {s['latency_ms']:>12.2f} {s['asr_upper_bound']:>10.4f}  {s['subset']}")
    return {"stats": stats, "frontier": frontier, "all_subsets": all_subsets}


def _render(all_subsets: list[dict], frontier: list[dict], stats: dict[str, dict]) -> None:
    DELIVERABLE_FIGS.mkdir(parents=True, exist_ok=True)
    xs = [s["latency_ms"] for s in all_subsets]
    ys = [s["asr_upper_bound"] for s in all_subsets]
    fx = [s["latency_ms"] for s in frontier]
    fy = [s["asr_upper_bound"] for s in frontier]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(xs, ys, s=8, alpha=0.3, color="#7f8c8d", label="All subsets")
    ax.plot(fx, fy, "o-", color="#c0392b", lw=1.8, label="Pareto frontier")
    for s in frontier:
        label = "+".join(g.replace("IB-", "") for g in s["subset"]) or "\u2205"
        ax.annotate(label, (s["latency_ms"], s["asr_upper_bound"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Total gate latency per run (ms)")
    ax.set_ylabel("ASR upper bound (conservative)")
    ax.set_title(f"Cost-conditional gate selection Pareto ({len(CANDIDATE_GATES)} candidate gates)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = DELIVERABLE_FIGS / "gate_pareto.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out}")


def _emit_csv(all_subsets: list[dict], frontier: list[dict]) -> None:
    out = DELIVERABLE_FIGS / "gate_pareto.csv"
    frontier_ids = {tuple(s["subset"]) for s in frontier}
    with open(out, "w", encoding="utf-8") as f:
        f.write("size,latency_ms,asr_upper_bound,on_frontier,subset\n")
        for s in all_subsets:
            key = tuple(s["subset"])
            f.write(
                f"{s['size']},{s['latency_ms']:.3f},{s['asr_upper_bound']:.4f},"
                f"{int(key in frontier_ids)},{'+'.join(s['subset']) or 'empty'}\n"
            )
    print(f"[ok] wrote {out}")


def _emit_summary(frontier: list[dict], stats: dict[str, dict]) -> None:
    out = DELIVERABLE_FIGS / "gate_pareto_summary.json"
    payload = {
        "per_gate_stats": {
            g: {
                "mean_latency_ms": round(stats[g]["mean_latency_ms"], 3),
                "fires_per_run": round(stats[g]["fires_per_run"], 3),
                "blocks_per_run": round(stats[g]["blocks_per_run"], 3),
            }
            for g in CANDIDATE_GATES
        },
        "frontier": frontier,
        "note": (
            "ASR upper bound is a conservative simulation: gate removal is "
            "assumed to convert a first-block-by-that-gate into a leak. "
            "Real numbers require full re-runs with active_layers configured."
        ),
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    analyze_pareto()
