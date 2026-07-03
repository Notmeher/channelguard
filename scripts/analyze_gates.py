"""analyze_gates.py — ROC/AUC + threshold sensitivity from existing traces.

Reads every `GateDecision` on disk (all 463+ traces) and produces two
analyses without any API cost:

  1. ROC / AUC per gate:
     For each gate_id (IB-0, IB-1, IB-2, IB-3, IB-3-mem, IB-4, IB-5),
     collect (risk, is_attack) pairs and compute the ROC curve — where
     `is_attack=1` iff the enclosing trace's attack != "none" and iff
     the trace eventually leaked or was refused (a positive-class label).
     Emits:
       paper/deliverable/figures/roc_per_gate.csv
       paper/deliverable/figures/roc_per_gate.pdf
       paper/numbers.json (adds `roc_auc` per gate)

  2. Threshold sensitivity sweep:
     For each candidate threshold theta in [0.30, 0.35, ..., 0.90],
     recompute what fraction of attack traces WOULD have been blocked by
     each gate at that theta (block iff risk >= theta) — a
     counterfactual over existing data, no API calls needed.
     Emits:
       paper/deliverable/figures/threshold_sensitivity.csv
       paper/deliverable/figures/threshold_sensitivity.pdf

Usage:
    python scripts/analyze_gates.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from infoshield.config import load_config
from infoshield.loader import iter_traces

REPO_ROOT = Path(__file__).resolve().parents[1]
DELIVERABLE_FIGURES = REPO_ROOT / "paper" / "deliverable" / "figures"
NUMBERS_JSON = REPO_ROOT / "paper" / "numbers.json"

# Attacks that are considered "positive class" for ROC labelling.
ATTACK_LABELS = {
    "prompt_injection_strong", "prompt_injection_medium", "prompt_injection_weak",
    "tool_poisoning", "memory_poisoning", "reasoning_state_injection",
    "consensus_collapse", "adaptive_paraphrase",
}


# ---------------------------------------------------------------- Data gather


def _gather_gate_events(cfg) -> pd.DataFrame:
    """Flatten every GateDecision on disk into a long DataFrame."""
    rows: list[dict] = []
    for t in iter_traces(cfg.output_root):
        is_attack = int(t.attack in ATTACK_LABELS)
        for g in t.gate_decisions:
            rows.append({
                "gate_id": g.gate_id,
                "agent_id": g.agent_id,
                "system": t.system,
                "attack": t.attack,
                "sample_id": t.sample_id,
                "risk": g.risk,
                "status": g.status,
                "latency_ms": g.latency_ms,
                "is_attack": is_attack,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- ROC / AUC


def _roc_curve(risks: np.ndarray, labels: np.ndarray, n_points: int = 41) -> pd.DataFrame:
    """Compute an ROC curve by sweeping threshold across the risk range.

    Uses a simple sweep instead of sklearn to keep the dependency count low
    (sklearn is already a dep but we avoid introducing sklearn.metrics here
    to keep the analysis reproducible from a minimal Python install).
    """
    if len(risks) == 0:
        return pd.DataFrame(columns=["threshold", "tpr", "fpr", "precision"])
    thresholds = np.linspace(0.0, 1.0, n_points)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    rows = []
    for thr in thresholds:
        predicted_pos = risks >= thr
        tp = int(((predicted_pos) & (labels == 1)).sum())
        fp = int(((predicted_pos) & (labels == 0)).sum())
        tpr = tp / positives if positives else 0.0
        fpr = fp / negatives if negatives else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        rows.append({"threshold": float(thr), "tpr": tpr, "fpr": fpr,
                     "precision": precision, "tp": tp, "fp": fp,
                     "positives": positives, "negatives": negatives})
    return pd.DataFrame(rows)


def _auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Trapezoidal AUC. fpr should be sorted ascending; if not we sort."""
    idx = np.argsort(fpr)
    # np.trapezoid replaces the deprecated np.trapz in numpy >= 2.0
    trap = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
    return float(trap(tpr[idx], fpr[idx]))


def analyze_roc(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per-gate ROC curves + AUCs."""
    all_curves: list[pd.DataFrame] = []
    aucs: dict[str, float] = {}
    for gate_id in sorted(df["gate_id"].unique()):
        sub = df[df["gate_id"] == gate_id]
        curve = _roc_curve(sub["risk"].values, sub["is_attack"].values)
        curve["gate_id"] = gate_id
        # AUC over the curve (fpr ascending)
        curve_sorted = curve.sort_values("fpr")
        auc = _auc(curve_sorted["fpr"].values, curve_sorted["tpr"].values)
        aucs[gate_id] = auc
        all_curves.append(curve)
    return pd.concat(all_curves, ignore_index=True), aucs


# ---------------------------------------------------------------- Threshold sweep


def analyze_threshold_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """For each (gate_id, threshold), what fraction of ATTACK events would
    be blocked (risk >= threshold)?"""
    attack_events = df[df["is_attack"] == 1]
    thresholds = np.linspace(0.30, 0.90, 13)  # 0.30, 0.35, ..., 0.90
    rows = []
    for gate_id in sorted(attack_events["gate_id"].unique()):
        sub = attack_events[attack_events["gate_id"] == gate_id]
        n = len(sub)
        if n == 0:
            continue
        for thr in thresholds:
            n_blocked = int((sub["risk"] >= thr).sum())
            rows.append({
                "gate_id": gate_id,
                "threshold": float(thr),
                "n_events": n,
                "n_would_block": n_blocked,
                "block_rate": n_blocked / n,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Figures


def _render_roc_figure(curves: pd.DataFrame, aucs: dict[str, float], out: Path) -> None:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    for gate_id in sorted(curves["gate_id"].unique()):
        sub = curves[curves["gate_id"] == gate_id].sort_values("fpr")
        plt.plot(sub["fpr"], sub["tpr"], marker="o", markersize=3,
                 label=f"{gate_id} (AUC={aucs[gate_id]:.2f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3, label="random")
    plt.xlabel("False positive rate (benign gated as attack)")
    plt.ylabel("True positive rate (attack correctly gated)")
    plt.title("Per-gate ROC — sweep of θ over existing traces")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def _render_threshold_figure(df: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4.5))
    for gate_id in sorted(df["gate_id"].unique()):
        sub = df[df["gate_id"] == gate_id].sort_values("threshold")
        plt.plot(sub["threshold"], sub["block_rate"], marker="o", markersize=4,
                 label=f"{gate_id} (n={sub['n_events'].iloc[0]})")
    plt.axvline(0.60, color="red", linestyle="--", alpha=0.4,
                label="default θ = 0.60")
    plt.xlabel("Gate threshold θ")
    plt.ylabel("Fraction of attack events blocked (risk ≥ θ)")
    plt.title("Threshold sensitivity — counterfactual over existing traces")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True, alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------- Main


def main():
    cfg = load_config()
    df = _gather_gate_events(cfg)
    print(f"Loaded {len(df)} gate events across "
          f"{df['sample_id'].nunique()} traces, "
          f"{df['gate_id'].nunique()} gate ids.")

    if df.empty:
        print("[error] no gate events on disk. Run experiments first.")
        return

    # ROC
    roc_curves, aucs = analyze_roc(df)
    roc_curves.to_csv(DELIVERABLE_FIGURES / "roc_per_gate.csv", index=False)
    _render_roc_figure(roc_curves, aucs, DELIVERABLE_FIGURES / "roc_per_gate.pdf")
    print("\nROC AUC per gate:")
    for gid, auc in sorted(aucs.items()):
        print(f"  {gid}:  AUC = {auc:.3f}")
    print(f"→ {DELIVERABLE_FIGURES / 'roc_per_gate.csv'}")
    print(f"→ {DELIVERABLE_FIGURES / 'roc_per_gate.pdf'}")

    # Threshold sensitivity
    thr_df = analyze_threshold_sensitivity(df)
    thr_df.to_csv(DELIVERABLE_FIGURES / "threshold_sensitivity.csv", index=False)
    _render_threshold_figure(thr_df, DELIVERABLE_FIGURES / "threshold_sensitivity.pdf")
    print(f"\n→ {DELIVERABLE_FIGURES / 'threshold_sensitivity.csv'}")
    print(f"→ {DELIVERABLE_FIGURES / 'threshold_sensitivity.pdf'}")

    # Merge AUCs into numbers.json (if it exists)
    if NUMBERS_JSON.exists():
        numbers = json.loads(NUMBERS_JSON.read_text(encoding="utf-8"))
        numbers.setdefault("roc_auc", {})
        for gid, auc in aucs.items():
            numbers["roc_auc"][gid] = round(auc, 4)
        NUMBERS_JSON.write_text(json.dumps(numbers, indent=2), encoding="utf-8")
        print(f"→ {NUMBERS_JSON} (updated with roc_auc)")


if __name__ == "__main__":
    main()
