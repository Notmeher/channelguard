"""analyze_certified_radius.py — certified robustness under embedding-space perturbations.

For every IB GateDecision recorded in the 499+ traces on disk, compute the
minimum L2 perturbation in embedding space that would flip the gate's
BLOCKED/PASSED verdict. Emit:

  paper/deliverable/figures/certified_radius_per_gate.csv
  paper/deliverable/figures/certified_radius_per_gate.pdf
  paper/deliverable/figures/certified_radius_cdf.pdf

Also updates paper/numbers.json with `certified_radius.<gate_id>.median`.

------------------------------------------------------------------------------
Mathematical setup
------------------------------------------------------------------------------
The gate scores an input text `t` as
    risk(t) = max_{s in sentences(t)} max_{b in bank} cos(e(s), e(b))
where e(.) is an L2-normalized sentence embedding. The gate BLOCKs iff
    risk(t) >= theta      (theta = 0.60 by default).

For a single (sentence, phrase) pair with unit vectors u, v and cosine
similarity c = u . v, the first-order effect of perturbing u by delta on the
renormalized cosine is
    d cos = (v - c * u) . delta                              (Taylor, |delta| small)
so the *minimum* ||delta||_2 needed to move cos by |Delta| is
    eps_min = |Delta| / ||v - c u||_2 = |Delta| / sqrt(1 - c^2).

For a BLOCKED trace with risk r >= theta, the certified detection margin is
    eps_defender(r) = (r - theta) / sqrt(1 - r^2)
i.e. an attacker who perturbs the embedding by strictly less than this cannot
push the gate into PASSED.

For a PASSED trace with risk r < theta, the certified benign margin is
    eps_benign(r) = (theta - r) / sqrt(1 - r^2)
i.e. no benign perturbation of that magnitude will accidentally trigger the
gate. This gives a per-input false-positive robustness certificate.

Both use the same closed form. We report them per gate_id + per attack.

Caveats (documented in the paper appendix):
  - First-order approximation, tight only for small perturbations.
  - Bounds the *worst-case* (sentence, phrase) pair. A conservative estimate
    of the input-space perturbation is at least this large in embedding
    space; converting to an input-space (character/token) perturbation
    requires additional Lipschitz analysis of the encoder, which is
    encoder-specific and out of scope here.
  - Assumes the attacker cannot change the phrase bank (which is exactly
    the defender's assumption in Section 3.2).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
DELIVERABLE_FIGS = REPO_ROOT / "paper" / "deliverable" / "figures"
NUMBERS_JSON = REPO_ROOT / "paper" / "numbers.json"

THETA_DEFAULT = 0.60


# --------------------------------------------------------------------- helpers

def _iter_trace_files(root: Path):
    if not root.exists():
        return
    for p in root.rglob("*.json"):
        yield p


def _load_gate_events(root: Path) -> list[dict]:
    events: list[dict] = []
    for p in _iter_trace_files(root):
        try:
            t = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for g in t.get("gate_decisions", []) or []:
            events.append({
                "system": t.get("system"),
                "attack": t.get("attack"),
                "gate_id": g.get("gate_id"),
                "risk": g.get("risk"),
                "status": g.get("status"),
            })
    return events


def _certified_eps(r: float, theta: float) -> float:
    """|theta - r| / sqrt(1 - r^2), clamped for numerical stability."""
    r = max(min(r, 0.999999), -0.999999)
    denom = math.sqrt(max(1.0 - r * r, 1e-9))
    return abs(theta - r) / denom


# --------------------------------------------------------------------- analysis

def analyze_certified_radius(theta: float = THETA_DEFAULT) -> dict:
    events = _load_gate_events(RESULTS_ROOT)
    if not events:
        raise SystemExit(f"[error] no gate events under {RESULTS_ROOT}")

    per_gate: dict[str, dict[str, list[float]]] = {}
    for e in events:
        gid = e["gate_id"]
        r = e.get("risk")
        if r is None or not isinstance(r, (int, float)):
            continue
        eps = _certified_eps(float(r), theta)
        bucket = per_gate.setdefault(gid, {"defender": [], "benign": [], "all": []})
        bucket["all"].append(eps)
        if float(r) >= theta:
            bucket["defender"].append(eps)   # defender margin on BLOCKED events
        else:
            bucket["benign"].append(eps)     # false-positive margin on PASSED events

    summary: dict[str, dict[str, float]] = {}
    for gid, bucket in per_gate.items():
        summary[gid] = {
            "n_defender": len(bucket["defender"]),
            "n_benign": len(bucket["benign"]),
            "median_defender": float(np.median(bucket["defender"])) if bucket["defender"] else 0.0,
            "median_benign": float(np.median(bucket["benign"])) if bucket["benign"] else 0.0,
            "p10_defender": float(np.percentile(bucket["defender"], 10)) if bucket["defender"] else 0.0,
            "p90_defender": float(np.percentile(bucket["defender"], 90)) if bucket["defender"] else 0.0,
            "p10_benign": float(np.percentile(bucket["benign"], 10)) if bucket["benign"] else 0.0,
            "p90_benign": float(np.percentile(bucket["benign"], 90)) if bucket["benign"] else 0.0,
        }

    _render_bar(summary, theta)
    _render_cdf(per_gate, theta)
    _emit_csv(summary)
    _update_numbers_json(summary, theta)

    print(f"\n===== certified_radius (theta = {theta:.2f}) =====")
    print(f"{'gate':10} {'n_defend':>9} {'med_defend':>11} {'p10_defend':>11} {'n_benign':>9} {'med_benign':>11}")
    for gid in sorted(summary):
        s = summary[gid]
        print(
            f"{gid:10} {s['n_defender']:>9d} {s['median_defender']:>11.4f} "
            f"{s['p10_defender']:>11.4f} {s['n_benign']:>9d} {s['median_benign']:>11.4f}"
        )
    return summary


# --------------------------------------------------------------------- outputs

def _render_bar(summary: dict[str, dict[str, float]], theta: float) -> None:
    DELIVERABLE_FIGS.mkdir(parents=True, exist_ok=True)
    gates = sorted(summary)
    med_def = [summary[g]["median_defender"] for g in gates]
    med_ben = [summary[g]["median_benign"] for g in gates]

    x = np.arange(len(gates))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, med_def, w, label="Defender margin (BLOCKED events)", color="#27ae60")
    ax.bar(x + w / 2, med_ben, w, label="Benign margin (PASSED events)", color="#3498db")
    ax.set_xticks(x)
    ax.set_xticklabels(gates, rotation=15)
    ax.set_ylabel(r"Median certified $\varepsilon$ (L2 in embedding space)")
    ax.set_title(f"Certified perturbation radius per gate (theta = {theta:.2f})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = DELIVERABLE_FIGS / "certified_radius_per_gate.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out}")


def _render_cdf(per_gate: dict[str, dict[str, list[float]]], theta: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.tab10.colors
    for i, gid in enumerate(sorted(per_gate)):
        vals = sorted(per_gate[gid]["all"])
        if not vals:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, label=gid, color=colors[i % len(colors)], lw=1.5)
    ax.set_xlabel(r"Certified $\varepsilon$ (L2 in embedding space)")
    ax.set_ylabel("CDF over gate events")
    ax.set_title(f"CDF of certified radii per gate (theta = {theta:.2f})")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(left=0)
    fig.tight_layout()
    out = DELIVERABLE_FIGS / "certified_radius_cdf.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out}")


def _emit_csv(summary: dict[str, dict[str, float]]) -> None:
    out = DELIVERABLE_FIGS / "certified_radius_per_gate.csv"
    lines = ["gate,n_defender,median_defender,p10_defender,p90_defender,n_benign,median_benign,p10_benign,p90_benign"]
    for gid in sorted(summary):
        s = summary[gid]
        lines.append(
            f"{gid},{s['n_defender']},{s['median_defender']:.6f},{s['p10_defender']:.6f},{s['p90_defender']:.6f},"
            f"{s['n_benign']},{s['median_benign']:.6f},{s['p10_benign']:.6f},{s['p90_benign']:.6f}"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out}")


def _update_numbers_json(summary: dict[str, dict[str, float]], theta: float) -> None:
    numbers = {}
    if NUMBERS_JSON.exists():
        try:
            numbers = json.loads(NUMBERS_JSON.read_text(encoding="utf-8"))
        except Exception:
            numbers = {}
    numbers.setdefault("certified_radius", {})
    numbers["certified_radius"]["theta"] = theta
    for gid, s in summary.items():
        numbers["certified_radius"][gid] = {
            "median_defender": round(s["median_defender"], 6),
            "median_benign": round(s["median_benign"], 6),
            "n_defender": s["n_defender"],
            "n_benign": s["n_benign"],
        }
    NUMBERS_JSON.write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ok] wrote {NUMBERS_JSON}")


# --------------------------------------------------------------------- main

if __name__ == "__main__":
    analyze_certified_radius(theta=THETA_DEFAULT)
