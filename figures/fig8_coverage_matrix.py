"""fig8_coverage_matrix.py — 2D heatmap of defense coverage.

Rows = attacks, Columns = systems. Cell colour + label = the attribution
class of the first-blocking layer for that (system, attack) pair.

We use the priority-ordered classification from run_final's attribution table
(the same one that populates FINAL.md section 4):
    leaked > ib_blocked > azure_filter > verifier_unsafe > synth_refused > answered_safely

The heatmap makes the paper's central story instantly visible:
  * Which cells rely on Azure (opaque, stochastic)?
  * Which cells rely on IB gates (application-level, deterministic)?
  * Which cells still leak?

Reads the per-cell counts already emitted to
  paper/deliverable/figures/attribution_by_cell.csv
by run_final.py, or reconstructs them directly from results/*.json if that CSV
is missing.

Outputs:
  paper/deliverable/figures/fig8_coverage_matrix.pdf
  paper/deliverable/figures/fig8_coverage_matrix.csv
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from infoshield.attacks.registry import contains_attack_marker

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
DELIVERABLE_FIGS = REPO_ROOT / "paper" / "deliverable" / "figures"

AZURE_SENTINEL = "[AZURE_CONTENT_FILTER_REFUSED]"


def _classify_trace(trace: dict) -> str:
    """Mirror run_final.py's priority-ordered attribution classifier."""
    if contains_attack_marker(trace.get("final_output")):
        return "leaked"
    for g in trace.get("gate_decisions", []) or []:
        if g.get("status") == "BLOCKED":
            return "ib_blocked"
    # Azure sentinel: a zero-token azure call or the sentinel string in worker outputs.
    for c in trace.get("llm_calls", []) or []:
        if (
            c.get("provider") == "azure_openai"
            and int(c.get("prompt_tokens", 0)) == 0
            and int(c.get("completion_tokens", 0)) == 0
        ):
            return "azure_filter"
    worker_outputs = (trace.get("extra") or {}).get("worker_outputs") or []
    if any(AZURE_SENTINEL in (w or "") for w in worker_outputs):
        return "azure_filter"
    for r in trace.get("verifier_reports", []) or []:
        if r.get("flagged_unsafe"):
            return "verifier_unsafe"
    if trace.get("refused"):
        return "synth_refused"
    return "answered_safely"


def _collect_matrix() -> tuple[list[str], list[str], dict[tuple[str, str], Counter]]:
    """Walk results/main/*/*/seed_*/*.json, classify each attack trace."""
    matrix: dict[tuple[str, str], Counter] = {}
    for p in (RESULTS_ROOT / "main").rglob("*.json"):
        try:
            t = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        system = t.get("system")
        attack = t.get("attack")
        if not system or not attack or attack == "none":
            continue
        key = (system, attack)
        matrix.setdefault(key, Counter())[_classify_trace(t)] += 1
    systems = sorted({k[0] for k in matrix})
    attacks = sorted({k[1] for k in matrix})
    return systems, attacks, matrix


def _dominant_class(counter: Counter) -> tuple[str, float]:
    """Return the majority attribution class + its share, ignoring empties."""
    if not counter:
        return "none", 0.0
    total = sum(counter.values())
    cls, count = counter.most_common(1)[0]
    return cls, count / total


CLASS_COLORS = {
    "leaked":            "#c0392b",   # red — bad
    "ib_blocked":        "#27ae60",   # green — INFOSHIELD's catch
    "azure_filter":      "#3498db",   # blue — provider filter
    "verifier_unsafe":   "#e67e22",   # orange
    "synth_refused":     "#f1c40f",   # yellow
    "answered_safely":   "#95a5a6",   # gray — benign
    "none":              "#ecf0f1",
}

CLASS_INDEX = {c: i for i, c in enumerate(CLASS_COLORS)}


def _render_heatmap(systems: list[str], attacks: list[str],
                    matrix: dict[tuple[str, str], Counter]) -> None:
    DELIVERABLE_FIGS.mkdir(parents=True, exist_ok=True)

    n_rows = len(attacks)
    n_cols = len(systems)

    grid_cls = np.full((n_rows, n_cols), CLASS_INDEX["none"], dtype=int)
    grid_share = np.zeros((n_rows, n_cols))
    grid_n = np.zeros((n_rows, n_cols), dtype=int)

    for i, attack in enumerate(attacks):
        for j, system in enumerate(systems):
            counter = matrix.get((system, attack))
            if not counter:
                continue
            cls, share = _dominant_class(counter)
            grid_cls[i, j] = CLASS_INDEX[cls]
            grid_share[i, j] = share
            grid_n[i, j] = sum(counter.values())

    # Build a colormap from CLASS_COLORS in index order.
    from matplotlib.colors import ListedColormap
    ordered_classes = list(CLASS_COLORS.keys())
    cmap = ListedColormap([CLASS_COLORS[c] for c in ordered_classes])

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * n_cols + 2), max(4, 0.5 * n_rows + 2)))
    ax.imshow(grid_cls, cmap=cmap, vmin=0, vmax=len(ordered_classes) - 1, aspect="auto")

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(systems, rotation=25, ha="right", fontsize=9)
    ax.set_yticklabels(attacks, fontsize=9)
    ax.set_title("Defense coverage matrix: dominant attribution class per (system, attack)")

    for i in range(n_rows):
        for j in range(n_cols):
            n = grid_n[i, j]
            if n == 0:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#7f8c8d")
                continue
            cls_name = ordered_classes[grid_cls[i, j]]
            short = {
                "leaked": "LEAK", "ib_blocked": "IB",
                "azure_filter": "AZR", "verifier_unsafe": "VER",
                "synth_refused": "SYN", "answered_safely": "ok",
                "none": "—",
            }.get(cls_name, cls_name)
            share = grid_share[i, j]
            ax.text(
                j, i, f"{short}\n{int(round(share * n))}/{n}",
                ha="center", va="center", fontsize=8, color="white" if cls_name in {"leaked", "ib_blocked"} else "black",
            )

    # Legend patches
    from matplotlib.patches import Patch
    legend_classes = ["leaked", "ib_blocked", "azure_filter", "verifier_unsafe", "synth_refused", "answered_safely"]
    handles = [Patch(color=CLASS_COLORS[c], label=c) for c in legend_classes]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    fig.tight_layout()
    out = DELIVERABLE_FIGS / "fig8_coverage_matrix.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out}")


def _emit_csv(systems: list[str], attacks: list[str],
              matrix: dict[tuple[str, str], Counter]) -> None:
    out = DELIVERABLE_FIGS / "fig8_coverage_matrix.csv"
    lines = ["system,attack,n,leaked,ib_blocked,azure_filter,verifier_unsafe,synth_refused,answered_safely,dominant"]
    for attack in attacks:
        for system in systems:
            counter = matrix.get((system, attack)) or Counter()
            n = sum(counter.values())
            dominant, _ = _dominant_class(counter)
            lines.append(
                f"{system},{attack},{n},"
                f"{counter.get('leaked', 0)},{counter.get('ib_blocked', 0)},"
                f"{counter.get('azure_filter', 0)},{counter.get('verifier_unsafe', 0)},"
                f"{counter.get('synth_refused', 0)},{counter.get('answered_safely', 0)},"
                f"{dominant}"
            )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    systems, attacks, matrix = _collect_matrix()
    if not systems:
        raise SystemExit(f"[error] no traces found under {RESULTS_ROOT / 'main'}")
    print(f"[info] {len(systems)} systems x {len(attacks)} attacks")
    _render_heatmap(systems, attacks, matrix)
    _emit_csv(systems, attacks, matrix)
