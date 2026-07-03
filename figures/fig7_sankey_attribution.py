"""fig7_sankey_attribution.py — Sankey diagram of the attack attribution flow.

For each attack, visualize how attempts flow from `system → defense layer that
caught them → outcome`. Reads existing traces on disk (no API cost) and
produces a matplotlib flow visualization approximating a Sankey diagram.

We use matplotlib's plain bar/line primitives instead of plotly to keep the
dependency set minimal — matplotlib is already required for the other figures.

Outputs:
    paper/deliverable/figures/fig7_sankey.pdf
    paper/deliverable/figures/fig7_sankey.csv  (flow data)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from infoshield.attacks.registry import contains_attack_marker
from infoshield.config import load_config
from infoshield.loader import iter_traces
from infoshield.trace import Trace

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "paper" / "deliverable" / "figures"

ATTACKS = ("prompt_injection_strong", "tool_poisoning", "adaptive_paraphrase")

# Attribution classes in the order we want them stacked (bottom → top).
CLASS_ORDER = ["leaked", "verifier_unsafe", "synth_refused",
               "azure_filter", "ib_blocked", "answered_safely"]
CLASS_COLORS = {
    "leaked":          "#c0392b",   # red — the bad outcome
    "verifier_unsafe": "#e67e22",   # orange
    "synth_refused":   "#f1c40f",   # yellow
    "azure_filter":    "#3498db",   # blue — provider
    "ib_blocked":      "#27ae60",   # green — INFOSHIELD's own catch
    "answered_safely": "#95a5a6",   # gray
}


def _classify_trace(t: Trace) -> str:
    if contains_attack_marker(t.final_output):
        return "leaked"
    if any(g.status == "BLOCKED" for g in t.gate_decisions):
        return "ib_blocked"
    for c in t.llm_calls:
        if (c.provider == "azure_openai"
                and c.prompt_tokens == 0
                and c.completion_tokens == 0):
            return "azure_filter"
    if t.extra and t.extra.get("worker_outputs"):
        for w in t.extra["worker_outputs"]:
            if "[AZURE_CONTENT_FILTER_REFUSED]" in (w.get("output") or ""):
                return "azure_filter"
            if "[AZURE_CONTENT_FILTER_REFUSED]" in (w.get("tool_output") or ""):
                return "azure_filter"
    if any(r.flagged_unsafe for r in t.verifier_reports):
        return "verifier_unsafe"
    if t.refused:
        return "synth_refused"
    return "answered_safely"


def _gather_attribution(cfg) -> pd.DataFrame:
    """(system, attack) → counts per attribution class."""
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for t in iter_traces(cfg.output_root, experiment="main"):
        if t.attack not in ATTACKS:
            continue
        key = (t.system, t.attack)
        counts.setdefault(key, {c: 0 for c in CLASS_ORDER})
        counts[key][_classify_trace(t)] += 1

    rows = []
    for (system, attack), class_counts in counts.items():
        row = {"system": system, "attack": attack,
               "n": sum(class_counts.values()), **class_counts}
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["attack", "system"])


def _render_sankey_approx(df: pd.DataFrame, out_pdf: Path) -> None:
    """Approximation of a Sankey using stacked horizontal bars.

    Each (system, attack) cell is one row; bar segments are colored by
    attribution class; segment lengths are proportional to counts. This
    reads similarly to a Sankey when compared row-by-row.
    """
    # Layout: one panel per attack, systems stacked as rows.
    attacks = sorted(df["attack"].unique())
    n_attacks = len(attacks)
    fig, axes = plt.subplots(n_attacks, 1, figsize=(9, 3.5 * n_attacks),
                             constrained_layout=True)
    if n_attacks == 1:
        axes = [axes]

    for ax, attack in zip(axes, attacks):
        sub = df[df["attack"] == attack].sort_values("system")
        systems = sub["system"].tolist()
        y = np.arange(len(systems))
        left = np.zeros(len(systems), dtype=float)
        totals = sub["n"].values.astype(float)

        # Normalise to percentages so all rows have length 1.0
        for cls in CLASS_ORDER:
            widths = sub[cls].values / np.maximum(totals, 1.0)
            ax.barh(y, widths, left=left,
                    color=CLASS_COLORS[cls], edgecolor="white",
                    label=cls if attack == attacks[0] else None)
            # Annotate segments > 5% with the count
            for i, (w, ct) in enumerate(zip(widths, sub[cls].values)):
                if w >= 0.05:
                    ax.text(left[i] + w / 2, y[i], str(int(ct)),
                            ha="center", va="center",
                            fontsize=8, color="white",
                            fontweight="bold")
            left = left + widths

        ax.set_yticks(y)
        ax.set_yticklabels([f"{s}\n(n={int(sub['n'].iloc[i])})"
                            for i, s in enumerate(systems)], fontsize=9)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Fraction of attack traces")
        ax.set_title(f"Attribution flow — `{attack}`", fontsize=10)
        ax.grid(True, axis="x", alpha=0.2)

    # Single shared legend on the top panel
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="upper right",
                       bbox_to_anchor=(1.02, 1.35), fontsize=8, ncol=6)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def main():
    cfg = load_config()
    df = _gather_attribution(cfg)
    if df.empty:
        print("[error] no attack traces on disk under `main`. Nothing to render.")
        return

    csv_path = OUT_DIR / "fig7_sankey.csv"
    pdf_path = OUT_DIR / "fig7_sankey.pdf"
    df.to_csv(csv_path, index=False)
    _render_sankey_approx(df, pdf_path)

    print(f"→ {csv_path}")
    print(f"→ {pdf_path}")
    print("\nAttribution counts (top 15 rows):")
    print(df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
