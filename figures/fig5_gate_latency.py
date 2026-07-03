"""Fig 5 — gate-latency histogram (six subplots, one per gate)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="paper/figures/fig5_gate_latency.csv")
    ap.add_argument("--out", default="paper/figures/fig5_gate_latency.pdf")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)  # columns: gate_id, latency_ms (+ others)

    gate_ids = sorted(df["gate_id"].unique())
    n = len(gate_ids)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, gid in zip(axes, gate_ids):
        sub = df.loc[df["gate_id"] == gid, "latency_ms"]
        ax.hist(sub, bins=30, color="#4c72b0", edgecolor="white")
        ax.set_title(f"{gid}  (n={len(sub)})")
        ax.set_xlabel("latency (ms)")
    axes[0].set_ylabel("count")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
