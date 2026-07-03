"""Fig 2 — beta Pareto curve: ASR vs CTCA across beta values."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="paper/figures/fig2_beta.csv")
    ap.add_argument("--out", default="paper/figures/fig2_beta.pdf")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)  # columns: beta, asr, ctca
    df = df.sort_values("beta")

    plt.figure(figsize=(6, 5))
    plt.plot(df["ctca"], df["asr"], "o-", linewidth=2, markersize=9)
    for _, row in df.iterrows():
        plt.annotate(f"β={row['beta']}", (row["ctca"], row["asr"]),
                     textcoords="offset points", xytext=(6, 6), fontsize=9)
    plt.xlabel("CTCA (task completion accuracy)")
    plt.ylabel("ASR (attack success rate)")
    plt.grid(True, alpha=0.3)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
