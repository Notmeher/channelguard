"""Fig 3 — ablation heatmap: rows = configs, cols = ASR / CTCA."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="paper/figures/fig3_ablation.csv")
    ap.add_argument("--out", default="paper/figures/fig3_ablation.pdf")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)  # columns: config, asr, ctca
    df = df.set_index("config")[["asr", "ctca"]]

    plt.figure(figsize=(6, 4))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="RdYlGn_r", vmin=0.0, vmax=1.0, cbar=True)
    plt.title("Per-layer ablation")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
