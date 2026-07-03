"""Fig 1 — grouped bar chart of ASR across systems × attacks."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="paper/figures/fig1_asr.csv")
    ap.add_argument("--out", default="paper/figures/fig1_asr.pdf")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)  # columns: system, attack, asr, ci_low, ci_high
    plt.figure(figsize=(11, 5))
    sns.set_style("whitegrid")
    order = ["undefended", "ibprotector", "smoothllm", "llm_guard", "perplexity_filter", "infoshield"]
    ax = sns.barplot(
        data=df, x="attack", y="asr", hue="system",
        hue_order=[s for s in order if s in df["system"].unique()],
        palette="Set2",
    )
    ax.set_ylabel("Attack Success Rate")
    ax.set_xlabel("")
    ax.set_ylim(0.0, 1.0)
    plt.xticks(rotation=25, ha="right")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
