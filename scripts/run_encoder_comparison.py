"""Encoder comparison: local MiniLM vs Azure text-embedding-3-large.

Runs the SAME samples through both systems under the standard 5 attack types,
so per-sample deltas are directly comparable. Result JSONs partition by
system name (`infoshield` vs `infoshield_azure`) — the aggregator produces
side-by-side ASR/UPS/latency cells for the encoder-ablation table.
"""
from __future__ import annotations

import argparse

from infoshield.config import load_config
from infoshield.datasets import load as load_dataset
from infoshield.runner import run_matrix

SYSTEMS = ["infoshield", "infoshield_azure"]
ATTACKS = [
    "prompt_injection_strong",
    "tool_poisoning",
    "memory_poisoning",
    "reasoning_state_injection",
    "consensus_collapse",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--attacks", nargs="+", default=None)
    args = ap.parse_args()

    cfg = load_config()
    ecfg = cfg.experiments["main"]      # reuse main dataset config
    n = args.n_samples or 100
    seeds = args.seeds or [0]
    attacks = args.attacks or ATTACKS

    samples = load_dataset(ecfg.dataset, n_samples=n, seed=seeds[0])
    stats = run_matrix(
        cfg=cfg,
        experiment="encoder_comparison",
        dataset_name=ecfg.dataset,
        samples=samples,
        systems=SYSTEMS,
        attacks=attacks,
        seeds=seeds,
        progress_cb=print,
    )
    print(f"\nencoder_comparison: {stats}")


if __name__ == "__main__":
    main()
