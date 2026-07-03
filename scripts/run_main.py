"""Main experiment: 6 systems × 5 attacks × N HotpotQA samples × 3 seeds."""
from __future__ import annotations

import argparse

from infoshield.config import load_config
from infoshield.datasets import load as load_dataset
from infoshield.runner import run_matrix

SYSTEMS = ["undefended", "ibprotector", "smoothllm", "llm_guard", "perplexity_filter", "infoshield"]
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
    ap.add_argument("--systems", nargs="+", default=None)
    ap.add_argument("--attacks", nargs="+", default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    ecfg = cfg.experiments["main"]
    n = args.n_samples or ecfg.n_samples
    seeds = args.seeds or ecfg.seeds
    systems = args.systems or SYSTEMS
    attacks = args.attacks or ATTACKS

    samples = load_dataset(ecfg.dataset, n_samples=n, seed=seeds[0])
    stats = run_matrix(
        cfg=cfg,
        experiment="main",
        dataset_name=ecfg.dataset,
        samples=samples,
        systems=systems,
        attacks=attacks,
        seeds=seeds,
        resume=not args.no_resume,
        progress_cb=print,
    )
    print(f"\nmain: {stats}")


if __name__ == "__main__":
    main()
