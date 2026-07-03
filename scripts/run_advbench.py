"""AdvBench refusal benchmark — every prompt is a jailbreak; correct behavior is refusal."""
from __future__ import annotations

import argparse

from infoshield.config import load_config
from infoshield.datasets import load as load_dataset
from infoshield.runner import run_matrix

SYSTEMS = ["undefended", "ibprotector", "smoothllm", "llm_guard", "perplexity_filter", "infoshield"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    args = ap.parse_args()

    cfg = load_config()
    ecfg = cfg.experiments["advbench"]
    samples = load_dataset(ecfg.dataset, n_samples=args.n_samples or ecfg.n_samples, seed=(args.seeds or ecfg.seeds)[0])
    stats = run_matrix(
        cfg=cfg,
        experiment="advbench",
        dataset_name=ecfg.dataset,
        samples=samples,
        systems=SYSTEMS,
        attacks=["none"],
        seeds=args.seeds or ecfg.seeds,
        progress_cb=print,
    )
    print(f"advbench: {stats}")


if __name__ == "__main__":
    main()
