"""Phrase-bank size ablation: sweep bank sizes {5, 20, 50}."""
from __future__ import annotations

import argparse
import copy

from infoshield.baselines.registry import SYSTEMS
from infoshield.config import load_config
from infoshield.datasets import load as load_dataset
from infoshield.ib.gates import build_ib_gates
from infoshield.pipeline.langgraph_pipeline import build_pipeline
from infoshield.runner import run_matrix


def _register_bank_sizes(cfg, sizes: list[int]):
    for sz in sizes:
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy.ib.phrase_bank_size = sz
        gates = build_ib_gates(cfg_copy)

        def _make(g=gates):
            def _b(_c):
                return build_pipeline(_c, ib_gates=g)
            return _b

        SYSTEMS[f"infoshield_bank_{sz}"] = _make()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    args = ap.parse_args()

    cfg = load_config()
    ecfg = cfg.experiments["phrase_bank_ablation"]
    sizes = ecfg.sizes or [5, 20, 50]
    _register_bank_sizes(cfg, sizes)

    samples = load_dataset(ecfg.dataset, n_samples=args.n_samples or ecfg.n_samples, seed=(args.seeds or ecfg.seeds)[0])
    systems = [f"infoshield_bank_{sz}" for sz in sizes]

    stats = run_matrix(
        cfg=cfg,
        experiment="phrase_bank_ablation",
        dataset_name=ecfg.dataset,
        samples=samples,
        systems=systems,
        attacks=["prompt_injection_strong"],
        seeds=args.seeds or ecfg.seeds,
        progress_cb=print,
    )
    print(f"phrase_bank_ablation: {stats}")


if __name__ == "__main__":
    main()
