"""Per-layer ablation.

7 configs: full system + 6 "remove one gate" variants. Each drops one IB index
from cfg.ib.active_layers and rebuilds the infoshield pipeline.
"""
from __future__ import annotations

import argparse
import copy

from infoshield.baselines.registry import register, SYSTEMS
from infoshield.config import load_config
from infoshield.datasets import load as load_dataset
from infoshield.ib.gates import build_ib_gates
from infoshield.pipeline.langgraph_pipeline import build_pipeline
from infoshield.runner import run_matrix


def _register_ablations(cfg):
    """Register 6 'no_IB-i' systems + a 'full' alias."""
    for drop in range(6):
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy.ib.active_layers = [i for i in cfg.ib.active_layers if i != drop]
        gates = build_ib_gates(cfg_copy)
        name = f"infoshield_no_ib{drop}"

        # capture-by-value trick
        def _make_builder(g=gates):
            def _b(_cfg):
                return build_pipeline(_cfg, ib_gates=g)
            return _b

        SYSTEMS[name] = _make_builder()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    args = ap.parse_args()

    cfg = load_config()
    ecfg = cfg.experiments["ablation"]
    _register_ablations(cfg)

    systems = ["infoshield"] + [f"infoshield_no_ib{i}" for i in range(6)]
    samples = load_dataset(ecfg.dataset, n_samples=args.n_samples or ecfg.n_samples, seed=(args.seeds or ecfg.seeds)[0])

    stats = run_matrix(
        cfg=cfg,
        experiment="ablation",
        dataset_name=ecfg.dataset,
        samples=samples,
        systems=systems,
        attacks=["prompt_injection_strong"],
        seeds=args.seeds or ecfg.seeds,
        progress_cb=print,
    )
    print(f"ablation: {stats}")


if __name__ == "__main__":
    main()
