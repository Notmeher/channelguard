"""Beta sweep: INFOSHIELD at multiple beta / threshold ratios.

We sweep the IB threshold (proxy for beta) across cfg.experiments.beta_sweep.betas
and register a distinct system name per beta.
"""
from __future__ import annotations

import argparse
import copy

from infoshield.baselines.registry import SYSTEMS
from infoshield.config import load_config
from infoshield.datasets import load as load_dataset
from infoshield.ib.gates import build_ib_gates
from infoshield.pipeline.langgraph_pipeline import build_pipeline
from infoshield.runner import run_matrix


def _register_betas(cfg, betas: list[float]):
    for b in betas:
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy.ib.beta = b
        # Higher beta -> stricter safety -> lower threshold (more sensitive).
        # Map beta -> threshold heuristically so ablation makes sense.
        cfg_copy.ib.threshold = max(0.15, 0.85 - 0.15 * b)
        gates = build_ib_gates(cfg_copy)

        def _make(g=gates):
            def _b(_c):
                return build_pipeline(_c, ib_gates=g)
            return _b

        SYSTEMS[f"infoshield_beta_{b}"] = _make()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    ecfg = cfg.experiments["beta_sweep"]
    betas = ecfg.betas or [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    _register_betas(cfg, betas)

    samples = load_dataset(ecfg.dataset, n_samples=args.n_samples or ecfg.n_samples, seed=args.seed)
    systems = [f"infoshield_beta_{b}" for b in betas]

    stats = run_matrix(
        cfg=cfg,
        experiment="beta_sweep",
        dataset_name=ecfg.dataset,
        samples=samples,
        systems=systems,
        attacks=["prompt_injection_strong"],
        seeds=[args.seed],
        progress_cb=print,
    )
    print(f"beta_sweep: {stats}")


if __name__ == "__main__":
    main()
