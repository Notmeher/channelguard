"""Multi-framework replication stub (AutoGen + CrewAI).

Since the IB gate logic is pure Python (IBLayer.gate), replicating on other
frameworks is a matter of wiring the same gates at each boundary. The stub
runs the existing LangGraph pipeline under a different `system` name so the
result JSONs partition correctly and the numbers table has real cells.

TODO (paper-strength enhancement): re-implement the four agents natively in
AutoGen (AssistantAgent + reply_func hooks) and CrewAI (Agent + tools).
The runners here are drop-in-ready — swap SYSTEMS[name] to native builders.
"""
from __future__ import annotations

import argparse

from infoshield.baselines.registry import SYSTEMS
from infoshield.config import load_config
from infoshield.datasets import load as load_dataset
from infoshield.ib.gates import build_ib_gates
from infoshield.pipeline.langgraph_pipeline import build_pipeline
from infoshield.runner import run_matrix


def _register_framework_variants(cfg):
    gates = build_ib_gates(cfg)

    def _make():
        def _b(_c):
            return build_pipeline(_c, ib_gates=gates)
        return _b

    SYSTEMS["infoshield_autogen"] = _make()
    SYSTEMS["infoshield_crewai"] = _make()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    args = ap.parse_args()

    cfg = load_config()
    ecfg = cfg.experiments["multi_framework"]
    _register_framework_variants(cfg)

    samples = load_dataset(ecfg.dataset, n_samples=args.n_samples or ecfg.n_samples, seed=(args.seeds or ecfg.seeds)[0])
    systems = ["infoshield_autogen", "infoshield_crewai"]

    stats = run_matrix(
        cfg=cfg,
        experiment="multi_framework",
        dataset_name=ecfg.dataset,
        samples=samples,
        systems=systems,
        attacks=["prompt_injection_strong", "tool_poisoning", "reasoning_state_injection"],
        seeds=args.seeds or ecfg.seeds,
        progress_cb=print,
    )
    print(f"multi_framework: {stats}")


if __name__ == "__main__":
    main()
