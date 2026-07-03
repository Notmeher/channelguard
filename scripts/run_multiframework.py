"""Multi-framework replication stub (AutoGen + CrewAI).

This runner is currently NOT IMPLEMENTED. A faithful port requires
re-implementing planner / workers / verifier / synthesizer natively in each
framework (AutoGen's AssistantAgent + reply_func hooks, CrewAI's Agent + tool
registry) so the six IB gates fire at the equivalent boundaries.

Running identical LangGraph copies under different names — as the previous
stub did — produces meaningless "replication" numbers, so the runner now
exits cleanly and prints a TODO instead of writing misleading traces.

To force the old stub behaviour (only useful for smoke tests of the wiring),
pass `--force-stub`.
"""
from __future__ import annotations

import argparse
import sys

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
    ap.add_argument(
        "--force-stub",
        action="store_true",
        help="Run the LangGraph pipeline under AutoGen/CrewAI aliases. Produces "
             "meaningless replication numbers; only for smoke-testing the wiring.",
    )
    args = ap.parse_args()

    if not args.force_stub:
        print(
            "[skip] run_multiframework: native AutoGen / CrewAI ports are not "
            "implemented yet. Re-run with --force-stub to write LangGraph traces "
            "under the AutoGen / CrewAI aliases (smoke-only, not a real result)."
        )
        sys.exit(0)

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
    print(f"multi_framework (stub): {stats}")


if __name__ == "__main__":
    main()
