"""Phase 1 smoke: 3 HotpotQA samples through the undefended baseline pipeline.

Verifies:
  * dataset load
  * planner/worker/verifier/synthesizer nodes run
  * one JSON trace is written per sample under results/phase1_smoke/
"""
from __future__ import annotations

import argparse
from pathlib import Path

from infoshield.config import load_config
from infoshield.datasets import load as load_dataset
from infoshield.pipeline.langgraph_pipeline import build_pipeline


def main(n_samples: int = 3, seed: int = 0):
    cfg = load_config()
    pipeline = build_pipeline(cfg, ib_gates=None)  # undefended
    samples = load_dataset("hotpot_qa", n_samples=n_samples, seed=seed)

    print(f"Running {len(samples)} HotpotQA samples through undefended baseline...")
    for i, s in enumerate(samples, 1):
        print(f"  [{i}/{len(samples)}] {s['sample_id']} :: {s['input'][:80]}...")
        trace = pipeline(
            sample_id=s["sample_id"],
            raw_input=s["input"],
            ground_truth=s["ground_truth"],
            experiment="phase1_smoke",
            system="undefended",
            attack="none",
            seed=seed,
            dataset="hotpot_qa",
        )
        path = trace.write(cfg.output_root)
        print(f"      -> {path}")
        print(f"      final: {trace.final_output[:120]}...")
        print(f"      tokens: {trace.total_tokens()}   verifiers: {len(trace.verifier_reports)}")

    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args.n_samples, args.seed)
