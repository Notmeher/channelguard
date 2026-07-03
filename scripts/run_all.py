"""Orchestrator: run every experiment in the recommended order.

Usage:
    python scripts/run_all.py                   # full plan
    python scripts/run_all.py --smoke           # tiny 2-sample smoke of each
    python scripts/run_all.py --only main gsm8k # subset

Every runner writes trace JSONs to `results/`. Nothing here contains numbers;
`scripts/aggregate_results.py` turns the JSONs into `paper/numbers.json` and
`paper/figures/*.csv`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ORDER = [
    "run_main",
    "run_advbench",
    "run_gsm8k",
    "run_injecagent",
    "run_adaptive",
    "run_ablation",
    "run_beta_sweep",
    "run_multiframework",
    "run_encoder_comparison",
    "run_weak_model",
    "run_phrase_bank_ablation",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Run 2 samples per experiment.")
    ap.add_argument("--only", nargs="+", help="Subset of runner basenames to run.")
    args = ap.parse_args()

    scripts_dir = Path(__file__).parent
    runners = args.only or ORDER

    for name in runners:
        script = scripts_dir / f"{name}.py"
        if not script.exists():
            print(f"[skip] {name} not found")
            continue
        cmd = [sys.executable, str(script)]
        if args.smoke:
            cmd += ["--n_samples", "2", "--seeds", "0"]
        print(f"\n$ {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[warn] {name} exited with code {result.returncode}")


if __name__ == "__main__":
    main()
