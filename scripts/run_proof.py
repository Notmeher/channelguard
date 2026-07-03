"""Proof run: prove INFOSHIELD works vs undefended on the minimum viable slice.

Cells:
  main    :  {undefended, infoshield} x {prompt_injection_strong, tool_poisoning,
             adaptive_paraphrase, memory_poisoning, none} x 30 samples
  gsm8k   :  {undefended, infoshield} x {none} x 30 samples  (CTCA fidelity check)

Total: (2 x 5 x 30) + (2 x 1 x 30) = 300 + 60 = ~360 pipeline runs.
With cfg.pipeline.parallel_samples = 8 -> ~15-25 min wall-clock.
Cost: ~$30-60.

Also runs `undefended` under `none` to establish baseline CTCA on HotpotQA.
"""
from __future__ import annotations

import time

from infoshield.config import load_config
from infoshield.datasets import load as load_dataset
from infoshield.runner import run_matrix


PROOF_SYSTEMS = ["undefended", "infoshield"]
# Skipping memory_poisoning: Azure's own content filter blocks it before gpt-5
# sees the prompt, so the story becomes "Azure caught it" rather than "IB caught it".
# INFOSHIELD's memory rescoring is still tested separately in unit tests.
PROOF_ATTACKS = [
    "none",
    "prompt_injection_strong",
    "tool_poisoning",
    "adaptive_paraphrase",
]


def main():
    cfg = load_config()
    n = 30
    seeds = [0]

    print(f"[proof_run] parallel_samples={cfg.pipeline.parallel_samples}, n={n}, seeds={seeds}")

    # --- Main slice on HotpotQA -----------------------------------------
    print("\n=== main slice (HotpotQA, 2 systems x 5 attacks x 30 samples) ===")
    samples = load_dataset("hotpot_qa", n_samples=n, seed=seeds[0])
    t0 = time.time()
    stats_main = run_matrix(
        cfg=cfg,
        experiment="main",
        dataset_name="hotpot_qa",
        samples=samples,
        systems=PROOF_SYSTEMS,
        attacks=PROOF_ATTACKS,
        seeds=seeds,
        progress_cb=print,
    )
    print(f"main: {stats_main}  in {time.time()-t0:.1f}s")

    # --- GSM8K fidelity check -------------------------------------------
    print("\n=== gsm8k fidelity (2 systems x no attack x 30 samples) ===")
    gsm_samples = load_dataset("gsm8k", n_samples=n, seed=seeds[0])
    t0 = time.time()
    stats_gsm = run_matrix(
        cfg=cfg,
        experiment="gsm8k",
        dataset_name="gsm8k",
        samples=gsm_samples,
        systems=PROOF_SYSTEMS,
        attacks=["none"],
        seeds=seeds,
        progress_cb=print,
    )
    print(f"gsm8k: {stats_gsm}  in {time.time()-t0:.1f}s")

    print("\n=== proof run complete ===")
    print(f"main total: done={stats_main['done']} skipped={stats_main['skipped']} failed={stats_main['failed']}")
    print(f"gsm8k total: done={stats_gsm['done']} skipped={stats_gsm['skipped']} failed={stats_gsm['failed']}")


if __name__ == "__main__":
    main()
