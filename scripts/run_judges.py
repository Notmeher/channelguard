"""run_judges.py — LLM-as-judge evaluations over existing traces.

Runs three judge-based evaluations that were implemented in the framework
but never invoked in the report:

  1. hallucination_rate ([metrics/reasoning.py]): Claude Opus asks
     "does the final answer contain a factual claim not supported by the
     provided context?" over a sample of traces. Reports fraction YES.
     ~$0.03 per trace × 30 traces = ~$1

  2. multi_step_consistency ([metrics/reasoning.py]): Claude Sonnet asks
     "is the reasoning chain internally consistent?" over the same sample.
     ~$0.01 per trace × 30 traces = ~$0.30

  3. Two-judge borderline eval ([metrics/llm_eval.py]): Opus + Sonnet each
     answer 3 yes/no questions about borderline traces. Disagreements are
     tiebroken by gpt-5. Reports Cohen's κ between the two Claude judges.
     ~$0.08 per trace × 30 borderline traces = ~$2.40

Sample selection:
  - Uses `is_borderline(trace)` from `metrics/llm_eval.py` to pick 30 traces
    where the pipeline made a non-obvious decision (some gates fired, verifier
    flagged, but not a hard refusal). Both the hallucination and consistency
    judges are applied to the SAME 30 traces so the numbers are comparable.

Total: ~$4-6 depending on sample count.

Outputs:
  paper/deliverable/judges_report.json  (all raw results)
  paper/deliverable/FINAL.md            (adds §6.7 automatically? No —
                                          just references judges_report.json)

Usage:
    python scripts/run_judges.py                # 30 borderline traces, ~$5
    python scripts/run_judges.py --n 20         # smaller subset
    python scripts/run_judges.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from infoshield.config import load_config
from infoshield.llm import AnthropicClient, AzureOpenAIClient
from infoshield.loader import iter_traces
from infoshield.metrics import llm_eval, reasoning

REPO_ROOT = Path(__file__).resolve().parents[1]
DELIVERABLE = REPO_ROOT / "paper" / "deliverable"
JUDGES_JSON = DELIVERABLE / "judges_report.json"

# Rough cost per judged trace, for the dry-run estimate.
COST_HALLUCINATION_USD = 0.03    # Claude Opus, ~1 short prompt+response
COST_CONSISTENCY_USD = 0.01      # Claude Sonnet, ~1 short prompt+response
COST_TWO_JUDGE_USD = 0.08        # Opus + Sonnet + possible gpt-5 tiebreaker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30,
                    help="Number of borderline traces to sample.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config()

    all_traces = list(iter_traces(cfg.output_root))
    print(f"Loaded {len(all_traces)} traces on disk.")

    borderline = llm_eval.select_borderline(all_traces, k=args.n)
    print(f"Selected {len(borderline)} borderline traces "
          f"(is_borderline == True) for judge evaluation.")

    est_cost = (
        len(borderline) * COST_HALLUCINATION_USD
        + len(borderline) * COST_CONSISTENCY_USD
        + len(borderline) * COST_TWO_JUDGE_USD
    )
    print(f"\nEstimated cost: ~${est_cost:.2f}")
    print(f"  hallucination_rate:  {len(borderline)} × ${COST_HALLUCINATION_USD} = ${len(borderline) * COST_HALLUCINATION_USD:.2f}")
    print(f"  multi_step_consist:  {len(borderline)} × ${COST_CONSISTENCY_USD} = ${len(borderline) * COST_CONSISTENCY_USD:.2f}")
    print(f"  two_judge + tiebrk:  {len(borderline)} × ${COST_TWO_JUDGE_USD} = ${len(borderline) * COST_TWO_JUDGE_USD:.2f}")

    if args.dry_run:
        return
    if not borderline:
        print("[error] no borderline traces found. Run experiments first.")
        sys.exit(1)

    # --- Build clients (all from config.yaml) ---------------------------
    opus_cfg = cfg.models.judge_hallucination
    sonnet_cfg = cfg.models.judge_consistency
    agent_cfg = cfg.models.agent  # tiebreaker

    judge_opus = AnthropicClient(
        model=opus_cfg.model, temperature=opus_cfg.temperature,
        max_tokens=opus_cfg.max_tokens,
    )
    judge_sonnet = AnthropicClient(
        model=sonnet_cfg.model, temperature=sonnet_cfg.temperature,
        max_tokens=sonnet_cfg.max_tokens,
    )
    tiebreaker = AzureOpenAIClient(
        deployment=agent_cfg.deployment, temperature=agent_cfg.temperature,
        max_tokens=agent_cfg.max_tokens,
    )

    t0 = time.time()
    result: dict = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "n_borderline": len(borderline),
    }

    # --- Hallucination rate ---------------------------------------------
    print("\n===== hallucination_rate (Claude Opus judge) =====")
    hr = reasoning.hallucination_rate(borderline, judge_chat=judge_opus.chat)
    result["hallucination_rate"] = hr
    print(f"  hallucination_rate = {hr:.3f}")

    # --- Multi-step consistency -----------------------------------------
    print("\n===== multi_step_consistency (Claude Sonnet judge) =====")
    mc = reasoning.multi_step_consistency(borderline, judge_chat=judge_sonnet.chat)
    result["multi_step_consistency"] = mc
    print(f"  multi_step_consistency = {mc:.3f}")

    # --- Two-judge borderline eval + Cohen's κ --------------------------
    print("\n===== two_judge_eval (Opus + Sonnet + gpt-5 tiebreaker) =====")
    two_judge = llm_eval.run_llm_eval(
        borderline,
        judge_opus=judge_opus,
        judge_sonnet=judge_sonnet,
        tiebreaker=tiebreaker,
        k_samples=len(borderline),
    )
    result["two_judge_eval"] = {
        "n": two_judge["n"],
        "cohens_kappa": two_judge["kappa"],
        "final_labels": two_judge["labels"],
    }
    print(f"  n = {two_judge['n']}")
    print(f"  Cohen's κ (Opus vs Sonnet):")
    for axis, k in two_judge["kappa"].items():
        print(f"    {axis:12} κ = {k:.3f}")

    # --- Persist --------------------------------------------------------
    duration = time.time() - t0
    result["ended_at"] = datetime.now(timezone.utc).isoformat()
    result["duration_s"] = duration
    DELIVERABLE.mkdir(parents=True, exist_ok=True)
    JUDGES_JSON.write_text(json.dumps(result, indent=2, default=str),
                           encoding="utf-8")

    print(f"\n===== DONE ({duration:.0f}s) =====")
    print(f"→ {JUDGES_JSON}")


if __name__ == "__main__":
    main()
