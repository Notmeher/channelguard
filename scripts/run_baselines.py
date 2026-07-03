"""run_baselines.py — Tier 1 fix for Weakness #1 (main safety table only had 2 systems).

Adds three registered competitive baselines to the `main` experiment:

  - ibprotector       (Zhu et al. NeurIPS'24; IB-0 gate only, INFOSHIELD's phrase-bank
                       approach applied at the input boundary only)
  - llm_guard         (Claude Sonnet classifier at IB-0; stand-in for Llama Guard 3)
  - perplexity_filter (Alon & Kamfonas 2023; GPT-2 perplexity threshold at IB-0.
                       Downloads gpt2 (~500 MB) on first use; local inference,
                       zero API cost for the gate itself.)

The two headline attacks (prompt_injection_strong, adaptive_paraphrase) are used
so the new baselines are directly comparable to the existing undefended / infoshield
rows on the same attacks.

SmoothLLM is deliberately EXCLUDED because its K=10 input perturbations multiply
per-query cost by ~10x (paper standard K=10), blowing the small-budget scope of
this script. Include in a follow-up when budget allows.

After the runs, this script invokes `scripts/run_final.py --report-only` to
refresh paper/deliverable/FINAL.md. The main safety table and attribution table
group traces by (system, attack) directly from disk, so the new baseline rows
appear automatically — no changes to run_final.py needed.

Usage:
    python scripts/run_baselines.py                    # runs all 3 baselines at n=6
    python scripts/run_baselines.py --n 8              # bump to n=8 (~$12)
    python scripts/run_baselines.py --dry-run          # show plan + est cost
    python scripts/run_baselines.py --skip-perplexity  # skip GPT-2 download
    python scripts/run_baselines.py --no-report        # do not refresh FINAL.md
    python scripts/run_baselines.py --systems ibprotector llm_guard    # subset
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from infoshield.config import load_config
from infoshield.datasets import load as load_dataset
from infoshield.runner import run_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
DELIVERABLE = REPO_ROOT / "paper" / "deliverable"
PROGRESS_JSONL = DELIVERABLE / "progress.jsonl"

# Empirical cost/run from prior runs (~4000 tokens per pipeline call).
COST_PER_RUN_USD = 0.25
# llm_guard adds a short Claude Sonnet classifier call at IB-0 (max_tokens=16).
LLM_GUARD_EXTRA_USD = 0.03

ATTACKS = ["prompt_injection_strong", "adaptive_paraphrase"]

# (system_name, one-line description, estimated cost per run)
BASELINES = [
    ("ibprotector",
     "IB-0 gate only (Zhu et al. NeurIPS'24)",
     COST_PER_RUN_USD),
    ("llm_guard",
     "Claude Sonnet classifier at IB-0 (stand-in for Llama Guard 3)",
     COST_PER_RUN_USD + LLM_GUARD_EXTRA_USD),
    ("perplexity_filter",
     "GPT-2 perplexity threshold at IB-0 (Alon & Kamfonas 2023)",
     COST_PER_RUN_USD),
]


# ---------------------------------------------------------------- Progress log


def _log(event: dict) -> None:
    """Append one timestamped JSON line to progress.jsonl."""
    DELIVERABLE.mkdir(parents=True, exist_ok=True)
    event = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    with open(PROGRESS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


# --------------------------------------------------------------- Cost estimate


def _est_new_runs(cfg, system: str, attacks: list[str], n_samples: int, seed: int) -> int:
    """Count how many (sample, attack) cells for `system` do NOT already have a trace."""
    samples = load_dataset("hotpot_qa", n_samples=n_samples, seed=seed)
    root = Path(cfg.output_root)
    new = 0
    for attack in attacks:
        for s in samples:
            dest = (
                root / "main" / system / attack / f"seed_{seed}" / f"{s['sample_id']}.json"
            )
            if not dest.exists():
                new += 1
    return new


def _print_plan(plan: list[dict], attacks: list[str], n_samples: int, seed: int) -> tuple[int, float]:
    total_new = sum(p["new_runs"] for p in plan)
    total_cost = sum(p["est_cost"] for p in plan)
    print("\n===== run_baselines plan =====\n")
    for p in plan:
        print(
            f"  {p['system']:20}  {p['new_runs']:3d} new runs  "
            f"~${p['est_cost']:5.2f}   ({p['desc']})"
        )
    print(f"\n  TOTAL: {total_new} new runs  ~${total_cost:.2f}")
    print(f"  Attacks: {attacks}   n_samples: {n_samples}   seed: {seed}")
    print("  (SmoothLLM excluded: K=10 perturbations → ~10x cost per query.)")
    print()
    return total_new, total_cost


# ---------------------------------------------------------------- Main driver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="Samples per (system, attack) cell.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="Show plan + est cost only.")
    ap.add_argument(
        "--skip-perplexity", action="store_true",
        help="Skip perplexity_filter (avoids one-time GPT-2 download).",
    )
    ap.add_argument(
        "--no-report", action="store_true",
        help="Do not regenerate paper/deliverable/FINAL.md at the end.",
    )
    ap.add_argument(
        "--systems", nargs="+", default=None,
        help="Subset of baselines to run. Default: all three.",
    )
    args = ap.parse_args()

    cfg = load_config()

    # Filter baselines
    active = BASELINES
    if args.skip_perplexity:
        active = [b for b in active if b[0] != "perplexity_filter"]
    if args.systems:
        active = [b for b in active if b[0] in args.systems]

    if not active:
        print("[error] no baselines selected.")
        sys.exit(1)

    # Estimate
    plan = []
    for system, desc, cost_per_run in active:
        new = _est_new_runs(cfg, system, ATTACKS, args.n, args.seed)
        plan.append({
            "system": system, "desc": desc,
            "new_runs": new, "est_cost": new * cost_per_run,
        })

    _print_plan(plan, ATTACKS, args.n, args.seed)

    if args.dry_run:
        return

    _log({
        "kind": "run_baselines_start",
        "plan": plan,
        "n_samples": args.n,
        "seed": args.seed,
    })
    print(f"===== running (started {datetime.now(timezone.utc).isoformat()}) =====")

    orchestrator_t0 = time.time()
    samples = load_dataset("hotpot_qa", n_samples=args.n, seed=args.seed)
    all_stats: dict[str, dict] = {}

    for system, desc, _cost in active:
        print(f"\n--- {system} on {ATTACKS} ---")
        t0 = time.time()

        def _event_cb(evt, sys=system):
            _log({"system": sys, **evt})

        stats = run_matrix(
            cfg=cfg,
            experiment="main",
            dataset_name="hotpot_qa",
            samples=samples,
            systems=[system],
            attacks=ATTACKS,
            seeds=[args.seed],
            resume=True,
            on_error="log",
            progress_cb=lambda line: print(f"  {line}"),
            event_cb=_event_cb,
        )
        duration = time.time() - t0
        all_stats[system] = {"stats": stats, "duration_s": duration}
        print(
            f"  {system}: done={stats['done']} skipped={stats['skipped']} "
            f"failed={stats['failed']}  ({duration:.0f}s)"
        )

    total_duration = time.time() - orchestrator_t0
    total_done = sum(a["stats"]["done"] for a in all_stats.values())
    total_failed = sum(a["stats"]["failed"] for a in all_stats.values())
    # Rough post-hoc cost (average, not per-baseline exact)
    est_actual = total_done * COST_PER_RUN_USD

    _log({
        "kind": "run_baselines_end",
        "total_duration_s": total_duration,
        "total_runs_done": total_done,
        "total_runs_failed": total_failed,
        "est_cost_usd": est_actual,
    })

    print(
        f"\n===== run_baselines DONE "
        f"({total_duration:.0f}s, {total_done} runs, ~${est_actual:.2f}) ====="
    )

    if not args.no_report:
        print("\n===== refreshing FINAL.md via `run_final.py --report-only` =====")
        subprocess.run(
            [sys.executable, "scripts/run_final.py", "--report-only"],
            cwd=REPO_ROOT, check=False,
        )

    print(f"\nSee {DELIVERABLE / 'FINAL.md'}")


if __name__ == "__main__":
    main()
