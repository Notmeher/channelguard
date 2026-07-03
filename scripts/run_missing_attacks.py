"""run_missing_attacks.py — fill the 3 attack families not yet evaluated.

Runs the three multi-agent-specific attack families that are implemented in
`src/infoshield/attacks/` but were skipped from earlier experiments:

  - memory_poisoning          — pre-populate shared memory with 5 adversarial
                                entries; tests whether MemoryController's
                                read-time rescore hook (IB-3-mem) catches them
  - reasoning_state_injection — append an adversarial suffix to each subtask
                                after IB-1 sees it; tests IB-2 (worker output)
  - consensus_collapse        — prepend a directive forcing all workers toward
                                the same unsafe answer; tests IB-2 + IB-5

Only tests undefended vs infoshield at n=6 (2 systems × 3 attacks × 6 = 36
runs, ~$9). Baselines are excluded — they are IB-0-only and architecturally
cannot help against inter-agent-boundary attacks (established in the
tool_poisoning attribution table).

After the runs, invokes `scripts/run_final.py --report-only` to refresh
paper/deliverable/FINAL.md — the main safety table and attribution table
automatically pick up the new attack rows.

Usage:
    python scripts/run_missing_attacks.py                  # runs all 3 at n=6
    python scripts/run_missing_attacks.py --n 15           # bump per-cell n
    python scripts/run_missing_attacks.py --dry-run        # show plan + est cost
    python scripts/run_missing_attacks.py --attacks memory_poisoning  # subset
    python scripts/run_missing_attacks.py --systems undefended  # subset
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

COST_PER_RUN_USD = 0.25

DEFAULT_ATTACKS = ["memory_poisoning", "reasoning_state_injection", "consensus_collapse"]
DEFAULT_SYSTEMS = ["undefended", "infoshield"]

VALID_ATTACKS = {
    "prompt_injection_strong", "prompt_injection_medium", "prompt_injection_weak",
    "tool_poisoning", "memory_poisoning", "reasoning_state_injection",
    "consensus_collapse", "adaptive_paraphrase", "none",
}


# ---------------------------------------------------------------- Progress log


def _log(event: dict) -> None:
    DELIVERABLE.mkdir(parents=True, exist_ok=True)
    event = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    with open(PROGRESS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _est_new_runs(cfg, system: str, attacks: list[str], n_samples: int, seed: int) -> int:
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
    print("\n===== run_missing_attacks plan =====\n")
    for p in plan:
        print(f"  {p['system']:12} on {len(p['attacks'])} attacks  {p['new_runs']:3d} new runs  ~${p['est_cost']:5.2f}")
    print(f"\n  TOTAL: {total_new} new runs  ~${total_cost:.2f}")
    print(f"  Attacks: {attacks}")
    print(f"  Systems: {[p['system'] for p in plan]}")
    print(f"  n_samples: {n_samples}   seed: {seed}\n")
    return total_new, total_cost


# ---------------------------------------------------------------- Main driver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="Samples per (system, attack) cell.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--attacks", nargs="+", default=None,
        help="Subset of attacks. Default: memory_poisoning + reasoning_state_injection + consensus_collapse.",
    )
    ap.add_argument(
        "--systems", nargs="+", default=None,
        help="Subset of systems. Default: undefended + infoshield.",
    )
    ap.add_argument(
        "--no-report", action="store_true",
        help="Do not regenerate paper/deliverable/FINAL.md at the end.",
    )
    args = ap.parse_args()

    cfg = load_config()

    attacks = args.attacks if args.attacks else DEFAULT_ATTACKS
    unknown = set(attacks) - VALID_ATTACKS
    if unknown:
        print(f"[error] unknown attacks: {sorted(unknown)}")
        sys.exit(1)
    systems = args.systems if args.systems else DEFAULT_SYSTEMS

    plan = []
    for system in systems:
        new = _est_new_runs(cfg, system, attacks, args.n, args.seed)
        plan.append({
            "system": system, "attacks": attacks,
            "new_runs": new, "est_cost": new * COST_PER_RUN_USD,
        })
    total_new, total_cost = _print_plan(plan, attacks, args.n, args.seed)

    if args.dry_run:
        return

    _log({
        "kind": "run_missing_attacks_start",
        "plan": plan, "n_samples": args.n, "seed": args.seed,
    })
    print(f"===== running (started {datetime.now(timezone.utc).isoformat()}) =====")

    orchestrator_t0 = time.time()
    samples = load_dataset("hotpot_qa", n_samples=args.n, seed=args.seed)
    all_stats: dict[str, dict] = {}

    for system in systems:
        print(f"\n--- {system} on {attacks} ---")
        t0 = time.time()

        def _event_cb(evt, sys=system):
            _log({"system": sys, **evt})

        stats = run_matrix(
            cfg=cfg,
            experiment="main",
            dataset_name="hotpot_qa",
            samples=samples,
            systems=[system],
            attacks=attacks,
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
    est_actual = total_done * COST_PER_RUN_USD

    _log({
        "kind": "run_missing_attacks_end",
        "total_duration_s": total_duration,
        "total_runs_done": total_done,
        "est_cost_usd": est_actual,
    })

    print(
        f"\n===== run_missing_attacks DONE "
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
