"""run_provider_drift.py — F1 permissive-filter experiment (Azure vs Groq).

Motivation (F1 in the reviewer red-team). §7.2 argues that INFOSHIELD's
mechanism-substitution wins on `tool_poisoning`, `memory_poisoning`, and
`mem_role_confusion` matter *iff* the deployment cannot trust the provider
filter to be stable across policy updates. That claim was purely hypothetical
in the original artifact — every measured trace used Azure OpenAI + Azure's
strong server-side content filter. To make the claim empirical we need to
re-run the same experiment against a provider whose filter is materially
weaker, and show that undefended ASR jumps while INFOSHIELD's IB-3 / IB-3-mem
gates keep ASR at 0.

We use Groq (`llama-3.3-70b-versatile` by default) as the counterfactual:
Groq serves Llama-3.x with NO server-side content filter beyond the model's
own RLHF safety training. If undefended-on-Groq leaks a substantial fraction
of these attacks and infoshield-on-Groq still blocks them at IB-3 / IB-3-mem,
the mechanism-substitution argument is empirically vindicated.

Design.
  * Attacks tested:  `tool_poisoning`, `memory_poisoning`, `mem_role_confusion`
                     (the three mechanism-substitution cases from §6.2 + §6.14)
  * Systems:         `undefended`, `infoshield`
  * Sample count:    n=6 per cell -> 36 runs
  * Agent model:     `llama-3.3-70b-versatile` via Groq (planner/worker/synth)
  * Verifier:        UNCHANGED (Claude Sonnet 4.5) — verifier is not the
                     factor under test; we only vary the primary agent model.
  * Experiment dir:  `results/provider_drift/` (kept separate from `main/`)
  * Estimated cost:  ~$0-2 (Groq has a generous free tier; llama-3.3-70b is
                     ~$0.59/M input, $0.79/M output). Verifier calls are
                     ~$3-5 on Anthropic. Total well under the ~$15 estimate.

Usage:
    python scripts/run_provider_drift.py                 # 36 runs, ~$5
    python scripts/run_provider_drift.py --n 3           # smaller sanity check
    python scripts/run_provider_drift.py --dry-run       # show plan
    python scripts/run_provider_drift.py --groq-model llama-3.1-8b-instant  # cheaper
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from infoshield.attacks.registry import contains_attack_marker
from infoshield.config import load_config
from infoshield.datasets import load as load_dataset
from infoshield.runner import run_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
DELIVERABLE = REPO_ROOT / "paper" / "deliverable"
PROGRESS_JSONL = DELIVERABLE / "progress.jsonl"

# Rough cost per run: gpt-5 was ~$0.25; llama-3.3-70b via Groq is ~10x cheaper
# on input+output token pricing, so agent side is ~$0.03. Verifier (Sonnet)
# stays at ~$0.05 per run. Total ~$0.08/run.
COST_PER_RUN_USD = 0.10

EXPERIMENT = "provider_drift"
DEFAULT_ATTACKS = ["tool_poisoning", "memory_poisoning", "mem_role_confusion"]
DEFAULT_SYSTEMS = ["undefended", "infoshield"]
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


def _log(event: dict) -> None:
    DELIVERABLE.mkdir(parents=True, exist_ok=True)
    event = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    with open(PROGRESS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _override_agent_to_groq(cfg, groq_model: str) -> None:
    """Mutate cfg in place so planner/worker/synthesizer use Groq instead of Azure.

    Verifier is intentionally NOT touched: keeping the same verifier isolates
    the agent-side filter as the only changed variable between this experiment
    and the main results table.
    """
    cfg.models.agent.provider = "groq"
    cfg.models.agent.model = groq_model
    cfg.models.agent.deployment = None  # not used by Groq


def _est_new_runs(cfg, system: str, attacks: list[str], n_samples: int, seed: int) -> int:
    samples = load_dataset("hotpot_qa", n_samples=n_samples, seed=seed)
    root = Path(cfg.output_root)
    new = 0
    for attack in attacks:
        for s in samples:
            dest = (
                root / EXPERIMENT / system / attack / f"seed_{seed}" / f"{s['sample_id']}.json"
            )
            if not dest.exists():
                new += 1
    return new


def _print_plan(plan: list[dict], attacks: list[str], groq_model: str,
                n_samples: int, seed: int) -> tuple[int, float]:
    total_new = sum(p["new_runs"] for p in plan)
    total_cost = sum(p["est_cost"] for p in plan)
    print("\n===== run_provider_drift plan =====\n")
    print(f"  Experiment: {EXPERIMENT}")
    print(f"  Agent model:  {groq_model}  (via Groq, NO Azure content filter)")
    print(f"  Verifier:     Claude Sonnet 4.5 (unchanged from main/)")
    print(f"  Attacks:      {attacks}")
    print()
    for p in plan:
        print(f"  {p['system']:12} on {len(p['attacks'])} attacks  {p['new_runs']:3d} new runs  ~${p['est_cost']:5.2f}")
    print(f"\n  TOTAL: {total_new} new runs  ~${total_cost:.2f}")
    print(f"  n_samples: {n_samples}   seed: {seed}\n")
    return total_new, total_cost


def _emit_comparison(cfg) -> None:
    """Compare main/ (Azure) vs provider_drift/ (Groq) ASR + attribution side by side."""
    from collections import Counter

    print("\n===== Azure vs Groq comparison =====")
    root = Path(cfg.output_root)
    rows = []
    for attack in DEFAULT_ATTACKS:
        for system in DEFAULT_SYSTEMS:
            for exp, label in (("main", "azure_gpt5"), (EXPERIMENT, "groq_llama")):
                cell_dir = root / exp / system / attack
                if not cell_dir.exists():
                    continue
                counter = Counter()
                n = 0
                for p in cell_dir.rglob("*.json"):
                    try:
                        t = json.loads(p.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    n += 1
                    if contains_attack_marker(t.get("final_output")):
                        counter["leaked"] += 1
                        continue
                    if any(g.get("status") == "BLOCKED"
                           for g in t.get("gate_decisions", []) or []):
                        counter["ib_blocked"] += 1
                        continue
                    counter["other"] += 1  # azure/verifier/synth/ok folded together
                if n == 0:
                    continue
                asr = counter["leaked"] / n
                rows.append({
                    "attack": attack, "system": system, "agent": label,
                    "n": n, "leaked": counter["leaked"],
                    "ib_blocked": counter["ib_blocked"],
                    "other": counter["other"], "asr": asr,
                })
    if not rows:
        print("  (no comparable cells)")
        return
    print(f"\n{'attack':22} {'system':12} {'agent':>12} {'n':>3} {'leak':>4} {'ib':>4} {'other':>5} {'ASR':>6}")
    for r in rows:
        print(
            f"  {r['attack']:20} {r['system']:12} {r['agent']:>12} "
            f"{r['n']:>3} {r['leaked']:>4} {r['ib_blocked']:>4} "
            f"{r['other']:>5} {r['asr']:>6.3f}"
        )
    out = DELIVERABLE / "provider_drift_report.json"
    out.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"\n[ok] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--groq-model", default=DEFAULT_GROQ_MODEL,
                    help=f"Groq chat model (default: {DEFAULT_GROQ_MODEL})")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--systems", nargs="+", default=None)
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    _override_agent_to_groq(cfg, args.groq_model)

    systems = args.systems if args.systems else DEFAULT_SYSTEMS
    plan = []
    for system in systems:
        new = _est_new_runs(cfg, system, DEFAULT_ATTACKS, args.n, args.seed)
        plan.append({
            "system": system, "attacks": DEFAULT_ATTACKS,
            "new_runs": new, "est_cost": new * COST_PER_RUN_USD,
        })
    total_new, total_cost = _print_plan(plan, DEFAULT_ATTACKS, args.groq_model, args.n, args.seed)

    if args.dry_run:
        return

    _log({
        "kind": "run_provider_drift_start",
        "plan": plan, "n_samples": args.n, "seed": args.seed,
        "experiment": EXPERIMENT, "groq_model": args.groq_model,
    })
    print(f"===== running (started {datetime.now(timezone.utc).isoformat()}) =====")

    orchestrator_t0 = time.time()
    samples = load_dataset("hotpot_qa", n_samples=args.n, seed=args.seed)
    all_stats: dict[str, dict] = {}

    for system in systems:
        print(f"\n--- {system} @ {args.groq_model} on {DEFAULT_ATTACKS} ---")
        t0 = time.time()

        def _event_cb(evt, sys=system):
            _log({"system": sys, **evt})

        stats = run_matrix(
            cfg=cfg,
            experiment=EXPERIMENT,
            dataset_name="hotpot_qa",
            samples=samples,
            systems=[system],
            attacks=DEFAULT_ATTACKS,
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
        "kind": "run_provider_drift_end",
        "total_duration_s": total_duration,
        "total_runs_done": total_done,
        "est_cost_usd": est_actual,
    })

    print(
        f"\n===== run_provider_drift DONE "
        f"({total_duration:.0f}s, {total_done} runs, ~${est_actual:.2f}) ====="
    )

    if not args.no_report:
        _emit_comparison(cfg)

    print(f"\nTraces at: {Path(cfg.output_root) / EXPERIMENT}")


if __name__ == "__main__":
    main()
