"""run_provider_drift.py — F1 permissive-filter experiment across providers.

Motivation (F1 in the reviewer red-team). §7.2 argues that INFOSHIELD's
mechanism-substitution wins on `tool_poisoning`, `memory_poisoning`, and
`mem_role_confusion` matter *iff* the deployment cannot trust the provider
filter to be stable across policy updates. To move that claim from hypothetical
to empirical we re-run the same three attacks with the primary agent (planner /
worker / synthesizer) hosted by a NON-Azure provider that has no server-side
content filter. Two providers are supported today:

  * `--provider groq`      (default): agent = `llama-3.3-70b-versatile`
                           Verifier: Claude Sonnet 4.5 (unchanged from main/).
                           Traces at `results/provider_drift_groq/`.

  * `--provider anthropic`:            agent = `claude-haiku-4-5`
                           Verifier: Claude Sonnet 4.5 (unchanged).
                           Traces at `results/provider_drift_anthropic/`.

Both experiments share the same 6 HotpotQA samples, same 3 attacks, same 2
systems, and the same verifier — so per-cell attribution deltas isolate the
effect of the agent model + its content-filter layer.

Design.
  * Attacks tested:  `tool_poisoning`, `memory_poisoning`, `mem_role_confusion`
                     (the three mechanism-substitution cases from §6.2 + §6.14)
  * Systems:         `undefended`, `infoshield`
  * Sample count:    n=6 per cell → 36 runs per provider
  * Verifier:        UNCHANGED (Claude Sonnet 4.5) — verifier is not the
                     factor under test; we only vary the agent side.
  * Cost per batch:  Groq ~$4, Anthropic ~$5

Usage:
    python scripts/run_provider_drift.py                          # Groq/Llama-3.3-70B, ~$4
    python scripts/run_provider_drift.py --provider anthropic     # Anthropic/Haiku-4.5, ~$5
    python scripts/run_provider_drift.py --provider groq --model llama-3.1-8b-instant  # cheaper Groq model
    python scripts/run_provider_drift.py --provider anthropic --model claude-sonnet-4-5-20250929  # override
    python scripts/run_provider_drift.py --dry-run                # show plan for current --provider

After both provider batches complete, the comparison table
(`paper/deliverable/provider_drift_report.json`) automatically includes all
`provider_drift_*` experiments discovered on disk — no code changes needed
to add a third or fourth provider (just implement its client in
`src/infoshield/llm/` and add a case in `_override_agent()`).
"""
from __future__ import annotations

import argparse
import json
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

# Rough per-run cost by provider (agent side only; verifier stays Sonnet ~$0.05):
#   groq/llama-3.3-70b:  ~$0.03  (very cheap on Groq's tier)
#   anthropic/haiku-4.5: ~$0.05  (Haiku is cheaper Anthropic tier)
COST_PER_RUN_USD = {
    "groq": 0.10,
    "anthropic": 0.15,
}

DEFAULT_ATTACKS = ["tool_poisoning", "memory_poisoning", "mem_role_confusion"]
DEFAULT_SYSTEMS = ["undefended", "infoshield"]

# Per-provider defaults (agent side; verifier is always Sonnet from config.yaml).
PROVIDER_DEFAULTS: dict[str, dict] = {
    "groq": {
        "model": "llama-3.3-70b-versatile",
        "experiment": "provider_drift_groq",
        "label": "groq_llama",
        "filter_note": "NO Azure content filter; only Llama's own RLHF",
    },
    "anthropic": {
        "model": "claude-haiku-4-5",
        "experiment": "provider_drift_anthropic",
        "label": "anthropic_haiku",
        "filter_note": "Anthropic's own safety training (no Azure filter)",
    },
}


def _log(event: dict) -> None:
    DELIVERABLE.mkdir(parents=True, exist_ok=True)
    event = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    with open(PROGRESS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _override_agent(cfg, provider: str, model: str) -> None:
    """Mutate cfg in place so planner/worker/synthesizer use the chosen provider.

    Verifier is intentionally NOT touched: keeping the same verifier isolates
    the agent-side filter as the only changed variable between this experiment
    and the main results table.
    """
    if provider == "groq":
        cfg.models.agent.provider = "groq"
        cfg.models.agent.model = model
        cfg.models.agent.deployment = None
    elif provider == "anthropic":
        cfg.models.agent.provider = "anthropic"
        cfg.models.agent.model = model
        cfg.models.agent.deployment = None
    else:
        raise ValueError(f"unsupported provider {provider!r}")


def _est_new_runs(cfg, experiment: str, system: str, attacks: list[str],
                  n_samples: int, seed: int) -> int:
    samples = load_dataset("hotpot_qa", n_samples=n_samples, seed=seed)
    root = Path(cfg.output_root)
    new = 0
    for attack in attacks:
        for s in samples:
            dest = (
                root / experiment / system / attack / f"seed_{seed}" / f"{s['sample_id']}.json"
            )
            if not dest.exists():
                new += 1
    return new


def _print_plan(plan: list[dict], attacks: list[str], provider: str, model: str,
                experiment: str, n_samples: int, seed: int) -> tuple[int, float]:
    total_new = sum(p["new_runs"] for p in plan)
    total_cost = sum(p["est_cost"] for p in plan)
    print("\n===== run_provider_drift plan =====\n")
    print(f"  Provider:     {provider}")
    print(f"  Agent model:  {model}  ({PROVIDER_DEFAULTS[provider]['filter_note']})")
    print(f"  Verifier:     Claude Sonnet 4.5 (unchanged from main/)")
    print(f"  Experiment:   {experiment}")
    print(f"  Attacks:      {attacks}")
    print()
    for p in plan:
        print(f"  {p['system']:12} on {len(p['attacks'])} attacks  {p['new_runs']:3d} new runs  ~${p['est_cost']:5.2f}")
    print(f"\n  TOTAL: {total_new} new runs  ~${total_cost:.2f}")
    print(f"  n_samples: {n_samples}   seed: {seed}\n")
    return total_new, total_cost


def _emit_comparison(cfg) -> None:
    """Compare Azure (main/) vs all provider_drift_* experiments side by side.

    Automatically walks every `results/provider_drift_*` directory that exists
    so re-running under a new provider adds a new column without any code
    changes here.
    """
    from collections import Counter

    print("\n===== Azure vs alternate-provider comparison =====")
    root = Path(cfg.output_root)

    # Discover provider-drift experiments dynamically.
    experiments: list[tuple[str, str]] = [("main", "azure_gpt5")]
    for provider_dir in sorted(root.glob("provider_drift_*")):
        if not provider_dir.is_dir():
            continue
        suffix = provider_dir.name.removeprefix("provider_drift_")
        # Map dir suffix back to human-readable agent label.
        label = {
            "groq": "groq_llama",
            "anthropic": "anthropic_haiku",
        }.get(suffix, suffix)
        experiments.append((provider_dir.name, label))

    rows = []
    for attack in DEFAULT_ATTACKS:
        for system in DEFAULT_SYSTEMS:
            for exp, label in experiments:
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
                    "attack": attack, "system": system, "agent": label, "experiment": exp,
                    "n": n, "leaked": counter["leaked"],
                    "ib_blocked": counter["ib_blocked"],
                    "other": counter["other"], "asr": asr,
                })
    if not rows:
        print("  (no comparable cells)")
        return
    print(f"\n{'attack':22} {'system':12} {'agent':>18} {'n':>3} {'leak':>4} {'ib':>4} {'other':>5} {'ASR':>6}")
    for r in rows:
        print(
            f"  {r['attack']:20} {r['system']:12} {r['agent']:>18} "
            f"{r['n']:>3} {r['leaked']:>4} {r['ib_blocked']:>4} "
            f"{r['other']:>5} {r['asr']:>6.3f}"
        )
    out = DELIVERABLE / "provider_drift_report.json"
    out.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"\n[ok] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=list(PROVIDER_DEFAULTS.keys()),
                    default="groq",
                    help="Alternate agent provider (default: groq)")
    ap.add_argument("--model", default=None,
                    help="Override agent model (default: provider-specific)")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--systems", nargs="+", default=None)
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    defaults = PROVIDER_DEFAULTS[args.provider]
    model = args.model or defaults["model"]
    experiment = defaults["experiment"]
    cost_per_run = COST_PER_RUN_USD[args.provider]

    cfg = load_config()
    _override_agent(cfg, args.provider, model)

    systems = args.systems if args.systems else DEFAULT_SYSTEMS
    plan = []
    for system in systems:
        new = _est_new_runs(cfg, experiment, system, DEFAULT_ATTACKS, args.n, args.seed)
        plan.append({
            "system": system, "attacks": DEFAULT_ATTACKS,
            "new_runs": new, "est_cost": new * cost_per_run,
        })
    total_new, total_cost = _print_plan(
        plan, DEFAULT_ATTACKS, args.provider, model, experiment, args.n, args.seed
    )

    if args.dry_run:
        return

    _log({
        "kind": "run_provider_drift_start",
        "plan": plan, "n_samples": args.n, "seed": args.seed,
        "experiment": experiment, "provider": args.provider, "model": model,
    })
    print(f"===== running (started {datetime.now(timezone.utc).isoformat()}) =====")

    orchestrator_t0 = time.time()
    samples = load_dataset("hotpot_qa", n_samples=args.n, seed=args.seed)
    all_stats: dict[str, dict] = {}

    for system in systems:
        print(f"\n--- {system} @ {model} on {DEFAULT_ATTACKS} ---")
        t0 = time.time()

        def _event_cb(evt, sys=system):
            _log({"system": sys, **evt})

        stats = run_matrix(
            cfg=cfg,
            experiment=experiment,
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
    est_actual = total_done * cost_per_run

    _log({
        "kind": "run_provider_drift_end",
        "total_duration_s": total_duration,
        "total_runs_done": total_done,
        "est_cost_usd": est_actual,
        "experiment": experiment,
    })

    print(
        f"\n===== run_provider_drift DONE "
        f"({total_duration:.0f}s, {total_done} runs, ~${est_actual:.2f}) ====="
    )

    if not args.no_report:
        _emit_comparison(cfg)

    print(f"\nTraces at: {Path(cfg.output_root) / experiment}")


if __name__ == "__main__":
    main()
