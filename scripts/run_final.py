"""run_final.py — orchestrates the $40 plan end-to-end.

Runs four experiment steps in order (each resumes from disk):

  1. main / infoshield / {tool_poisoning, adaptive_paraphrase}       ~60 runs
  2. main / undefended  / {tool_poisoning}                            ~26 runs
  3. gsm8k / {undefended, infoshield} / none  @ n=15                  30 runs
  4. ablation (7 configs, prompt_injection_strong) @ n=6              42 runs

After each step, emits:
  paper/deliverable/progress.jsonl        (one JSON line per sample event)
  paper/deliverable/step_N_<name>.md      (per-step summary + metrics)

After all steps, runs consolidate.py and writes:
  paper/deliverable/FINAL.md              (overall summary)

Usage:
    python scripts/run_final.py                # runs everything
    python scripts/run_final.py --dry-run      # show what would run + est cost
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from infoshield.baselines.registry import SYSTEMS
from infoshield.config import load_config
from infoshield.datasets import load as load_dataset
from infoshield.ib.gates import build_ib_gates
from infoshield.loader import iter_traces
from infoshield.metrics import efficiency, reasoning, safety
from infoshield.pipeline.langgraph_pipeline import build_pipeline
from infoshield.runner import run_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
DELIVERABLE = REPO_ROOT / "paper" / "deliverable"
PROGRESS_JSONL = DELIVERABLE / "progress.jsonl"

# Empirical cost/run from TASKS.md §9. Actual variance ±30%.
COST_PER_RUN_USD = 0.25


# ---------------------------------------------------------------- Progress log


def _log(event: dict) -> None:
    """Append one timestamped JSON line to progress.jsonl."""
    DELIVERABLE.mkdir(parents=True, exist_ok=True)
    event = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    with open(PROGRESS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _reset_progress_log() -> None:
    """Truncate progress.jsonl at the start of a new run_final invocation."""
    DELIVERABLE.mkdir(parents=True, exist_ok=True)
    PROGRESS_JSONL.write_text("", encoding="utf-8")


# --------------------------------------------------------------- Ablation prep


def _register_ablation_systems(cfg) -> list[str]:
    """Register the six 'infoshield_no_ib<i>' systems used by the ablation."""
    names: list[str] = ["infoshield"]
    for drop in range(6):
        name = f"infoshield_no_ib{drop}"
        if name in SYSTEMS:
            names.append(name)
            continue
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy.ib.active_layers = [i for i in cfg.ib.active_layers if i != drop]
        gates = build_ib_gates(cfg_copy)

        def _make_builder(g=gates):
            def _b(_cfg):
                return build_pipeline(_cfg, ib_gates=g)
            return _b

        SYSTEMS[name] = _make_builder()
        names.append(name)
    return names


# ---------------------------------------------------------------- Step runner


def _run_step(
    step_num: int,
    step_name: str,
    description: str,
    cfg,
    experiment: str,
    dataset_name: str,
    n_samples: int,
    seeds: list[int],
    systems: list[str],
    attacks: list[str],
) -> dict:
    """Run one step and emit its per-step Markdown summary."""
    print(f"\n===== Step {step_num}: {step_name} =====")
    print(f"     {description}")
    print(f"     systems={systems}  attacks={attacks}  n={n_samples}  seeds={seeds}")

    _log({
        "kind": "step_start",
        "step": step_num,
        "name": step_name,
        "description": description,
        "experiment": experiment,
        "systems": systems,
        "attacks": attacks,
        "n_samples": n_samples,
        "seeds": seeds,
    })

    samples = load_dataset(dataset_name, n_samples=n_samples, seed=seeds[0])
    t0 = time.time()

    def _event_cb(evt):
        _log({"step": step_num, "name": step_name, **evt})

    stats = run_matrix(
        cfg=cfg,
        experiment=experiment,
        dataset_name=dataset_name,
        samples=samples,
        systems=systems,
        attacks=attacks,
        seeds=seeds,
        resume=True,
        on_error="log",
        progress_cb=lambda line: print(f"  {line}"),
        event_cb=_event_cb,
    )
    duration_s = time.time() - t0
    est_cost = stats["done"] * COST_PER_RUN_USD

    _log({
        "kind": "step_end",
        "step": step_num,
        "name": step_name,
        "duration_s": duration_s,
        "stats": stats,
        "est_cost_usd": est_cost,
    })

    _write_step_md(
        step_num=step_num,
        step_name=step_name,
        description=description,
        cfg=cfg,
        experiment=experiment,
        systems=systems,
        attacks=attacks,
        n_samples=n_samples,
        seeds=seeds,
        stats=stats,
        duration_s=duration_s,
        est_cost=est_cost,
    )

    print(
        f"     done={stats['done']} skipped={stats['skipped']} failed={stats['failed']}  "
        f"({duration_s:.0f}s, est ${est_cost:.2f})"
    )
    return {"stats": stats, "duration_s": duration_s, "est_cost_usd": est_cost}


# -------------------------------------------------------------- Metric helpers


def _cell_table(
    cfg,
    experiment: str,
    group: tuple[str, ...],
    metric_fns: dict[str, callable],
) -> list[dict]:
    """Group traces of an experiment by `group` and compute metrics per cell."""
    traces = list(iter_traces(cfg.output_root, experiment=experiment))
    buckets: dict[tuple, list] = {}
    for t in traces:
        key = tuple(getattr(t, g) for g in group)
        buckets.setdefault(key, []).append(t)
    rows = []
    for key, tl in sorted(buckets.items()):
        row = {**dict(zip(group, key)), "n": len(tl)}
        for mname, mfn in metric_fns.items():
            row[mname] = mfn(tl)
        rows.append(row)
    return rows


def _write_step_md(
    *,
    step_num: int,
    step_name: str,
    description: str,
    cfg,
    experiment: str,
    systems: list[str],
    attacks: list[str],
    n_samples: int,
    seeds: list[int],
    stats: dict,
    duration_s: float,
    est_cost: float,
) -> Path:
    """Write paper/deliverable/step_N_<name>.md with the per-step summary."""
    out = DELIVERABLE / f"step_{step_num}_{step_name}.md"
    lines: list[str] = []
    lines.append(f"# Step {step_num} — {step_name}")
    lines.append("")
    lines.append(description)
    lines.append("")
    lines.append(f"- **Experiment**: `{experiment}`")
    lines.append(f"- **Systems**: {', '.join(f'`{s}`' for s in systems)}")
    lines.append(f"- **Attacks**: {', '.join(f'`{a}`' for a in attacks)}")
    lines.append(f"- **n_samples**: {n_samples}  |  **seeds**: {seeds}")
    lines.append("")
    lines.append("## Runtime")
    lines.append("")
    lines.append(f"- runs done: **{stats['done']}**")
    lines.append(f"- runs skipped (resumed): {stats['skipped']}")
    lines.append(f"- runs failed: {stats['failed']}")
    lines.append(f"- wall-clock: **{duration_s:.0f} s** ({duration_s/60:.1f} min)")
    lines.append(f"- estimated cost: **${est_cost:.2f}**  _(at ${COST_PER_RUN_USD}/run)_")
    lines.append("")

    # Per-cell metrics for THIS experiment
    metric_fns = {
        "asr": safety.asr,
        "ctca": reasoning.ctca,
        "mean_latency_ms": efficiency.mean_latency_ms,
        "mean_tokens": efficiency.mean_tokens,
    }
    rows = _cell_table(cfg, experiment, ("system", "attack", "seed"), metric_fns)
    lines.append("## Per-cell metrics (all traces on disk for this experiment)")
    lines.append("")
    lines.append("| System | Attack | Seed | n | ASR | CTCA | Mean latency (ms) | Mean tokens |")
    lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        lines.append(
            f"| `{r['system']}` | `{r['attack']}` | {r['seed']} | {r['n']} | "
            f"{r['asr']:.3f} | {r['ctca']:.3f} | {r['mean_latency_ms']:.0f} | {r['mean_tokens']:.0f} |"
        )
    lines.append("")
    lines.append(f"_Written at {datetime.now(timezone.utc).isoformat()}._")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"     -> {out}")
    return out


# ---------------------------------------------------------------- Final report


def _write_final_md(step_results: list[dict]) -> Path:
    """Write paper/deliverable/FINAL.md aggregating all four steps."""
    cfg = load_config()
    out = DELIVERABLE / "FINAL.md"

    total_done = sum(r["stats"]["done"] for r in step_results)
    total_skipped = sum(r["stats"]["skipped"] for r in step_results)
    total_failed = sum(r["stats"]["failed"] for r in step_results)
    total_duration = sum(r["duration_s"] for r in step_results)
    total_cost = sum(r["est_cost_usd"] for r in step_results)

    lines: list[str] = []
    lines.append("# INFOSHIELD — Final Run Report")
    lines.append("")
    lines.append(f"Auto-generated by `scripts/run_final.py` at {datetime.now(timezone.utc).isoformat()}.")
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append(f"- Total NEW runs: **{total_done}**")
    lines.append(f"- Resumed (skipped): {total_skipped}")
    lines.append(f"- Failed: {total_failed}")
    lines.append(f"- Total wall-clock: **{total_duration:.0f} s** ({total_duration/60:.1f} min)")
    lines.append(f"- Estimated cost: **~${total_cost:.2f}** _(at ${COST_PER_RUN_USD}/run avg)_")
    lines.append("")

    lines.append("## Per-step breakdown")
    lines.append("")
    lines.append("| # | Step | Runs | Skipped | Failed | Duration | Cost |")
    lines.append("| ---: | :--- | ---: | ---: | ---: | ---: | ---: |")
    for i, r in enumerate(step_results, 1):
        lines.append(
            f"| {i} | {r['name']} | {r['stats']['done']} | {r['stats']['skipped']} | "
            f"{r['stats']['failed']} | {r['duration_s']:.0f} s | ${r['est_cost_usd']:.2f} |"
        )
    lines.append("")

    # Headline tables computed from ALL traces on disk (post-run)
    metric_fns = {"asr": safety.asr, "ctca": reasoning.ctca}

    # Main safety table
    lines.append("## Main safety table (all `main` traces)")
    lines.append("")
    main_rows = _cell_table(cfg, "main", ("system", "attack"), metric_fns)
    if main_rows:
        systems_seen = sorted({r["system"] for r in main_rows})
        attacks_seen = sorted({r["attack"] for r in main_rows})
        header = "| System | " + " | ".join(f"`{a}`" for a in attacks_seen) + " |"
        sep = "| :--- | " + " | ".join("---:" for _ in attacks_seen) + " |"
        lines.append(header)
        lines.append(sep)
        for sysn in systems_seen:
            row = [f"`{sysn}`"]
            for atk in attacks_seen:
                match = [r for r in main_rows if r["system"] == sysn and r["attack"] == atk]
                row.append(f"{match[0]['asr']:.3f} (n={match[0]['n']})" if match else "—")
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("_(no main traces on disk)_")
    lines.append("")

    # GSM8K fidelity
    lines.append("## GSM8K fidelity (CTCA under no attack)")
    lines.append("")
    gsm_rows = _cell_table(cfg, "gsm8k", ("system",), metric_fns)
    if gsm_rows:
        lines.append("| System | n | CTCA |")
        lines.append("| :--- | ---: | ---: |")
        for r in gsm_rows:
            lines.append(f"| `{r['system']}` | {r['n']} | {r['ctca']:.3f} |")
        base_ctca = next((r["ctca"] for r in gsm_rows if r["system"] == "undefended"), None)
        inf_ctca = next((r["ctca"] for r in gsm_rows if r["system"] == "infoshield"), None)
        if base_ctca is not None and inf_ctca is not None:
            delta_pp = (inf_ctca - base_ctca) * 100
            lines.append("")
            lines.append(f"**Delta**: infoshield − undefended = **{delta_pp:+.1f} pp** on CTCA.")
    else:
        lines.append("_(no gsm8k traces on disk)_")
    lines.append("")

    # Ablation
    lines.append("## Per-layer ablation (prompt_injection_strong)")
    lines.append("")
    abl_rows = _cell_table(cfg, "ablation", ("system",), metric_fns)
    if abl_rows:
        lines.append("| Config | n | ASR | CTCA |")
        lines.append("| :--- | ---: | ---: | ---: |")
        for r in abl_rows:
            lines.append(f"| `{r['system']}` | {r['n']} | {r['asr']:.3f} | {r['ctca']:.3f} |")
    else:
        lines.append("_(no ablation traces on disk)_")
    lines.append("")

    # Overall coverage
    all_traces = list(iter_traces(cfg.output_root))
    lines.append("## Overall coverage")
    lines.append("")
    lines.append(f"- Total traces on disk (all experiments): **{len(all_traces)}**")
    counts: dict[str, int] = {}
    for t in all_traces:
        counts[t.experiment] = counts.get(t.experiment, 0) + 1
    lines.append("")
    lines.append("| Experiment | # traces |")
    lines.append("| :--- | ---: |")
    for exp in sorted(counts):
        lines.append(f"| `{exp}` | {counts[exp]} |")
    lines.append("")

    lines.append("## Files")
    lines.append("")
    lines.append("- `FINAL.md` — this report")
    lines.append("- `RESULTS.md` — detailed results after consolidator")
    lines.append("- `numbers.json` — machine-readable headline numbers")
    lines.append("- `progress.jsonl` — one JSON line per sample/event during the run")
    lines.append("- `step_1_*.md` .. `step_4_*.md` — per-step summaries")
    lines.append("- `figures/*.pdf`, `figures/*.csv`, `figures/failure_gallery.md`")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n===> {out}")
    return out


# ---------------------------------------------------------------- Main driver


STEPS = [
    {
        "name": "main_infoshield_holes",
        "description": (
            "Fill the two missing cells in the main safety table: "
            "`infoshield` under `tool_poisoning` and `adaptive_paraphrase`."
        ),
        "experiment": "main",
        "dataset": "hotpot_qa",
        "systems": ["infoshield"],
        "attacks": ["tool_poisoning", "adaptive_paraphrase"],
        "n_samples": 30,
        "seeds": [0],
    },
    {
        "name": "main_undefended_topup",
        "description": (
            "Top up `undefended` / `tool_poisoning` from n=4 to n=30 so the "
            "row is symmetric with the infoshield row."
        ),
        "experiment": "main",
        "dataset": "hotpot_qa",
        "systems": ["undefended"],
        "attacks": ["tool_poisoning"],
        "n_samples": 30,
        "seeds": [0],
    },
    {
        "name": "gsm8k_fidelity",
        "description": (
            "GSM8K under no attack for both systems. Proves INFOSHIELD does "
            "not degrade multi-step reasoning (CTCA fidelity)."
        ),
        "experiment": "gsm8k",
        "dataset": "gsm8k",
        "systems": ["undefended", "infoshield"],
        "attacks": ["none"],
        "n_samples": 15,
        "seeds": [0],
    },
    {
        "name": "ablation_per_gate",
        "description": (
            "Seven configs (full INFOSHIELD + six variants each with one "
            "IB-i gate disabled) under `prompt_injection_strong`. Justifies "
            "each of the six gates."
        ),
        "experiment": "ablation",
        "dataset": "hotpot_qa",
        "systems": None,   # populated by _register_ablation_systems
        "attacks": ["prompt_injection_strong"],
        "n_samples": 6,
        "seeds": [0],
    },
]


def _est_new_runs(cfg, step: dict, systems: list[str]) -> int:
    """Estimate how many NEW runs this step will do (subtracting existing)."""
    samples = load_dataset(step["dataset"], n_samples=step["n_samples"], seed=step["seeds"][0])
    root = Path(cfg.output_root)
    new = 0
    for seed in step["seeds"]:
        for system in systems:
            for attack in step["attacks"]:
                for s in samples:
                    dest = (
                        root / step["experiment"] / system / attack /
                        f"seed_{seed}" / f"{s['sample_id']}.json"
                    )
                    if not dest.exists():
                        new += 1
    return new


def _dry_run(cfg) -> None:
    """Show what would run without executing."""
    _register_ablation_systems(cfg)
    total_new = 0
    print("\n===== DRY RUN =====\n")
    for i, step in enumerate(STEPS, 1):
        systems = step["systems"] or ([f"infoshield_no_ib{d}" for d in range(6)] + ["infoshield"])
        step["systems"] = systems
        new = _est_new_runs(cfg, step, systems)
        total_new += new
        print(f"Step {i}: {step['name']}")
        print(f"        {step['description']}")
        print(f"        systems={systems}  attacks={step['attacks']}  n={step['n_samples']}")
        print(f"        estimated NEW runs: {new}  (~${new * COST_PER_RUN_USD:.2f})")
        print()
    print(f"TOTAL estimated new runs: {total_new}  (~${total_new * COST_PER_RUN_USD:.2f})")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Show plan + est cost, do not run.")
    ap.add_argument(
        "--skip-consolidate", action="store_true",
        help="Do not invoke scripts/consolidate.py at the end.",
    )
    args = ap.parse_args()

    cfg = load_config()

    if args.dry_run:
        _dry_run(cfg)
        return

    _reset_progress_log()
    print(f"\n===== run_final ({datetime.now(timezone.utc).isoformat()}) =====")
    _log({"kind": "orchestrator_start", "steps": [s["name"] for s in STEPS]})

    # Register the ablation systems ONCE up front (used by step 4).
    ablation_systems = _register_ablation_systems(cfg)

    step_results: list[dict] = []
    orchestrator_t0 = time.time()

    for i, step in enumerate(STEPS, 1):
        if step["systems"] is None:
            step_systems = ablation_systems
        else:
            step_systems = step["systems"]
        result = _run_step(
            step_num=i,
            step_name=step["name"],
            description=step["description"],
            cfg=cfg,
            experiment=step["experiment"],
            dataset_name=step["dataset"],
            n_samples=step["n_samples"],
            seeds=step["seeds"],
            systems=step_systems,
            attacks=step["attacks"],
        )
        result["name"] = step["name"]
        step_results.append(result)

    total_duration = time.time() - orchestrator_t0
    total_cost = sum(r["est_cost_usd"] for r in step_results)
    _log({
        "kind": "orchestrator_end",
        "total_duration_s": total_duration,
        "total_est_cost_usd": total_cost,
    })

    # Run consolidator to refresh RESULTS.md, numbers.json, figures.
    if not args.skip_consolidate:
        print("\n===== consolidator =====")
        try:
            subprocess.run([sys.executable, "scripts/consolidate.py"], cwd=REPO_ROOT, check=False)
        except Exception as e:
            print(f"[warn] consolidator failed: {e}")

    # Emit the top-level FINAL.md
    _write_final_md(step_results)

    print(
        f"\n===== DONE ({total_duration:.0f}s, est ${total_cost:.2f}) ====="
    )
    print(f"See {DELIVERABLE / 'FINAL.md'} for the full report.")


if __name__ == "__main__":
    main()
