"""aggregate_results.py — the ONLY path from results/ to the deliverable folder.

Walks results/*.json, computes every metric, and emits:
  paper/numbers.json              — machine-readable copy of every headline number
  paper/figures/*.csv             — data files consumed by figures/*.py
  paper/aggregation_summary.json  — coverage + consistency report

Also runs self-consistency checks:
  * per-(experiment, seed) coverage of trace files
  * per-(experiment, system, attack, seed) sample count vs. cfg.experiments[..].n_samples
  * every JSON validates against Trace (via loader.iter_traces)

Exit code:
  0 — every experiment has data at target sample count
  1 — some experiments missing or cells under-filled (aggregation still runs)
  2 — schema errors (fatal)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from infoshield.attacks.registry import contains_attack_marker
from infoshield.config import load_config
from infoshield.loader import iter_traces
from infoshield.metrics import efficiency, reasoning, safety, statistical
from infoshield.trace import Trace


# ------------------------------------------------------------------------ Metrics


def _cell_metrics(traces: list[Trace]) -> dict:
    safety_m = safety.compute_all(traces)
    eff_m = efficiency.compute_all(traces)
    reasoning_m = {"ctca": reasoning.ctca(traces)}
    n = len(traces)
    ci = statistical.bootstrap_ci(
        [
            1.0 if contains_attack_marker(t.final_output) else 0.0
            for t in traces
        ]
    ) if traces else (0.0, 0.0)
    return {
        **safety_m,
        **reasoning_m,
        "mean_latency_ms": eff_m["mean_latency_ms"],
        "mean_tokens": eff_m["mean_tokens"],
        "verifier_overhead": eff_m["verifier_overhead"],
        "gate_latency": eff_m["gate_latency"],
        "asr_ci_low": ci[0],
        "asr_ci_high": ci[1],
        "n": n,
    }


def _by_group(root: str, experiment: str, group_by: tuple[str, ...]) -> pd.DataFrame:
    traces = list(iter_traces(root, experiment=experiment))
    buckets: dict[tuple, list[Trace]] = {}
    for t in traces:
        key = tuple(getattr(t, g) for g in group_by)
        buckets.setdefault(key, []).append(t)
    rows = []
    for key, tl in sorted(buckets.items()):
        m = _cell_metrics(tl)
        rows.append({**dict(zip(group_by, key)), **{k: v for k, v in m.items() if not isinstance(v, dict)}})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ Consistency checks


def _check_coverage(cfg, out: dict) -> int:
    """Per-cell coverage report.

    Emits:
      - `missing_experiments`: experiments with 0 traces on disk
      - `underfilled_cells`: (experiment, system, attack, seed) tuples with fewer
        traces than the experiment's configured `n_samples`
    Returns 0 when both lists are empty, 1 otherwise.
    """
    missing_experiments: list[str] = []
    underfilled: list[dict] = []

    for exp_name, ecfg in cfg.experiments.items():
        traces = list(iter_traces(cfg.output_root, experiment=exp_name))
        if not traces:
            missing_experiments.append(exp_name)
            continue

        buckets: dict[tuple[str, str, int], int] = {}
        for t in traces:
            key = (t.system, t.attack, t.seed)
            buckets[key] = buckets.get(key, 0) + 1

        for (system, attack, seed), count in sorted(buckets.items()):
            if count < ecfg.n_samples:
                underfilled.append(
                    {
                        "experiment": exp_name,
                        "system": system,
                        "attack": attack,
                        "seed": seed,
                        "have": count,
                        "expected": ecfg.n_samples,
                    }
                )

    out["missing_experiments"] = missing_experiments
    out["underfilled_cells"] = underfilled
    # Human-readable summary (kept under the legacy key for backwards compat).
    out["missing_cells"] = [
        f"{c['experiment']}/{c['system']}/{c['attack']}/seed_{c['seed']} "
        f"(have {c['have']}/{c['expected']})"
        for c in underfilled
    ] + [f"{name}/*" for name in missing_experiments]

    if missing_experiments:
        print(f"[warn] {len(missing_experiments)} experiment(s) have 0 traces: {missing_experiments}")
    if underfilled:
        print(f"[warn] {len(underfilled)} (system, attack, seed) cell(s) under-filled")

    return 0 if (not missing_experiments and not underfilled) else 1


# ------------------------------------------------------------------------ Main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default="paper")
    args = ap.parse_args()

    cfg = load_config()
    paper = Path(args.paper)
    paper_figs = paper / "figures"
    paper_figs.mkdir(parents=True, exist_ok=True)

    numbers: dict = {}
    checks: dict = {}
    exit_code = _check_coverage(cfg, checks)

    # ---------------- Main table ----------------
    main_df = _by_group(cfg.output_root, "main", ("system", "attack"))
    if not main_df.empty:
        main_df.to_csv(paper_figs / "fig1_asr.csv", index=False)
        for _, row in main_df.iterrows():
            numbers.setdefault("main", {}).setdefault(row["system"], {})[row["attack"] + "_asr"] = float(row["asr"])
            numbers["main"][row["system"]][row["attack"] + "_ctca"] = float(row.get("ctca", 0.0))

    # ---------------- Reasoning fidelity ----------------
    gsm_df = _by_group(cfg.output_root, "gsm8k", ("system",))
    if not gsm_df.empty:
        gsm_df[["system", "ctca", "n"]].to_csv(paper_figs / "reasoning_fidelity.csv", index=False)
        for _, row in gsm_df.iterrows():
            numbers.setdefault("gsm8k", {})[row["system"]] = {"ctca": float(row["ctca"])}

    # ---------------- Ablation ----------------
    abl_df = _by_group(cfg.output_root, "ablation", ("system",))
    if not abl_df.empty:
        abl_df = abl_df.rename(columns={"system": "config"})
        abl_df[["config", "asr", "ctca"]].to_csv(paper_figs / "fig3_ablation.csv", index=False)
        for _, row in abl_df.iterrows():
            numbers.setdefault("ablation", {})[row["config"]] = {
                "asr": float(row["asr"]),
                "ctca": float(row["ctca"]),
            }

    # ---------------- Beta sweep ----------------
    beta_df = _by_group(cfg.output_root, "beta_sweep", ("system",))
    if not beta_df.empty:
        beta_df["beta"] = beta_df["system"].str.extract(r"beta_([\d.]+)").astype(float)
        beta_df.dropna(subset=["beta"]).sort_values("beta")[["beta", "asr", "ctca"]].to_csv(
            paper_figs / "fig2_beta.csv", index=False
        )
        for _, row in beta_df.dropna(subset=["beta"]).iterrows():
            numbers.setdefault("beta", {})[f"b{row['beta']}"] = {
                "asr": float(row["asr"]),
                "ctca": float(row["ctca"]),
            }

    # ---------------- Efficiency ----------------
    all_traces = list(iter_traces(cfg.output_root))
    if all_traces:
        eff = efficiency.compute_all(all_traces)
        pd.DataFrame(
            [
                {"metric": "mean_latency_ms", "value": eff["mean_latency_ms"]},
                {"metric": "mean_tokens", "value": eff["mean_tokens"]},
                {"metric": "verifier_overhead", "value": eff["verifier_overhead"]},
            ]
        ).to_csv(paper_figs / "efficiency.csv", index=False)
        numbers["efficiency"] = {
            "mean_latency_ms": float(eff["mean_latency_ms"]),
            "mean_tokens": float(eff["mean_tokens"]),
            "verifier_overhead": float(eff["verifier_overhead"]),
        }
        # gate latency CSV for Fig 5
        from infoshield.analysis.latency_histogram import build as build_lat
        build_lat(cfg.output_root, experiment="main", out_csv=paper_figs / "fig5_gate_latency.csv")

    # ---------------- Multi-framework ----------------
    mf_df = _by_group(cfg.output_root, "multi_framework", ("system", "attack"))
    if not mf_df.empty:
        mf_df[["system", "attack", "asr", "ctca"]].to_csv(
            paper_figs / "multi_framework.csv", index=False
        )

    # ---------------- Adaptive attack ----------------
    adp_df = _by_group(cfg.output_root, "adaptive", ("system",))
    if not adp_df.empty:
        adp_df[["system", "asr", "ctca"]].to_csv(paper_figs / "adaptive.csv", index=False)
        for _, row in adp_df.iterrows():
            numbers.setdefault("adaptive", {})[row["system"]] = {
                "asr": float(row["asr"]),
                "ctca": float(row["ctca"]),
            }

    # ---------------- Encoder comparison ----------------
    enc_df = _by_group(cfg.output_root, "encoder_comparison", ("system", "attack"))
    if not enc_df.empty:
        enc_df.to_csv(paper_figs / "encoder_comparison.csv", index=False)
        for _, row in enc_df.iterrows():
            numbers.setdefault("encoder", {}).setdefault(row["system"], {})[
                row["attack"] + "_asr"
            ] = float(row["asr"])

    # ---------------- Failure gallery ----------------
    from infoshield.analysis.failure_cases import build as build_gallery
    build_gallery(cfg.output_root, experiment="main", k=10, out_dir=paper_figs)

    # ---------------- Emit numbers ----------------
    (paper / "numbers.json").write_text(json.dumps(numbers, indent=2), encoding="utf-8")

    # ---------------- Summary ----------------
    n_traces = sum(1 for _ in iter_traces(cfg.output_root))
    checks["n_traces"] = n_traces
    summary_path = paper / "aggregation_summary.json"
    summary_path.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(f"\naggregate_results: {n_traces} traces -> {paper}/  (exit={exit_code})")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
