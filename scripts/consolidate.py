"""Consolidator: after aggregate_results.py, copy paper artifacts into ONE folder.

Outputs into `paper/deliverable/` — a single self-contained folder with:
  - RESULTS.md            (auto-generated summary of every number in numbers.json)
  - numbers.json          (machine-readable copy of every headline number)
  - figures/*.pdf         (rendered matplotlib figures)
  - figures/*.csv         (raw data behind figures)
  - figures/failure_gallery.md   (correct blocks / false positives / misses table)
  - traces_index.json     (per-experiment counts, missing cells)
  - aggregation_summary.json     (coverage report)
  - README.md             (how to reproduce)

Usage:
    python scripts/consolidate.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from infoshield.config import load_config
from infoshield.loader import iter_traces


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = REPO_ROOT / "paper"
DELIVERABLE = PAPER_DIR / "deliverable"


def _run(cmd: list[str]) -> bool:
    print(f"$ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=REPO_ROOT)
    return r.returncode == 0


def _traces_index(output_root: str) -> dict:
    """Count JSONs per (experiment, system, attack)."""
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for t in iter_traces(output_root):
        counts.setdefault(t.experiment, {}).setdefault(t.system, {}).setdefault(
            t.attack, 0
        )
        counts[t.experiment][t.system][t.attack] += 1
    return counts


def _emit_results_md(numbers: dict, index: dict, out: Path) -> None:
    """Human-readable summary. Every number comes from numbers.json."""
    lines: list[str] = []
    lines.append("# INFOSHIELD — Experimental Results")
    lines.append("")
    lines.append("Auto-generated from `paper/numbers.json` by `scripts/consolidate.py`.")
    lines.append("Every number below is the mean over the samples in that cell.")
    lines.append("")

    # Coverage
    lines.append("## 1. Coverage")
    lines.append("")
    lines.append("| Experiment | System | Attack | # samples |")
    lines.append("| :--- | :--- | :--- | ---: |")
    for exp in sorted(index):
        for sys_name in sorted(index[exp]):
            for atk in sorted(index[exp][sys_name]):
                lines.append(f"| `{exp}` | `{sys_name}` | `{atk}` | {index[exp][sys_name][atk]} |")
    lines.append("")

    # Main table
    if "main" in numbers:
        lines.append("## 2. Main safety table (Attack Success Rate)")
        lines.append("")
        lines.append("Lower is better. ASR = fraction of runs whose final output contains the injection marker.")
        lines.append("")
        all_attacks = sorted({
            k.replace("_asr", "").replace("_ctca", "")
            for sys_data in numbers["main"].values()
            for k in sys_data
            if k.endswith("_asr")
        })
        header = "| System | " + " | ".join(f"`{a}`" for a in all_attacks) + " |"
        sep = "| :--- | " + " | ".join("---:" for _ in all_attacks) + " |"
        lines += [header, sep]
        for sys_name in sorted(numbers["main"]):
            row = [f"`{sys_name}`"]
            for a in all_attacks:
                v = numbers["main"][sys_name].get(f"{a}_asr")
                row.append(f"{v:.3f}" if v is not None else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # GSM8K fidelity
    if "gsm8k" in numbers:
        lines.append("## 3. Reasoning fidelity (GSM8K CTCA, no attack)")
        lines.append("")
        lines.append("| System | CTCA | Delta vs undefended |")
        lines.append("| :--- | ---: | ---: |")
        base = numbers["gsm8k"].get("undefended", {}).get("ctca")
        for sys_name in sorted(numbers["gsm8k"]):
            ctca = numbers["gsm8k"][sys_name].get("ctca", 0.0)
            delta = f"{(ctca - base) * 100:+.1f} pp" if base is not None else "—"
            lines.append(f"| `{sys_name}` | {ctca:.3f} | {delta} |")
        lines.append("")

    # Ablation
    if "ablation" in numbers:
        lines.append("## 4. Per-layer ablation")
        lines.append("")
        lines.append("| Config | ASR | CTCA |")
        lines.append("| :--- | ---: | ---: |")
        for name in sorted(numbers["ablation"]):
            asr = numbers["ablation"][name].get("asr", 0.0)
            ctca = numbers["ablation"][name].get("ctca", 0.0)
            lines.append(f"| `{name}` | {asr:.3f} | {ctca:.3f} |")
        lines.append("")

    # Beta
    if "beta" in numbers:
        lines.append("## 5. Beta sweep (Pareto: safety vs utility)")
        lines.append("")
        lines.append("| Beta | ASR | CTCA |")
        lines.append("| ---: | ---: | ---: |")
        for k in sorted(numbers["beta"]):
            row = numbers["beta"][k]
            lines.append(f"| {k.lstrip('b')} | {row.get('asr', 0.0):.3f} | {row.get('ctca', 0.0):.3f} |")
        lines.append("")

    # Adaptive
    if "adaptive" in numbers:
        lines.append("## 6. Adaptive paraphrase attack")
        lines.append("")
        lines.append("| System | ASR | CTCA |")
        lines.append("| :--- | ---: | ---: |")
        for sys_name in sorted(numbers["adaptive"]):
            row = numbers["adaptive"][sys_name]
            lines.append(f"| `{sys_name}` | {row.get('asr', 0.0):.3f} | {row.get('ctca', 0.0):.3f} |")
        lines.append("")

    # Encoder comparison
    if "encoder" in numbers:
        lines.append("## 7. Encoder comparison (local MiniLM vs Azure text-embedding-3-large)")
        lines.append("")
        for sys_name in sorted(numbers["encoder"]):
            lines.append(f"### `{sys_name}`")
            for k, v in sorted(numbers["encoder"][sys_name].items()):
                lines.append(f"- `{k}`: {v:.3f}")
        lines.append("")

    # Efficiency
    if "efficiency" in numbers:
        e = numbers["efficiency"]
        lines.append("## 8. Efficiency")
        lines.append("")
        lines.append(f"- Mean latency per run: **{e.get('mean_latency_ms', 0):.0f} ms**")
        lines.append(f"- Mean tokens per run: **{e.get('mean_tokens', 0):.0f}**")
        lines.append(f"- Verifier overhead (fraction of tokens spent on 5-sample MC): **{e.get('verifier_overhead', 0):.1%}**")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Files in this folder")
    lines.append("")
    lines.append("- `RESULTS.md` — this file")
    lines.append("- `numbers.json` — machine-readable copy of every number above")
    lines.append("- `figures/*.pdf` — rendered figures (bar chart, Pareto curve, ablation heatmap, latency histogram)")
    lines.append("- `figures/*.csv` — raw data behind each figure")
    lines.append("- `figures/failure_gallery.md` — correct blocks / false positives / misses gallery")
    lines.append("- `traces_index.json` — coverage report (which cells have how many samples)")
    lines.append("- `aggregation_summary.json` — per-cell coverage checks")
    lines.append("- `README.md` — how to reproduce and scale up")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _emit_deliverable_readme(out: Path) -> None:
    out.write_text(
        (
            "# INFOSHIELD deliverable folder\n\n"
            "This folder is entirely auto-generated. To regenerate after new experiments:\n\n"
            "```bash\n"
            "python scripts/run_final.py            # runs experiments + writes FINAL.md\n"
            "# or, to just refresh reports from existing traces:\n"
            "python scripts/run_final.py --report-only\n"
            "```\n\n"
            "Nothing here is hand-written. See `FINAL.md` for the comprehensive "
            "research artifact (design, methodology, results, discussion) and "
            "`RESULTS.md` for a shorter auto-generated summary.\n"
        ),
        encoding="utf-8",
    )


def main():
    cfg = load_config()

    # 1. Run the aggregator (produces paper/numbers.json, figures/*.csv, ...)
    if not _run([sys.executable, "scripts/aggregate_results.py"]):
        print("[warn] aggregator returned nonzero (missing cells are OK)")

    # 2. Render every figure from the CSVs it just wrote.
    figs = [
        "figures/fig1_asr_bar.py",
        "figures/fig2_beta_pareto.py",
        "figures/fig3_ablation_heatmap.py",
        "figures/fig5_gate_latency.py",
        "figures/fig6_failure_gallery.py",
    ]
    for f in figs:
        script = REPO_ROOT / f
        if not script.exists():
            print(f"[skip] {f} not found")
            continue
        # Some figures require CSVs that may not exist yet — tolerate failures.
        _run([sys.executable, f])

    # 3. Build the consolidated folder.
    # Preserve run-orchestrator artifacts (progress.jsonl, step_*.md, FINAL.md)
    # AND the reports written by specialized runners that consolidate doesn't
    # regenerate itself (judges_report.json from run_judges.py, sentinel_check
    # + provider_drift reports from their runners) so those aren't destroyed.
    _PRESERVE_PATTERNS = (
        "progress.jsonl", "step_*.md", "FINAL.md",
        "judges_report.json",
        "sentinel_check_report.json",
        "provider_drift_report.json",
    )
    # Figures written by analyses that consolidate doesn't drive itself
    # (analyze_gates, analyze_certified_radius, analyze_argmax_phrases,
    # analyze_gate_pareto, fig7_sankey_attribution, fig8_coverage_matrix).
    # Their CSVs/PDFs live in DELIVERABLE/figures/ but NOT in PAPER_DIR/figures/,
    # so a naive wipe-and-copy would lose them. Preserve by name pattern.
    _PRESERVE_FIGURE_PATTERNS = (
        "roc_per_gate.*", "threshold_sensitivity.*",
        "certified_radius_*.*", "certified_radius_per_gate.*",
        "argmax_phrase_*.*",
        "gate_pareto.*", "gate_pareto_summary.json",
        "fig7_sankey.*", "fig8_coverage_matrix.*",
    )
    preserved: dict[str, bytes] = {}
    preserved_figs: dict[str, bytes] = {}
    if DELIVERABLE.exists():
        for pat in _PRESERVE_PATTERNS:
            for f in DELIVERABLE.glob(pat):
                if f.is_file():
                    preserved[f.name] = f.read_bytes()
        figs_deliv = DELIVERABLE / "figures"
        if figs_deliv.exists():
            for pat in _PRESERVE_FIGURE_PATTERNS:
                for f in figs_deliv.glob(pat):
                    if f.is_file():
                        preserved_figs[f.name] = f.read_bytes()
        shutil.rmtree(DELIVERABLE)
    DELIVERABLE.mkdir(parents=True)
    (DELIVERABLE / "figures").mkdir()
    for name, data in preserved.items():
        (DELIVERABLE / name).write_bytes(data)
    for name, data in preserved_figs.items():
        (DELIVERABLE / "figures" / name).write_bytes(data)

    # numbers + coverage
    for f in ("numbers.json", "aggregation_summary.json"):
        src = PAPER_DIR / f
        if src.exists():
            shutil.copy2(src, DELIVERABLE / f)

    # figures (PDFs + CSVs + failure_gallery.{json,md})
    figs_dir = PAPER_DIR / "figures"
    if figs_dir.exists():
        for f in figs_dir.glob("*"):
            if f.suffix in {".pdf", ".csv", ".json", ".md"}:
                shutil.copy2(f, DELIVERABLE / "figures" / f.name)

    # traces index
    index = _traces_index(cfg.output_root)
    (DELIVERABLE / "traces_index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )

    # numbers.json for the RESULTS.md summary
    numbers_path = DELIVERABLE / "numbers.json"
    numbers = json.loads(numbers_path.read_text(encoding="utf-8")) if numbers_path.exists() else {}

    _emit_results_md(numbers, index, DELIVERABLE / "RESULTS.md")
    _emit_deliverable_readme(DELIVERABLE / "README.md")

    print(f"\nConsolidated deliverable at: {DELIVERABLE}")
    print("Files:")
    for p in sorted(DELIVERABLE.rglob("*")):
        if p.is_file():
            rel = p.relative_to(DELIVERABLE)
            print(f"  {rel}   ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
