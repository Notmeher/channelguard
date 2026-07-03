"""Fig 6 — failure gallery. Not a matplotlib figure — emits a LaTeX table
with three column groups (correct blocks / false positives / misses).
Reads `paper/figures/failure_gallery.json` produced by
`infoshield.analysis.failure_cases.build`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _row(entry: dict) -> str:
    ipt = entry["raw_input"].replace("\n", " ").replace("&", "\\&")[:60]
    out = (entry["final_output"] or "").replace("\n", " ").replace("&", "\\&")[:60]
    return f"\\texttt{{{ipt}}} & \\texttt{{{out}}} \\\\"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="paper/figures/failure_gallery.json")
    ap.add_argument("--out", default="paper/tables/failure_gallery.tex")
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    lines = ["\\begin{tabular}{p{0.45\\linewidth} p{0.45\\linewidth}}", "\\toprule"]

    for section, header in [
        ("correct_blocks", "\\textbf{Correct blocks}"),
        ("false_positives", "\\textbf{False positives}"),
        ("misses", "\\textbf{Misses}"),
    ]:
        lines.append(f"\\multicolumn{{2}}{{c}}{{{header}}} \\\\ \\midrule")
        lines.append("\\textbf{Input} & \\textbf{Final output} \\\\ \\midrule")
        for e in data.get(section, [])[:10]:
            lines.append(_row(e))
        lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
