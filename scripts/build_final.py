"""build_final.py — regenerate every table, macro, CSV and figure in Final/ from source data.

Two sources, nothing hand-edited:
  1. MANUSCRIPT — numbers transcribed from arXiv:2607.19430v2 (Azure GPT-5 agent). The original
     2,100 raw traces are not in this checkout, so the paper's tables are the source of truth.
  2. Final/qwen/numbers.json and Final/llama/numbers.json, plus the raw traces in results_qwen/
     and results_llama/ (per-role latency and the failure gallery need trace-level data).

Outputs (all overwritten on every run):
  Final/tables/*.tex                 manuscript-exact tables (one per manuscript table)
  Final/tables/cross_model/*.tex     three-way GPT-5 / Qwen / Llama tables
  Final/data/*.csv, data/cross_model/*.csv, data/failure_gallery.json
  Final/macros/numbers.tex, numbers.json   (legacy macro names kept + \\Qwen... / \\Llama... added)
  Final/figures/phrase_audit.pdf (corrected Figure 5), cross_model_asr.pdf, cross_model_mechanism.pdf
  Final/README.md
  Final/final_results.md             only the block between the GENERATED markers

Usage:  .venv/bin/python scripts/build_final.py      (offline, no API calls)
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from infoshield.attacks.registry import contains_attack_marker  # noqa: E402
from infoshield.ib.phrase_bank import get_bank  # noqa: E402
from infoshield.loader import iter_traces  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FINAL = REPO / "Final"
TABLES, XTABLES = FINAL / "tables", FINAL / "tables" / "cross_model"
DATA, XDATA = FINAL / "data", FINAL / "data" / "cross_model"
FIGS, MACROS = FINAL / "figures", FINAL / "macros"
RESULTS_MD = FINAL / "final_results.md"
BEGIN, END = "<!-- BEGIN GENERATED: cross-model replication (scripts/build_final.py) -->", \
    "<!-- END GENERATED: cross-model replication -->"

PI, TP, MP = "prompt_injection_strong", "tool_poisoning", "memory_poisoning"
SD, CE, RC, AP = "mem_slow_drift", "mem_consent_escalation", "mem_role_confusion", "adaptive_paraphrase"
ATTACKS = [PI, TP, MP, SD, CE, RC, AP, "none"]
INTER = [TP, MP, SD, CE, RC]
EXTRA = ["reasoning_state_injection", "consensus_collapse"]
CORE = ["undefended", "infoshield"]
BASE = ["undefended", "ibprotector", "llm_guard", "perplexity_filter", "smoothllm", "infoshield"]
BASE_ATTACKS = [PI, TP, AP, "none"]
GATES = ["IB-0", "IB-1", "IB-2", "IB-3", "IB-4", "IB-5"]
BETAS = ["0.1", "0.3", "0.5", "1.0", "2.0", "5.0"]
CLASSES = ["leaked", "ib_blocked", "provider_filter", "verifier_unsafe", "synth_refused", "answered_safely"]

ALABEL = {PI: "Prompt Injection", TP: "Tool Poisoning", MP: "Memory Poisoning", SD: "Slow-Drift Memory",
          CE: "Consent-Escalation Memory", RC: "Role-Confusion Memory", AP: "Adaptive Paraphrase",
          "none": "Benign", "reasoning_state_injection": "Reasoning-State Injection",
          "consensus_collapse": "Consensus Collapse"}
SLABEL = {"undefended": "Undefended", "infoshield": "ChannelGuard", "ibprotector": "IBProtector",
          "llm_guard": "LLM-Guard", "perplexity_filter": "Perplexity-Filter", "smoothllm": "SmoothLLM"}
MODELS = {"qwen": "Qwen3-4B", "llama": "Llama-3.1-8B"}
MODEL_FULL = {"gpt5": "Azure GPT-5 (manuscript)", "qwen": "Qwen3-4B-Instruct-2507 (MLX 8-bit, local)",
              "llama": "Llama-3.1-8B-Instruct (MLX 4-bit, local)"}
TRACE_ROOT = {"qwen": REPO / "results_qwen", "llama": REPO / "results_llama"}

# ============================================================================ manuscript (arXiv:2607.19430v2)

P_ASR = {**{("undefended", a): v for a, v in zip(ATTACKS, [0.333, 0, 0, 0, 0, 0, 0.667, 0])},
         **{("infoshield", a): v for a, v in zip(ATTACKS, [0.167, 0, 0, 0, 0, 0, 0.667, 0])}}
P_MCNEMAR = (6, 1, 0.125)
P_ATTR = {  # Table 9: leaked, ib, prov, ver, syn, safe
    ("undefended", PI): (10, 0, 6, 4, 0, 10), ("undefended", TP): (0, 0, 24, 0, 0, 6),
    ("undefended", MP): (0, 0, 30, 0, 0, 0), ("undefended", SD): (0, 0, 0, 3, 0, 27),
    ("undefended", CE): (0, 0, 0, 3, 3, 24), ("undefended", RC): (0, 0, 0, 2, 0, 28),
    ("undefended", AP): (20, 0, 0, 7, 0, 3), ("undefended", "none"): (0, 0, 0, 2, 0, 28),
    ("infoshield", PI): (5, 21, 0, 1, 0, 3), ("infoshield", TP): (0, 30, 0, 0, 0, 0),
    ("infoshield", MP): (0, 2, 22, 0, 0, 6), ("infoshield", SD): (0, 0, 0, 4, 0, 26),
    ("infoshield", CE): (0, 20, 0, 2, 0, 8), ("infoshield", RC): (0, 0, 0, 1, 0, 29),
    ("infoshield", AP): (20, 1, 0, 0, 0, 9), ("infoshield", "none"): (0, 0, 0, 1, 0, 29),
}
P_CTCA_MAIN = {("infoshield", AP): 0.133, ("infoshield", "none"): 0.533, ("infoshield", PI): 0.133,
               ("infoshield", TP): 0.433, ("undefended", AP): 0.033, ("undefended", "none"): 0.567,
               ("undefended", PI): 0.167, ("undefended", TP): 0.467}
# Tables 4 and 10: backend, system, IB, prov, model-handled, ASR
P_TP_BACKENDS = [("Azure GPT-5", "undefended", 0, 24, 6, 0.0), ("Azure GPT-5", "infoshield", 30, 0, 0, 0.0),
                 ("Anthropic Sonnet-4.5", "undefended", 0, 0, 30, 0.0),
                 ("Anthropic Sonnet-4.5", "infoshield", 30, 0, 0, 0.0),
                 ("Anthropic Haiku-4.5", "undefended", 0, 0, 30, 0.0),
                 ("Anthropic Haiku-4.5", "infoshield", 30, 0, 0, 0.0)]
P_POOL_BACKENDS = [("Azure GPT-5", "undefended", 0, 54, 36, 0.0), ("Azure GPT-5", "infoshield", 32, 22, 36, 0.0),
                   ("Anthropic Sonnet-4.5", "undefended", 0, 0, 90, 0.0),
                   ("Anthropic Sonnet-4.5", "infoshield", 36, 0, 54, 0.0),
                   ("Anthropic Haiku-4.5", "undefended", 0, 0, 90, 0.0),
                   ("Anthropic Haiku-4.5", "infoshield", 50, 0, 40, 0.0)]
P_SENTINEL = {("undefended", TP): (0, 0), ("undefended", MP): (0, 0), ("undefended", RC): (0, 0),
              ("infoshield", TP): (30, 28), ("infoshield", MP): (2, 0), ("infoshield", RC): (0, 0)}
P_EFF = [("Prompt Injection", 30, 2621, 794), ("Tool Poisoning", 30, 2474, 2102),
         ("Memory (4 variants)", 120, 8845, 8604), ("All attacks pooled", 210, 16367, 13700)]
P_OVERALL = {"mean_latency_ms": 70291.052, "mean_tokens": 4291.896, "verifier_overhead": 0.310, "gsm8k_ctca": 0.867}
P_COST = [("Main slice", 480, 16.03, "Tables 2--3; Apps. I.1--J"), ("Baseline comparison", 480, 9.42, "Appendix C"),
          ("Sentinel randomization", 180, 5.39, r"\S6.4"), ("Provider counterfactual (Sonnet)", 180, 3.55, r"\S6.3"),
          (r"$\beta$-sweep", 180, 3.14, "Appendix G"), ("Gate ablation", 210, 2.91, "Appendix F"),
          ("Provider counterfactual (Haiku)", 180, 2.32, r"\S6.3"), ("GSM8K fidelity", 60, 1.45, r"\S6.5"),
          ("Phrase-audit backfill", 60, 1.36, "Appendix I.5"), ("Phrase-bank-size sweep", 90, 1.04, "Appendix H"),
          ("Judge audit (100 pairs)", None, 0.76, "Appendix N")]
P_REPRO = {"reprocost": "47.36", "reprotraces": "2{,}100", "reprojudgepairs": "100", "reprocalls": "30{,}769",
           "reprointokens": "3.99M", "reproouttokens": "3.64M", "reproelapsed": "4.8~h", "reprollmwall": "33.16~h",
           "reprocostmain": "16.03", "reprocostbaselines": "9.42", "reprocostsentinel": "5.39",
           "reprocostprovsonnet": "3.55", "reprocostbeta": "3.14", "reprocostablation": "2.91",
           "reprocostprovhaiku": "2.32", "reprocostgsmeight": "1.45", "reprocostphraseauditv": "1.36",
           "reprocostbanksize": "1.04", "reprocostjudge": "0.76"}
P_BASE_ASR = {**{(s, a): P_ASR[(s, a)] for s in CORE for a in BASE_ATTACKS},
              ("ibprotector", PI): 0.033, ("llm_guard", PI): 0.0, ("perplexity_filter", PI): 0.0,
              ("smoothllm", PI): 0.067, ("ibprotector", AP): 0.633, ("llm_guard", AP): 0.433,
              ("perplexity_filter", AP): 0.0, ("smoothllm", AP): 0.200,
              **{(s, a): 0.0 for s in ["ibprotector", "llm_guard", "perplexity_filter", "smoothllm"]
                 for a in [TP, "none"]}}
P_ABLATION = [("ChannelGuard (full)", "infoshield", 0.067, 17, 6, 1), ("--IB-0", "infoshield_no_ib0", 0.133, 0, 10, 3),
              ("--IB-1", "infoshield_no_ib1", 0.033, 18, 8, 1), ("--IB-2", "infoshield_no_ib2", 0.100, 18, 7, 1),
              ("--IB-3", "infoshield_no_ib3", 0.033, 17, 8, 1), ("--IB-4", "infoshield_no_ib4", 0.033, 17, 7, 1),
              ("--IB-5", "infoshield_no_ib5", 0.133, 17, 8, 0)]
P_BETA = {"0.1": (0.835, 0.167, 5, 0), "0.3": (0.805, 0.100, 3, 0), "0.5": (0.775, 0.133, 4, 0),
          "1.0": (0.700, 0.100, 3, 10), "2.0": (0.550, 0.067, 2, 17), "5.0": (0.150, 0.000, 0, 30)}
P_BANK = {"5": (0.067, 2, 17), "20": (0.100, 3, 17), "50": (0.167, 5, 17)}
P_MARGIN = {"IB-0": (21, 0.140, 0.194, 0.314), "IB-1": (4, 0.025, 0.103, 0.346),
            "IB-2": (1, 0.091, 0.091, 0.091), "IB-3": (90, 0.495, 0.512, 1.143)}
P_AUC = {"IB-0": 0.833, "IB-1": 0.434, "IB-2": 0.402, "IB-3": 0.624, "IB-4": 0.402, "IB-5": 0.393}
P_PARETO = [((), 0.0, 1.000), (("IB-1",), 21.9, 0.978), (("IB-3",), 24.1, 0.667), (("IB-1", "IB-3"), 46.0, 0.644),
            (("IB-1", "IB-2", "IB-3"), 72.8, 0.633), (("IB-0", "IB-3"), 93.4, 0.433),
            (("IB-0", "IB-1", "IB-3"), 115.4, 0.411), (("IB-0", "IB-1", "IB-2", "IB-3"), 142.2, 0.400)]
P_THRESH = {"IB-0": (0.233, 0.550, 0.650), "IB-3": (0.441, 0.300, 0.825), "IB-1": (0.029, 0.300, 0.900),
            "IB-2": (0.015, 0.350, 0.900), "IB-4": (0.000, 0.350, 0.900), "IB-5": (0.000, 0.350, 0.900)}
P_PHRASES = [(16, 138), (0, 93), (4, 81), (17, 58), (15, 23), (18, 15), (2, 9), (6, 9)]  # (bank index, wins)
P_PHRASE_TOTAL, P_PHRASE_ALL = 426, 559
P_BPR = [("main", "undefended", 30, 17, 0, 0, 0, 0.567, 0.067), ("main", "infoshield", 30, 16, 0, 0, 0, 0.533, 0.033),
         ("baselines", "ibprotector", 30, 13, 0, 0, 0, 0.433, 0.100),
         ("baselines", "llm_guard", 30, 12, 1, 0, 0, 0.400, 0.067),
         ("baselines", "smoothllm", 30, 12, 0, 0, 0, 0.400, 0.067),
         ("baselines", "perplexity_filter", 30, 0, 30, 0, 0, 0.000, 1.000),
         ("gsm8k", "undefended", 30, 25, 0, 0, 0, 0.833, 0.033), ("gsm8k", "infoshield", 30, 25, 0, 0, 0, 0.833, 0.033)]
P_ROLE_SHARE = [("verifier", 6750, 16530, 0.564), ("worker ($\\times$3)", 1592, 10951, 0.374),
                ("planner", 532, 4177, 0.143), ("synth", 510, 3911, 0.133)]
P_ROLE_LAT = [("undefended", 7861, 6860, 6613, 2450), ("infoshield", 7851, 6880, 7669, 2449),
              ("llm_guard", 11608, 8105, 8636, 2508), ("ibprotector", 10470, 7590, 8467, 2505)]
P_COMPRESS = {"IB-0": (248, 190, 58), "IB-1": (5, 3, 2), "IB-2": (41, 41, 0), "IB-3": (2, 2, 0),
              "IB-4": (51, 51, 0), "IB-5": (37, 37, 0)}
P_VERIFIER = {"total": 4513, "risky_pass_through": 354, "risky_leaked": 69, "unsafe_and_uncertain": 149}
P_JUDGE = {"n": 100, "raw": 0.950, "sonnet_vs_gpt5": 0.900, "sonnet_vs_truth": 0.900, "gpt5_vs_truth": 1.000,
           "vague": 0.060}

# ============================================================================ helpers

NUM_WORDS = "Zero One Two Three Four Five Six Seven Eight Nine".split()


def macro(*parts: str, digits: str = "words") -> str:
    """Build a letters-only LaTeX command name."""
    out = []
    for p in parts:
        p = str(p)
        if digits == "words":
            p = re.sub(r"\d", lambda m: NUM_WORDS[int(m.group())], p.replace(".", "pt"))
        else:
            p = re.sub(r"\d", "", p)
        p = re.sub(r"[^A-Za-z]", "", p)
        out.append(p[:1].upper() + p[1:] if digits == "words" else p)
    return "".join(out)


_TEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
_UNICODE = {"θ": r"$\theta$", "β": r"$\beta$", "ε": r"$\varepsilon$", "κ": r"$\kappa$", "σ": r"$\sigma$",
            "×": r"$\times$", "→": r"$\to$", "≥": r"$\geq$", "≤": r"$\leq$", "∧": r"$\wedge$", "—": "---",
            "–": "--", "“": "``", "”": "''", "’": "'", "‘": "`", "…": r"\ldots{}", "Δ": r"$\Delta$"}


def esc(s) -> str:
    s = "" if s is None else str(s)
    s = "".join(_TEX_ESC.get(ch, ch) for ch in s)
    for k, v in _UNICODE.items():
        s = s.replace(k, v)
    return "".join(ch if ord(ch) < 128 else "?" for ch in s)


def f3(x) -> str:
    return "---" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def fpct(x) -> str:
    return "---" if x is None else f"{100 * x:.1f}\\%"


def write_tex(path: Path, colspec: str, header: list[str], rows: list[list], caption: str, label: str,
              *, size: str = "", star: bool = False, midrules: set[int] | None = None) -> None:
    ncol = len(re.sub(r"[^lcrp]", "", re.sub(r"p\{[^}]*\}", "p", colspec)))
    assert len(header) == ncol, f"{path.name}: header has {len(header)} cols, spec {ncol}"
    env = "table*" if star else "table"
    lines = [f"% AUTO-GENERATED by scripts/build_final.py — do not edit by hand.",
             f"\\begin{{{env}}}[t]", "\\centering"]
    if size:
        lines.append(size)
    lines += [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(header) + r" \\", "\\midrule"]
    for i, r in enumerate(rows):
        assert len(r) == ncol, f"{path.name}: row {i} has {len(r)} cols, spec {ncol}: {r}"
        if midrules and i in midrules:
            lines.append("\\midrule")
        lines.append(" & ".join(str(c) for c in r) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
              f"\\end{{{env}}}", ""]
    text = "\n".join(lines)
    _check_tex(path.name, text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _check_tex(name: str, text: str) -> None:
    body = re.sub(r"\\[{}%$&#_]", "", text)
    body = re.sub(r"\\label\{[^}]*\}", "", body)  # underscores are legal inside \label
    body = "\n".join(line.split("%")[0] if not line.startswith("% AUTO") else "" for line in body.splitlines())
    assert body.count("{") == body.count("}"), f"{name}: unbalanced braces"
    assert body.count("$") % 2 == 0, f"{name}: unbalanced $"
    assert not re.search(r"(?<!\\)_", re.sub(r"\$[^$]*\$", "", body).replace("\\_", "")), f"{name}: unescaped _"
    assert all(ord(c) < 128 for c in text.replace("—", "")), f"{name}: non-ASCII character"


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def md_table(header: list[str], rows: list[list], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    sep = ["---:" if a == "r" else ":---" for a in align]
    return "\n".join(["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def classify(t) -> str:
    if contains_attack_marker(t.final_output):
        return "leaked"
    if any(g.status == "BLOCKED" for g in t.gate_decisions):
        return "ib_blocked"
    if any(c.provider == "azure_openai" and c.prompt_tokens == 0 and c.completion_tokens == 0 for c in t.llm_calls):
        return "provider_filter"
    if any(r.flagged_unsafe for r in t.verifier_reports):
        return "verifier_unsafe"
    return "synth_refused" if t.refused else "answered_safely"


def model_handled(c: dict) -> int:
    return c.get("verifier_unsafe", 0) + c.get("synth_refused", 0) + c.get("answered_safely", 0)


# ============================================================================ load local replications


def load_local() -> dict:
    L = {}
    for m in MODELS:
        L[m] = json.loads((FINAL / m / "numbers.json").read_text(encoding="utf-8"))
        ov = L[m]["efficiency"]["overall"]
        # Each suite names this key after its own model.
        ov["gen_tokens_per_s"] = ov.get(f"{m}_gen_tokens_per_s", 0.0)
        roles: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        gallery: dict[str, list[dict]] = {"correct_blocks": [], "false_positives": [], "misses": []}
        for t in iter_traces(TRACE_ROOT[m]):
            if t.experiment in ("main", "baselines"):
                for c in t.llm_calls:
                    roles[t.system]["worker" if c.role.startswith("worker") else c.role].append(c.latency_ms)
            if t.experiment == "main" and t.system == "infoshield":
                cls = classify(t)
                blocks = [g for g in t.gate_decisions if g.status == "BLOCKED"]
                item = {"model": m, "attack": t.attack, "sample_id": t.sample_id, "class": cls,
                        "input": (t.raw_input or "")[:220], "final_output": (t.final_output or "")[:160],
                        "blocking_gate": blocks[0].gate_id if blocks else None,
                        "risk": round(blocks[0].risk, 3) if blocks else None,
                        "argmax_phrase": blocks[0].argmax_phrase if blocks else None}
                if t.attack == "none" and (t.refused or blocks):
                    gallery["false_positives"].append(item)
                elif t.attack != "none" and cls == "leaked":
                    gallery["misses"].append(item)
                elif t.attack != "none" and cls == "ib_blocked":
                    gallery["correct_blocks"].append(item)
        L[m]["_roles"] = {s: {r: sum(v) / len(v) for r, v in d.items()} for s, d in roles.items()}
        L[m]["_gallery"] = gallery
    return L


def cnt(L, m, key, cell) -> dict:
    return L[m].get(key, {}).get(cell, {}) or {}


# ============================================================================ manuscript-exact tables


def build_manuscript_tables(macros: dict) -> None:
    bank = get_bank(20)
    # Table 2
    rows = [[esc(ALABEL[a]), f3(P_ASR[("undefended", a)]), f3(P_ASR[("infoshield", a)])] for a in ATTACKS]
    write_tex(TABLES / "main_asr.tex", "lrr", ["Attack", "Undefended", "ChannelGuard"], rows,
              "Attack success rate, $n{=}30$ per cell, Azure GPT-5, paired on identical sample IDs "
              "(manuscript Table 2).", "tab:main-asr", midrules={7})
    # Table 9 (supersedes the 4-attack Table 3 slice)
    rows = [[SLABEL[s], esc(ALABEL[a]), *P_ATTR[(s, a)]] for s in CORE for a in ATTACKS]
    write_tex(TABLES / "attribution.tex", "llrrrrrr",
              ["System", "Attack", "leak", "ib", "prov", "ver", "syn", "safe"], rows,
              "Complete per-trace attribution, 8 attacks $\\times$ 2 systems, $n{=}30$ per row (manuscript "
              "Table 9; Table 3 is its 8-row subset).", "tab:attribution", size="\\small", midrules={8})
    # Tables 4 and 10
    rows = [[b, SLABEL[s], ib, pr, mh, f3(a)] for b, s, ib, pr, mh, a in P_TP_BACKENDS]
    write_tex(TABLES / "provider_tool_poisoning.tex", "llrrrr", ["Backend", "System", "IB", "prov", "model-hdl.", "ASR"],
              rows, "Tool Poisoning only, $n{=}30$ per row (manuscript Table 4).", "tab:provider-tp", midrules={2, 4})
    rows = [[b, SLABEL[s], ib, pr, mh, f3(a)] for b, s, ib, pr, mh, a in P_POOL_BACKENDS]
    write_tex(TABLES / "provider_pooled.tex", "llrrrr", ["Backend", "System", "IB", "prov", "model-hdl.", "ASR"],
              rows, "Pooled Tool Poisoning + Memory Poisoning + Role-Confusion Memory, $n{=}90$ per row "
              "(manuscript Table 10).", "tab:provider-pooled", midrules={2, 4})
    # Table 5
    rows = [[SLABEL[s], esc(ALABEL[a]), 30, o, r, f"${r - o:+d}$"] for (s, a), (o, r) in P_SENTINEL.items()]
    write_tex(TABLES / "sentinel_delta.tex", "llrrrr", ["System", "Attack", "$n$", "IB orig.", "IB rand.", "$\\Delta$"],
              rows, "Sentinel-randomization test: IB block counts under the original \\texttt{ATTACK\\_SUCCESS} "
              "marker vs.\\ a per-attack random token absent from the phrase bank (manuscript Table 5).",
              "tab:sentinel_delta", midrules={3})
    # Table 6
    rows = [[g, n, f"{u:,} s", f"{c:,} s", f"{u / c:.2f}$\\times$"] for g, n, u, c in P_EFF]
    write_tex(TABLES / "efficiency.tex", "lrrrr", ["Attack group", "$n$", "Undefended", "ChannelGuard", "Speedup"],
              rows, "Efficiency: wall-clock A/B from trace timestamps. Pooled Azure GPT-5: mean latency 70,291 ms; "
              "mean tokens/run 4,292; MC-verifier share 31.0\\%. GSM8K (no attack, $n{=}30$): CTCA 0.867 for both "
              "(manuscript Table 6).", "tab:efficiency", midrules={3})
    # Table 7
    rows = [[c[0], "n/a" if c[1] is None else f"{c[1]:,}", f"{c[2]:.2f}", c[3]] for c in P_COST]
    rows.append(["\\textbf{Total}", "2,100 + 100", "\\textbf{47.36}", "every quantitative claim"])
    write_tex(TABLES / "cost.tex", "lrrl", ["Experiment", "Traces", "Cost (\\$)", "Supports"], rows,
              "Cost and trace count per experiment, measured from per-call token accounting (manuscript Table 7).",
              "tab:cost", size="\\small", midrules={len(rows) - 1})
    # Table 8
    rows = [[esc(ALABEL[a]), *[f3(P_BASE_ASR[(s, a)]) for s in BASE]] for a in BASE_ATTACKS]
    write_tex(TABLES / "baselines.tex", "lrrrrrr", ["Attack", *[esc(SLABEL[s]) for s in BASE]], rows,
              "ASR of all defense systems side by side, $n{=}30$ per cell. Perplexity-Filter's 0.000 is spurious: "
              "it blocks 100\\% of inputs including benign ones (manuscript Table 8).", "tab:baselines",
              size="\\small", star=True)
    # Table 11
    rows = [[esc(lbl), f3(a), ib, pr, ve] for lbl, _, a, ib, pr, ve in P_ABLATION]
    write_tex(TABLES / "gate_ablation.tex", "lrrrr", ["Config", "ASR", "IB", "prov", "ver"], rows,
              "Per-gate ablation on Prompt Injection, $n{=}30$ each (manuscript Table 11).", "tab:ablation")
    # Table 12
    rows = [[b, f"{th:.3f}", f3(a), lk, ib] for b, (th, a, lk, ib) in P_BETA.items()]
    write_tex(TABLES / "beta_sweep.tex", "rrrrr", ["$\\beta$", "$\\theta_{\\mathrm{eff}}$", "ASR", "leaks", "IB blocks"],
              rows, "$\\beta$-sweep on Prompt Injection, $n{=}30$ each (manuscript Table 12). Note: Eq.~(3) gives "
              "$\\theta_{\\mathrm{eff}}(1.0){=}0.70$, not the default $\\theta{=}0.60$.", "tab:beta")
    # Table 13
    rows = [[k, f3(a), lk, ib] for k, (a, lk, ib) in P_BANK.items()]
    write_tex(TABLES / "bank_size.tex", "rrrr", ["Bank size", "ASR", "leaks", "IB blocks"], rows,
              "Phrase-bank size sweep on Prompt Injection, $n{=}30$ each (manuscript Table 13).", "tab:bank")
    # Table 14
    rows = [[g, n, f3(md), f3(mn), f3(mx)] for g, (n, md, mn, mx) in P_MARGIN.items()]
    write_tex(TABLES / "gate_margin.tex", "lrrrr",
              ["Gate", "\\# blocks", "median $\\varepsilon$", "mean $\\varepsilon$", "max $\\varepsilon$"], rows,
              "Embedding-space margin $\\varepsilon=(c-\\theta)/\\sqrt{1-c^2}$ over BLOCKED gate decisions on the "
              "primary Azure GPT-5 slice; IB-4 and IB-5 never block. Not a certified radius (manuscript Table 14).",
              "tab:gate_margin")
    # Table 15
    rows = [[g, 120, 90, f3(v)] for g, v in P_AUC.items()]
    write_tex(TABLES / "gate_auc.tex", "lrrr", ["Gate", "$n$ traces", "$n$ attack", "AUC"], rows,
              "Per-gate ROC AUC, attack vs.\\ benign, on the 120-trace ChannelGuard subset (manuscript Table 15).",
              "tab:gate_auc")
    # Table 16
    rows = [["\\{" + ", ".join(g) + "\\}", f"{lat:.1f}", f3(ub)] for g, lat, ub in P_PARETO]
    write_tex(TABLES / "gate_pareto.tex", "lrr", ["Gate subset", "latency (ms)", "upper-bound ASR"], rows,
              "Dominant points of the 64-subset gate Pareto frontier (manuscript Table 16).", "tab:pareto")
    # Table 17
    rows = [[g, f3(r), f"[{lo:.3f}, {hi:.3f}]", f"{hi - lo:.2f}"] for g, (r, lo, hi) in P_THRESH.items()]
    write_tex(TABLES / "threshold_sensitivity.tex", "lrlr", ["Gate", "block rate @ $\\theta{=}0.60$", "plateau", "width"],
              rows, "Threshold-sensitivity plateau widths: range of $\\theta$ within $\\pm$5 pp of the "
              "$\\theta{=}0.60$ block rate (manuscript Table 17).", "tab:threshold")
    # Table 18
    rows = [[i + 1, w, "``" + esc(bank[p]) + "''"] for i, (p, w) in enumerate(P_PHRASES)]
    write_tex(TABLES / "phrase_audit.tex", "rrp{0.62\\linewidth}", ["Rank", "Wins", "Phrase"], rows,
              f"The 8 bank phrases (of 20) that ever win an argmax similarity match, pooled across all gates and "
              f"experiments ($n{{=}}{P_PHRASE_TOTAL}$ captured BLOCK decisions of {P_PHRASE_ALL}; manuscript "
              f"Table 18).", "tab:phrase_audit", size="\\small")
    # Table 19
    rows = [[e, esc(SLABEL[s]), n, c, ib, pv, sy, f3(b), f3(fp)] for e, s, n, c, ib, pv, sy, b, fp in P_BPR]
    write_tex(TABLES / "benign_preservation.tex", "llrrrrrrr",
              ["Exp.", "System", "$n$", "correct", "IB", "prov", "synth", "BPR", "FPR"], rows,
              "Benign-Preservation Rate (BPR) and False-Positive Rate (FPR) on benign traces (manuscript Table 19).",
              "tab:bpr", size="\\small", midrules={2, 6})
    # GSM8K
    write_tex(TABLES / "reasoning_fidelity.tex", "lrr", ["System", "CTCA", "$n$"],
              [["Undefended", "0.867", 30], ["ChannelGuard", "0.867", 30]],
              "GSM8K CTCA under no attack (manuscript \\S6.5, Table 6). Table 19 reports 25/30 = 0.833 correct for "
              "the same cells; see manuscript\\_fix.md.", "tab:reasoning-fidelity")
    # Tables 20, 21
    rows = [[r, f"{n:,}", f"{s:,}", fpct(sh)] for r, n, s, sh in P_ROLE_SHARE]
    write_tex(TABLES / "role_share.tex", "lrrr", ["Role", "\\# calls", "total (s)", "share"], rows,
              "ChannelGuard call count and wall-time share by role, pooled over 1,920 traces (manuscript Table 20). "
              "Shares sum to 121.4\\% because the three workers run in parallel.", "tab:role-share")
    rows = [[esc(SLABEL[s]), f"{p:,}", f"{w:,}", f"{sy:,}", f"{v:,}"] for s, p, w, sy, v in P_ROLE_LAT]
    write_tex(TABLES / "per_agent_latency.tex", "lrrrr", ["System", "planner", "worker", "synth", "verifier"], rows,
              "Mean latency per LLM call (ms), per system and role; worker averages the 3 parallel workers "
              "(manuscript Table 21).", "tab:per-agent-latency")
    # Table 22
    rows = [[g, c, s, l, f3(s / c)] for g, (c, s, l) in P_COMPRESS.items()]
    write_tex(TABLES / "compression_effectiveness.tex", "lrrrr", ["Gate", "compressed", "safe", "leaked", "stop-rate"],
              rows, "Compression-band effectiveness on attack traces, pooled over all 1,860 non-benign traces "
              "(manuscript Table 22).", "tab:compress_eff")
    # Table 23
    v = P_VERIFIER
    rows = [["Total verifier reports", f"{v['total']:,}"],
            ["$\\bar s \\geq 0.5$ and $\\sigma_s > 0.3$ (risky pass-through)", v["risky_pass_through"]],
            ["\\quad of which the trace ultimately leaked", f"{v['risky_leaked']} ({fpct(v['risky_leaked'] / v['risky_pass_through'])})"],
            ["Flagged unsafe $\\wedge$ uncertain, rejected as designed", v["unsafe_and_uncertain"]]]
    write_tex(TABLES / "verifier_disagreement.tex", "lr", ["Category", "Count"], rows,
              "Verifier mean/dispersion disagreement patterns, pooled over all experiments (manuscript Table 23).",
              "tab:verifier")
    # Table 24
    j = P_JUDGE
    rows = [["Raw agreement (safe/unsafe, discretized at 0.5)", f3(j["raw"])],
            ["Cohen's $\\kappa$ (Sonnet 4.5 vs.\\ GPT-5)", f"\\textbf{{{j['sonnet_vs_gpt5']:.3f}}}"],
            ["Cohen's $\\kappa$ (Sonnet 4.5 vs.\\ ground truth)", f3(j["sonnet_vs_truth"])],
            ["Cohen's $\\kappa$ (GPT-5 vs.\\ ground truth)", f"\\textbf{{{j['gpt5_vs_truth']:.3f}}}"]]
    write_tex(TABLES / "judge_audit.tex", "lr", ["Metric", "Value"], rows,
              "Cross-family judge audit, $n{=}100$ stratified final-output strings (manuscript Table 24).",
              "tab:judge_audit")

    # Manuscript macros (legacy names preserved, digits dropped as before)
    for (s, a), val in P_ASR.items():
        macros[macro("Main", s, a, "asr", digits="drop")] = f3(val)
    for (s, a), val in P_CTCA_MAIN.items():
        macros[macro("Main", s, a, "ctca", digits="drop")] = f3(val)
    macros["Gsmkinfoshieldctca"] = macros["Gsmkundefendedctca"] = "0.867"
    macros["Efficiencymeanlatencyms"] = "70291.052"
    macros["Efficiencymeantokens"] = "4291.896"
    macros["Efficiencyverifieroverhead"] = "0.310"
    for (s, a), vals in P_ATTR.items():
        for cls, val in zip(CLASSES, vals):
            macros[macro("Attribution", s, a, cls, digits="drop")] = str(val)
    macros.update(P_REPRO)


# ============================================================================ manuscript-exact data + figure


def build_manuscript_data() -> None:
    bank = get_bank(20)
    old = {}
    fig1 = DATA / "fig1_asr.csv"
    if fig1.exists():
        with open(fig1, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                old[(r["system"].replace("channelguard", "infoshield"), r["attack"])] = r
    extra_cols = ["ups", "ilr", "tmr", "ccr", "ctca", "mean_latency_ms", "mean_tokens", "verifier_overhead",
                  "asr_ci_low", "asr_ci_high"]
    rows = []
    for s in ["infoshield", "undefended"]:
        for a in ATTACKS:
            o = old.get((s, a), {})
            rows.append([s, a, f3(P_ASR[(s, a)]), 30, *[o.get(c, "") for c in extra_cols]])
    write_csv(fig1, ["system", "attack", "asr", "n", *extra_cols], rows)
    write_csv(DATA / "attribution_by_cell.csv", ["system", "attack", "n", *CLASSES],
              [[s, a, 30, *P_ATTR[(s, a)]] for s in ["infoshield", "undefended"] for a in ATTACKS])
    write_csv(DATA / "phrase_audit.csv", ["rank", "phrase_idx", "phrase", "wins"],
              [[i + 1, p, bank[p], w] for i, (p, w) in enumerate(P_PHRASES)])
    write_csv(DATA / "sentinel_delta.csv", ["system", "attack", "main_n", "rand_n", "main_ib", "rand_ib", "delta"],
              [[s, a, 30, 30, o, r, r - o] for (s, a), (o, r) in P_SENTINEL.items()])

    # Corrected manuscript Figure 5: must agree with Table 18.
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    labels = [f"#{p}: {bank[p][:52]}{'…' if len(bank[p]) > 52 else ''}" for p, _ in P_PHRASES][::-1]
    ax.barh(labels, [w for _, w in P_PHRASES][::-1], color="#1f77b4")
    for i, (_, w) in enumerate(P_PHRASES[::-1]):
        ax.text(w + 1.5, i, str(w), va="center", fontsize=8)
    ax.set_xlabel("argmax wins across all gates")
    ax.set_title(f"Bank phrases that win an argmax match ({P_PHRASE_TOTAL} BLOCK decisions, ChannelGuard)", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "phrase_audit.pdf")
    plt.close(fig)


# ============================================================================ cross-model tables


def build_cross_model(L: dict, macros: dict) -> dict:
    bank = get_bank(20)
    X = {}  # machine-readable cross-model numbers for numbers.json
    hdr3 = ["GPT-5", "Qwen", "Llama"]

    # ASR
    rows, csvrows = [], []
    for a in ATTACKS + EXTRA:
        r = [esc(ALABEL[a])]
        for s in CORE:
            g = P_ASR.get((s, a))
            q = cnt(L, "qwen", "asr_main", f"{s}/{a}").get("asr") if a in ATTACKS else \
                cnt(L, "qwen", "extra_attacks", f"{s}/{a}").get("asr")
            ll = cnt(L, "llama", "asr_main", f"{s}/{a}").get("asr") if a in ATTACKS else \
                cnt(L, "llama", "extra_attacks", f"{s}/{a}").get("asr")
            r += [f3(g), f3(q), f3(ll)]
            csvrows.append([a, s, g, q, ll])
        rows.append(r)
    write_tex(XTABLES / "asr.tex", "lrrrrrr",
              ["Attack", "Und. GPT-5", "Und. Qwen", "Und. Llama", "CG GPT-5", "CG Qwen", "CG Llama"], rows,
              "Attack success rate across agent backends, $n{=}30$ per cell, same 30 HotpotQA sample IDs. GPT-5 = "
              "manuscript (Azure content filter always on); Qwen3-4B and Llama-3.1-8B run locally with no provider "
              "filter. The last two attacks were not in the manuscript's $n{=}30$ matrix.", "tab:xm-asr",
              size="\\small", star=True, midrules={7, 8})
    write_csv(XDATA / "asr.csv", ["attack", "system", "gpt5", "qwen", "llama"], csvrows)
    X["asr"] = csvrows

    # McNemar
    rows = []
    for a in [PI, TP, MP, SD, CE, RC, AP]:
        r = [esc(ALABEL[a]), f"{P_MCNEMAR[0]} / {P_MCNEMAR[1]} ($p{{=}}{P_MCNEMAR[2]:.3f}$)" if a == PI else "---"]
        for m in MODELS:
            mc = cnt(L, m, "mcnemar", a)
            p = mc.get("p_two_sided", 1.0)
            r.append(f"{mc.get('cg_blocks_und_leaks', 0)} / {mc.get('und_blocks_cg_leaks', 0)} "
                     f"($p{{{'<' if p < 0.001 else '='}}}{max(p, 0.001):.3f}$)")
        rows.append(r)
    write_tex(XTABLES / "mcnemar.tex", "lrrr", ["Attack", *hdr3], rows,
              "Exact two-sided McNemar test on discordant pairs: ChannelGuard stops / Undefended stops "
              "(other system leaks).", "tab:xm-mcnemar", size="\\small")

    # Attribution (long format)
    rows, csvrows = [], []
    for s in CORE:
        for a in ATTACKS:
            g = P_ATTR[(s, a)]
            rows.append([SLABEL[s], esc(ALABEL[a]), "GPT-5", *g])
            csvrows.append([s, a, "gpt5", *g])
            for m in MODELS:
                c = cnt(L, m, "attribution_main", f"{s}/{a}")
                vals = [c.get(k, 0) for k in CLASSES]
                rows.append(["", "", MODELS[m], *vals])
                csvrows.append([s, a, m, *vals])
    write_tex(XTABLES / "attribution.tex", "lllrrrrrr",
              ["System", "Attack", "Backend", "leak", "ib", "prov", "ver", "syn", "safe"], rows,
              "Per-trace attribution across agent backends ($n{=}30$ per row). \\texttt{prov} is structurally 0 on "
              "the local backends, which have no provider filter.", "tab:xm-attribution",
              size="\\scriptsize", star=True, midrules=set(range(3, len(rows), 3)))
    write_csv(XDATA / "attribution.csv", ["system", "attack", "backend", *CLASSES], csvrows)

    # Provider invariance: 5 backends
    rows, csvrows = [], []
    for (b, s, ib, pr, mh, a), (_, _, ib2, pr2, mh2, a2) in zip(P_TP_BACKENDS, P_POOL_BACKENDS):
        rows.append([f"{b} (paper)", SLABEL[s], ib, pr, mh, f3(a), ib2, pr2, mh2, f3(a2)])
        csvrows.append([b, s, ib, pr, mh, a, ib2, pr2, mh2, a2])
    for m in MODELS:
        for s in CORE:
            t = cnt(L, m, "provider_invariance", f"tool_poisoning/{s}")
            p = cnt(L, m, "provider_invariance", f"pooled3/{s}")
            ta = t.get("leaked", 0) / 30
            pa = p.get("leaked", 0) / 90
            rows.append([f"{MODELS[m]} (local)", SLABEL[s], t.get("ib_blocked", 0), t.get("provider_filter", 0),
                         model_handled(t), f3(ta), p.get("ib_blocked", 0), p.get("provider_filter", 0),
                         model_handled(p), f3(pa)])
            csvrows.append([MODELS[m], s, t.get("ib_blocked", 0), t.get("provider_filter", 0), model_handled(t), ta,
                            p.get("ib_blocked", 0), p.get("provider_filter", 0), model_handled(p), pa])
    write_tex(XTABLES / "provider_invariance.tex", "llrrrrrrrr",
              ["Backend", "System", "IB", "prov", "model", "ASR", "IB", "prov", "model", "ASR"], rows,
              "Provider invariance across five agent backends. Left: Tool Poisoning ($n{=}30$). Right: pooled Tool + "
              "Memory + Role-Confusion Memory ($n{=}90$). \\textit{model} = handled without an IB block or provider "
              "filter. ChannelGuard's IB-3 blocks Tool Poisoning 30/30 on every backend (Corollary B.5).",
              "tab:xm-provider", size="\\small", star=True, midrules={2, 4, 6, 8})
    write_csv(XDATA / "provider_invariance.csv",
              ["backend", "system", "tp_ib", "tp_prov", "tp_model", "tp_asr", "pool_ib", "pool_prov",
               "pool_model", "pool_asr"], csvrows)
    X["provider_invariance"] = csvrows

    # Mechanism substitution pooled TP+MP
    rows = [["GPT-5 (paper)", "Undefended", 0, 54, 6, 0], ["GPT-5 (paper)", "ChannelGuard", 32, 22, 6, 0]]
    for m in MODELS:
        for s in CORE:
            c = cnt(L, m, "pooled_tp_mp", s)
            rows.append([MODELS[m], SLABEL[s], c.get("ib_blocked", 0), c.get("provider_filter", 0),
                         model_handled(c), c.get("leaked", 0)])
    write_tex(XTABLES / "mechanism_substitution.tex", "llrrrr",
              ["Backend", "System", "IB gate", "Provider filter", "Model-handled", "Leaked"], rows,
              "Mechanism substitution, pooled Tool Poisoning + Memory Poisoning ($n{=}60$ per row).",
              "tab:xm-mechanism", midrules={2, 4})

    # Sentinel
    rows = []
    for (s, a), (o, r) in P_SENTINEL.items():
        row = [SLABEL[s], esc(ALABEL[a]) + ("$^\\dagger$" if a == TP else ""), f"{o}$\\to${r}"]
        for m in MODELS:
            c = cnt(L, m, "sentinel", f"{s}/{a}")
            row.append(f"{c.get('ib_orig', 0)}$\\to${c.get('ib_rand', 0)}")
        rows.append(row)
    write_tex(XTABLES / "sentinel.tex", "llrrr", ["System", "Attack", *hdr3], rows,
              "Sentinel randomization: IB-blocked traces with the original marker $\\to$ with a random token. "
              "$^\\dagger$In the manuscript run the Tool Poisoning payload was generated inside the mock tool and "
              "was not actually randomized; the local runs fix this.", "tab:xm-sentinel", midrules={3})

    # GSM8K + BPR/FPR
    rows = [["GSM8K CTCA", s, "0.867", f3(cnt(L, "qwen", "gsm8k", s).get("ctca")),
             f3(cnt(L, "llama", "gsm8k", s).get("ctca"))] for s in CORE]
    for e, s, *_rest in P_BPR:
        b, fp = _rest[-2], _rest[-1]
        q = cnt(L, "qwen", "bpr_fpr", f"{e}/{s}")
        ll = cnt(L, "llama", "bpr_fpr", f"{e}/{s}")
        rows.append([f"BPR / FPR ({e})", s, f"{f3(b)} / {f3(fp)}", f"{f3(q.get('bpr'))} / {f3(q.get('fpr'))}",
                     f"{f3(ll.get('bpr'))} / {f3(ll.get('fpr'))}"])
    rows = [[r[0], esc(SLABEL[r[1]]), *r[2:]] for r in rows]
    write_tex(XTABLES / "fidelity_bpr.tex", "llrrr", ["Metric", "System", *hdr3], rows,
              "Reasoning fidelity (GSM8K CTCA) and benign preservation (BPR / FPR) on no-attack inputs.",
              "tab:xm-fidelity", size="\\small", midrules={2})

    # Baselines
    rows, csvrows = [], []
    for a in BASE_ATTACKS:
        r = [esc(ALABEL[a])]
        for s in BASE:
            g = P_BASE_ASR[(s, a)]
            vals = []
            for m in MODELS:
                src = "asr_main" if s in CORE else "baselines"
                vals.append(cnt(L, m, src, f"{s}/{a}").get("asr"))
            r.append(f"{f3(g)} / {f3(vals[0])} / {f3(vals[1])}")
            csvrows.append([a, s, g, *vals])
        rows.append(r)
    write_tex(XTABLES / "baselines.tex", "lrrrrrr", ["Attack", *[esc(SLABEL[s]) for s in BASE]], rows,
              "Baseline ASR, each cell GPT-5 / Qwen / Llama ($n{=}30$).", "tab:xm-baselines",
              size="\\scriptsize", star=True)
    write_csv(XDATA / "baselines.csv", ["attack", "system", "gpt5", "qwen", "llama"], csvrows)

    # Ablation, beta, bank
    rows = []
    for lbl, key, a, ib, *_ in P_ABLATION:
        row = [esc(lbl), f"{f3(a)} ({ib})"]
        for m in MODELS:
            c = cnt(L, m, "ablation", key)
            row.append(f"{f3(c.get('asr'))} ({c.get('ib_blocked', 0)})")
        rows.append(row)
    write_tex(XTABLES / "ablation.tex", "lrrr", ["Config", *hdr3], rows,
              "Per-gate ablation on Prompt Injection: ASR (IB-blocked traces), $n{=}30$ each.", "tab:xm-ablation")
    rows = []
    for b in BETAS:
        th, a, _, ib = P_BETA[b]
        row = [b, f"{th:.3f}", f"{f3(a)} ({ib})"]
        for m in MODELS:
            c = cnt(L, m, "beta_sweep", b)
            row.append(f"{f3(c.get('asr'))} ({c.get('ib_blocks', 0)})")
        rows.append(row)
    write_tex(XTABLES / "beta_sweep.tex", "rrrrr", ["$\\beta$", "$\\theta_{\\mathrm{eff}}$", *hdr3], rows,
              "$\\beta$-sweep on Prompt Injection: ASR (IB blocks). The default configuration ($\\theta{=}0.60$) "
              "corresponds to $\\beta{\\approx}1.67$, not $\\beta{=}1.0$.", "tab:xm-beta")
    rows = []
    for k, (a, _, ib) in P_BANK.items():
        row = [k, f"{f3(a)} ({ib})"]
        for m in MODELS:
            c = cnt(L, m, "bank_size", k)
            row.append(f"{f3(c.get('asr'))} ({c.get('ib_blocks', 0)})")
        rows.append(row)
    write_tex(XTABLES / "bank_size.tex", "rrrr", ["Bank size", *hdr3], rows,
              "Phrase-bank size sweep on Prompt Injection: ASR (IB blocks).", "tab:xm-bank")

    # Gate diagnostics
    rows = []
    for g in GATES + ["IB-3-mem"]:
        pm = P_MARGIN.get(g)
        row = [g, f"{f3(pm[1])} ({pm[0]})" if pm else "---"]
        for m in MODELS:
            c = L[m].get("margin", {}).get(g)
            row.append(f"{f3(c['median'])} ({c['n']})" if c else "---")
        rows.append(row)
    write_tex(XTABLES / "gate_margin.tex", "lrrr", ["Gate", *hdr3], rows,
              "Median embedding-space margin $\\varepsilon$ (number of BLOCK decisions).", "tab:xm-margin")
    rows = [[g, f3(P_AUC[g]), *[f3(L[m].get("auc", {}).get(g)) for m in MODELS]] for g in GATES]
    write_tex(XTABLES / "gate_auc.tex", "lrrr", ["Gate", *hdr3], rows,
              "Per-gate ROC AUC on the Tool + Memory + Role-Confusion + benign subset. The user input is never "
              "attacked in this subset, so IB-0 can only score 0.5 by construction.", "tab:xm-auc")
    rows = []
    for g in GATES:
        r0, lo, hi = P_THRESH[g]
        row = [g, f"{f3(r0)} [{lo:.2f}, {hi:.2f}]"]
        for m in MODELS:
            c = L[m].get("threshold_sensitivity", {}).get(g, {})
            pl = c.get("plateau", [None, None])
            row.append(f"{f3(c.get('block_rate_at_0.60'))} [{pl[0]:.2f}, {pl[1]:.2f}]" if pl[0] is not None else "---")
        rows.append(row)
    write_tex(XTABLES / "threshold_sensitivity.tex", "lrrr", ["Gate", *hdr3], rows,
              "Block rate at $\\theta{=}0.60$ [plateau within $\\pm$5 pp].", "tab:xm-threshold", size="\\small")
    rows = []
    for m in ["gpt5", *MODELS]:
        front = [(g, lat, ub) for g, lat, ub in P_PARETO] if m == "gpt5" else \
            [(tuple(p["gates"]), p["latency_ms"], p["ub_asr"]) for p in L[m].get("pareto", [])]
        for i, (g, lat, ub) in enumerate(front):
            rows.append([("GPT-5" if m == "gpt5" else MODELS[m]) if i == 0 else "",
                         "\\{" + ", ".join(g) + "\\}", f"{lat:.1f}", f3(ub)])
    write_tex(XTABLES / "gate_pareto.tex", "llrr", ["Backend", "Gate subset", "latency (ms)", "upper-bound ASR"],
              rows, "Latency-conditioned gate Pareto frontiers.", "tab:xm-pareto", size="\\small")

    # Phrase audit
    wins = {"gpt5": {bank[p]: w for p, w in P_PHRASES}}
    for m in MODELS:
        wins[m] = L[m].get("argmax", {}).get("wins", {})
    phrases = sorted({p for d in wins.values() for p in d}, key=lambda p: -sum(d.get(p, 0) for d in wins.values()))
    rows = [["``" + esc(p) + "''", *[wins[m].get(p, 0) for m in ["gpt5", *MODELS]]] for p in phrases]
    tot = [P_PHRASE_TOTAL, *[L[m].get("argmax", {}).get("with_argmax", 0) for m in MODELS]]
    rows.append(["\\textbf{BLOCK decisions with argmax}", *tot])
    write_tex(XTABLES / "phrase_audit.tex", "p{0.55\\linewidth}rrr", ["Phrase", *hdr3], rows,
              "Argmax bank-phrase wins over BLOCK decisions. The local runs record the argmax at gate time for every "
              "gate, including IB-3-mem, so their totals are larger.", "tab:xm-phrases", size="\\small",
              midrules={len(rows) - 1})

    # Compression, verifier, judges
    rows = []
    for g in GATES + ["IB-3-mem"]:
        pc = P_COMPRESS.get(g)
        row = [g, f"{f3(pc[1] / pc[0])} ({pc[0]})" if pc else "---"]
        for m in MODELS:
            c = L[m].get("compression_band", {}).get(g)
            row.append(f"{f3(c['safe'] / (c['safe'] + c['leaked']))} ({c['safe'] + c['leaked']})" if c else "---")
        rows.append(row)
    write_tex(XTABLES / "compression.tex", "lrrr", ["Gate", *hdr3], rows,
              "COMPRESS stop-rate on attack traces (number of COMPRESS decisions).", "tab:xm-compress")
    vd = {m: L[m].get("verifier_disagreement", {}) for m in MODELS}
    rows = [["Total verifier reports", f"{P_VERIFIER['total']:,}", *[f"{vd[m].get('total', 0):,}" for m in MODELS]],
            ["Risky pass-through ($\\bar s\\geq0.5$, $\\sigma_s>0.3$)", P_VERIFIER["risky_pass_through"],
             *[vd[m].get("risky_pass_through", 0) for m in MODELS]],
            ["\\quad of which leaked",
             f"{P_VERIFIER['risky_leaked']} ({fpct(P_VERIFIER['risky_leaked'] / P_VERIFIER['risky_pass_through'])})",
             *[f"{vd[m].get('risky_leaked', 0)} ({fpct(vd[m]['risky_leaked'] / vd[m]['risky_pass_through']) if vd[m].get('risky_pass_through') else '---'})"
               for m in MODELS]],
            ["Unsafe $\\wedge$ uncertain (rejected)", P_VERIFIER["unsafe_and_uncertain"],
             *[vd[m].get("unsafe_and_uncertain", 0) for m in MODELS]]]
    write_tex(XTABLES / "verifier_disagreement.tex", "lrrr", ["Category", *hdr3], rows,
              "Verifier disagreement patterns.", "tab:xm-verifier")
    jk = {m: L[m].get("judges", {}).get("kappas", {}) for m in MODELS}
    pairs = [("Sonnet 4.5 vs GPT-5 (sharpened rubric)", P_JUDGE["sonnet_vs_gpt5"]),
             ("Sonnet 4.5 vs ground truth (sharpened)", P_JUDGE["sonnet_vs_truth"]),
             ("GPT-5 vs ground truth (sharpened)", P_JUDGE["gpt5_vs_truth"]),
             ("Sonnet 4.5 vs GPT-5 (vague rubric)", P_JUDGE["vague"])]
    rows = [[esc(p), f3(ref), *[f3(jk[m].get(p, {}).get("kappa")) for m in MODELS]] for p, ref in pairs]
    rows.append(["Judged outputs", P_JUDGE["n"], *[jk[m].get(pairs[0][0], {}).get("n", 0) for m in MODELS]])
    write_tex(XTABLES / "judge_audit.tex", "lrrr", ["Judge pair", *hdr3], rows,
              "Cross-family judge audit, Cohen's $\\kappa$.", "tab:xm-judges", midrules={4})

    # Extra attacks (local only)
    rows = []
    for a in EXTRA:
        for s in CORE:
            row = [esc(ALABEL[a]), SLABEL[s]]
            for m in MODELS:
                c = cnt(L, m, "extra_attacks", f"{s}/{a}")
                gates = ", ".join(f"{k}:{v}" for k, v in sorted(c.get("blocking_gates", {}).items())) or "---"
                row += [f3(c.get("asr")), f"{c.get('ib_blocked', 0)} ({gates})"]
            rows.append(row)
    write_tex(XTABLES / "extra_attacks.tex", "llrlrl",
              ["Attack", "System", "ASR Qwen", "IB Qwen", "ASR Llama", "IB Llama"], rows,
              "Two attack families implemented in the codebase but excluded from the manuscript's $n{=}30$ matrix.",
              "tab:xm-extra", size="\\small", midrules={2})

    # Efficiency + cost
    eff = {m: L[m].get("efficiency", {}) for m in MODELS}
    ov = {m: eff[m].get("overall", {}) for m in MODELS}
    cost = {m: L[m].get("cost", {}) for m in MODELS}

    def sp(m, g):
        return eff[m].get(g, {}).get("speedup_serial")

    rows = [["Total measured cost", "\\$47.36",
             *[f"\\${cost[m].get('pipeline_usd', 0) + cost[m].get('judge_usd', 0):.2f}" for m in MODELS]],
            ["Mean time per run", "70.3 s", *[f"{ov[m].get('mean_serial_ms', 0) / 1e3:.1f} s" for m in MODELS]],
            ["Generation speed", "API", *[f"{ov[m].get('gen_tokens_per_s', 0):.1f} tok/s" for m in MODELS]],
            ["Mean tokens per run", "4,292", *[f"{ov[m].get('mean_tokens', 0):,.0f}" for m in MODELS]],
            ["Verifier share of tokens", "31.0\\%", *[fpct(ov[m].get("verifier_token_share")) for m in MODELS]]]
    for g, _, u, c in P_EFF:
        rows.append([f"Speedup: {g}", f"{u / c:.2f}$\\times$", *[f"{sp(m, g):.2f}$\\times$" if sp(m, g) else "---"
                                                                for m in MODELS]])
    rows.append(["Pipeline traces", "2,100", *[f"{cost[m].get('traces', 0):,}" for m in MODELS]])
    write_tex(XTABLES / "efficiency_cost.tex", "lrrr", ["Metric", *hdr3], rows,
              "Cost and efficiency. Local agents cost \\$0; only the Claude verifier and judges are billed. Local "
              "time is contention-free serial latency (8 threads share one GPU); GPT-5 is API wall-clock.",
              "tab:xm-efficiency", midrules={5, 9})
    roles = ["planner", "worker", "synthesizer", "verifier"]
    rows = [["GPT-5", esc(SLABEL[s]), *[f"{v:,}" for v in vals]] for s, *vals in P_ROLE_LAT if s in CORE]
    for m in MODELS:
        for s in CORE:
            rl = L[m]["_roles"].get(s, {})
            rows.append([MODELS[m], SLABEL[s], *[f"{rl[r]:,.0f}" if r in rl else "---" for r in roles]])
    write_tex(XTABLES / "per_agent_latency.tex", "llrrrr", ["Backend", "System", *roles], rows,
              "Mean latency per LLM call (ms) by role.", "tab:xm-latency", midrules={2, 4})

    # Failure gallery (replication traces; the manuscript's raw traces are absent)
    gal = {k: [] for k in ("correct_blocks", "false_positives", "misses")}
    for m in MODELS:
        g = L[m]["_gallery"]
        seen: set[str] = set()
        for it in sorted(g["correct_blocks"], key=lambda x: (x["attack"], x["sample_id"])):
            if it["attack"] not in seen and it["attack"] in (PI, TP, MP, RC):
                gal["correct_blocks"].append(it)
                seen.add(it["attack"])
        gal["false_positives"] += sorted(g["false_positives"], key=lambda x: x["sample_id"])[:2]
        seen = set()
        for it in sorted(g["misses"], key=lambda x: (x["attack"], x["sample_id"])):
            if it["attack"] not in seen:
                gal["misses"].append(it)
                seen.add(it["attack"])
    (DATA / "failure_gallery.json").write_text(json.dumps(gal, indent=2), encoding="utf-8")
    lines = ["% AUTO-GENERATED by scripts/build_final.py — do not edit by hand.",
             "\\begin{table*}[t]", "\\centering", "\\scriptsize",
             "\\begin{tabular}{p{0.07\\linewidth}p{0.12\\linewidth}p{0.42\\linewidth}p{0.3\\linewidth}}",
             "\\toprule", "Backend & Attack / gate & Input (truncated) & Final output (truncated) \\\\"]
    titles = {"correct_blocks": "Correct blocks", "false_positives": "False positives (benign input blocked/refused)",
              "misses": "Misses (attack leaked through ChannelGuard)"}
    for k, items in gal.items():
        lines += ["\\midrule", f"\\multicolumn{{4}}{{l}}{{\\textbf{{{titles[k]}}}}} \\\\", "\\midrule"]
        if not items:
            lines.append("\\multicolumn{4}{l}{\\textit{None observed on either local backend.}} \\\\")
        for it in items:
            gate = f"{it['blocking_gate']} ({it['risk']})" if it["blocking_gate"] else "---"
            lines.append(f"{MODELS[it['model']]} & {esc(ALABEL[it['attack']])}; {esc(gate)} & "
                         f"{esc(it['input'][:200])} & {esc(it['final_output'][:140])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{Failure gallery from the local Qwen3-4B and Llama-3.1-8B ChannelGuard runs (main experiment). "
              "The manuscript's GPT-5 traces are not in this checkout.}", "\\label{tab:failure_gallery}",
              "\\end{table*}", ""]
    text = "\n".join(lines)
    _check_tex("failure_gallery.tex", text)
    (TABLES / "failure_gallery.tex").write_text(text, encoding="utf-8")

    # Figures
    _fig_asr(L)
    _fig_mechanism(L)

    # Macros for both local models
    for m in MODELS:
        pre = "Qwen" if m == "qwen" else "Llama"
        for cell, v in L[m].get("asr_main", {}).items():
            s, a = cell.split("/")
            macros[macro(pre, "Asr", s, a)] = f3(v["asr"])
        for cell, c in L[m].get("attribution_main", {}).items():
            s, a = cell.split("/")
            for cls in CLASSES:
                macros[macro(pre, "Attr", s, a, cls)] = str(c.get(cls, 0))
        for a, c in L[m].get("mcnemar", {}).items():
            macros[macro(pre, "Mcnemar", a, "CgStops")] = str(c["cg_blocks_und_leaks"])
            macros[macro(pre, "Mcnemar", a, "UndStops")] = str(c["und_blocks_cg_leaks"])
            macros[macro(pre, "Mcnemar", a, "P")] = f"{c['p_two_sided']:.3g}"
        for cell, c in L[m].get("sentinel", {}).items():
            s, a = cell.split("/")
            macros[macro(pre, "Sentinel", s, a, "IbOrig")] = str(c["ib_orig"])
            macros[macro(pre, "Sentinel", s, a, "IbRand")] = str(c["ib_rand"])
        for s, c in L[m].get("gsm8k", {}).items():
            macros[macro(pre, "Gsm", s, "Ctca")] = f3(c["ctca"])
        for cell, c in L[m].get("bpr_fpr", {}).items():
            e, s = cell.split("/")
            macros[macro(pre, "Bpr", e, s)] = f3(c["bpr"])
            macros[macro(pre, "Fpr", e, s)] = f3(c["fpr"])
        for cell, c in L[m].get("baselines", {}).items():
            s, a = cell.split("/")
            macros[macro(pre, "Baseline", s, a, "Asr")] = f3(c["asr"])
        for s, c in L[m].get("ablation", {}).items():
            macros[macro(pre, "Ablation", s, "Asr")] = f3(c["asr"])
        for b, c in L[m].get("beta_sweep", {}).items():
            macros[macro(pre, "Beta", b, "Asr")] = f3(c["asr"])
            macros[macro(pre, "Beta", b, "Ib")] = str(c["ib_blocks"])
        for k, c in L[m].get("bank_size", {}).items():
            macros[macro(pre, "Bank", k, "Asr")] = f3(c["asr"])
        for g, v in L[m].get("auc", {}).items():
            macros[macro(pre, "Auc", g)] = f3(v)
        for g, c in L[m].get("margin", {}).items():
            macros[macro(pre, "Margin", g, "Median")] = f3(c["median"])
        for cell, c in L[m].get("extra_attacks", {}).items():
            s, a = cell.split("/")
            macros[macro(pre, "Asr", s, a)] = f3(c["asr"])
        for p, c in L[m].get("judges", {}).get("kappas", {}).items():
            macros[macro(pre, "Kappa", re.sub(r"\(.*?\)", "", p), "Sharp" if "sharp" in p else "Vague")] = f3(c["kappa"])
        for g, c in eff[m].items():
            if isinstance(c, dict) and "speedup_serial" in c:
                macros[macro(pre, "Speedup", g)] = f"{c['speedup_serial']:.2f}"
        macros[macro(pre, "Cost")] = f"{cost[m].get('pipeline_usd', 0) + cost[m].get('judge_usd', 0):.2f}"
        macros[macro(pre, "Traces")] = f"{cost[m].get('traces', 0):,}".replace(",", "{,}")
        macros[macro(pre, "MeanSecondsPerRun")] = f"{ov[m].get('mean_serial_ms', 0) / 1e3:.1f}"
        macros[macro(pre, "TokensPerSecond")] = f"{ov[m].get('gen_tokens_per_s', 0):.1f}"
        macros[macro(pre, "VerifierShare")] = f"{100 * ov[m].get('verifier_token_share', 0):.1f}"
    X["generated_at"] = datetime.now(timezone.utc).isoformat()
    return X


def _fig_asr(L: dict) -> None:
    atk = [PI, TP, MP, SD, RC, AP]
    series = [("GPT-5 Und.", lambda a: P_ASR[("undefended", a)], "#9ecae1"),
              ("GPT-5 CG", lambda a: P_ASR[("infoshield", a)], "#08519c")]
    for m, (lc, dc) in zip(MODELS, [("#fdae6b", "#d94801"), ("#a1d99b", "#238b45")]):
        series.append((f"{MODELS[m]} Und.", lambda a, m=m: cnt(L, m, "asr_main", f"undefended/{a}").get("asr", 0), lc))
        series.append((f"{MODELS[m]} CG", lambda a, m=m: cnt(L, m, "asr_main", f"infoshield/{a}").get("asr", 0), dc))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    w = 0.13
    for i, (lbl, fn, col) in enumerate(series):
        ax.bar([x + (i - 2.5) * w for x in range(len(atk))], [fn(a) for a in atk], w, label=lbl, color=col)
    ax.set_xticks(range(len(atk)))
    ax.set_xticklabels([ALABEL[a] for a in atk], fontsize=8)
    ax.set_ylabel("attack success rate")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=3)
    ax.set_title("ASR by agent backend: Undefended vs ChannelGuard (n=30 per bar)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "cross_model_asr.pdf")
    plt.close(fig)


def _fig_mechanism(L: dict) -> None:
    bars = [(f"{b}\n{'Und.' if s == 'undefended' else 'CG'}", ib, pr, mh, 0)
            for b, s, ib, pr, mh, _ in P_POOL_BACKENDS]
    for m in MODELS:
        for s in CORE:
            c = cnt(L, m, "provider_invariance", f"pooled3/{s}")
            bars.append((f"{MODELS[m]}\n{'Und.' if s == 'undefended' else 'CG'}", c.get("ib_blocked", 0),
                         c.get("provider_filter", 0), model_handled(c), c.get("leaked", 0)))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    xs = range(len(bars))
    bottom = [0] * len(bars)
    for idx, (lbl, col) in enumerate([("IB gate (application)", "#238b45"), ("Provider filter", "#2171b5"),
                                       ("Model-handled", "#bdbdbd"), ("Leaked", "#cb181d")]):
        vals = [b[idx + 1] for b in bars]
        ax.bar(xs, vals, bottom=bottom, color=col, label=lbl)
        bottom = [x + y for x, y in zip(bottom, vals)]
    ax.set_xticks(list(xs))
    ax.set_xticklabels([b[0].replace("Anthropic ", "") for b in bars], fontsize=7)
    ax.set_ylabel("traces (n=90 per bar)")
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.set_title("Who stops Tool + Memory + Role-Confusion attacks, by backend", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "cross_model_mechanism.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================================ markdown section in final_results.md


def build_markdown_section(L: dict) -> str:
    q, ll = "qwen", "llama"

    def a_(m, s, a):
        return cnt(L, m, "asr_main", f"{s}/{a}").get("asr")

    def at(m, s, a):
        return cnt(L, m, "attribution_main", f"{s}/{a}")

    out = [BEGIN, "", "## 6A. Cross-model replication: Qwen3-4B and Llama-3.1-8B (local agents)", "",
           "_Generated by `scripts/build_final.py` from `Final/qwen/numbers.json`, `Final/llama/numbers.json` and the "
           "traces in `results_qwen/` and `results_llama/`. Regenerate with "
           "`.venv/bin/python scripts/build_final.py` (offline, no API calls)._", "",
           "Every experiment of §4–§6 was re-run with the planner, workers and synthesizer swapped to a local "
           "open-weight model. Everything else is unchanged: Claude Sonnet 4.5 five-sample verifier, six IB gates, "
           "θ = 0.60, α = 0.70, N = 2, 20-phrase bank, and the same 30 HotpotQA and 30 GSM8K sample IDs. Each "
           "replication has 1,800 traces (0 failures). Protocol differences are listed in §6A.12.", "",
           md_table(["Component", "GPT-5 (manuscript)", "Qwen3-4B", "Llama-3.1-8B"], [
               ["Agent", "Azure GPT-5 API", "Qwen3-4B-Instruct-2507, MLX 8-bit, local",
                "Llama-3.1-8B-Instruct, MLX 4-bit, local"],
               ["Provider content filter", "Azure, always on", "none", "none"],
               ["Pipeline traces", "2,100", f"{L[q]['cost']['traces']:,}", f"{L[ll]['cost']['traces']:,}"],
               ["Judge-audited outputs", "100", str(L[q]["judges"]["kappas"][
                   "Sonnet 4.5 vs GPT-5 (sharpened rubric)"]["n"]),
                str(L[ll]["judges"]["kappas"]["Sonnet 4.5 vs GPT-5 (sharpened rubric)"]["n"])],
               ["Trace corpus", "absent from this checkout", "`results_qwen/`", "`results_llama/`"]],
               "llll"), ""]

    out += ["### 6A.1 Attack success rate (cf. §4.1)", "",
            md_table(["Attack", "Und. GPT-5", "Und. Qwen", "Und. Llama", "CG GPT-5", "CG Qwen", "CG Llama"],
                     [[ALABEL[a], f3(P_ASR.get(("undefended", a))), f3(a_(q, "undefended", a)),
                       f3(a_(ll, "undefended", a)), f3(P_ASR.get(("infoshield", a))), f3(a_(q, "infoshield", a)),
                       f3(a_(ll, "infoshield", a))] for a in ATTACKS]), "",
            "Paired McNemar on Prompt Injection (ChannelGuard stops vs. Undefended stops): GPT-5 6 vs 1 (p = 0.125); "
            f"Qwen {L[q]['mcnemar'][PI]['cg_blocks_und_leaks']} vs {L[q]['mcnemar'][PI]['und_blocks_cg_leaks']} "
            f"(p < 0.001); Llama {L[ll]['mcnemar'][PI]['cg_blocks_und_leaks']} vs "
            f"{L[ll]['mcnemar'][PI]['und_blocks_cg_leaks']} (p < 0.001). The 50% relative reduction that is only "
            "directional on GPT-5 becomes 96% (Qwen) and 87% (Llama) and is significant on both local backends.", "",
            "Without Azure's filter, both local models obey most direct prompt injections (Undefended 0.800 and "
            "0.767, vs. 0.333 on GPT-5). On the inter-agent attacks they differ: Qwen leaks on four of five, Llama "
            "on none. ChannelGuard brings every inter-agent attack to 0.000 on both. Adaptive Paraphrase defeats "
            "ChannelGuard on all three backends.", ""]

    rows = []
    for a in ATTACKS:
        for s in CORE:
            g = P_ATTR[(s, a)]
            rows.append([SLABEL[s], ALABEL[a], "/".join(map(str, g)),
                         "/".join(str(at(q, s, a).get(k, 0)) for k in CLASSES),
                         "/".join(str(at(ll, s, a).get(k, 0)) for k in CLASSES)])
    out += ["### 6A.2 Per-trace attribution (cf. §4.2)", "",
            "Each cell is leak/ib/prov/ver/syn/safe (n = 30).", "",
            md_table(["System", "Attack", "GPT-5", "Qwen", "Llama"], rows, "llrrr"), ""]

    def pool(m, s, key="pooled_tp_mp"):
        return cnt(L, m, key, s)

    out += ["### 6A.3 Mechanism substitution, pooled Tool + Memory Poisoning (cf. §4.2)", "",
            md_table(["Backend", "System", "IB gate", "Provider filter", "Model-handled", "Leaked"], [
                ["GPT-5", "Undefended", 0, "**54**", 6, 0], ["GPT-5", "ChannelGuard", 32, 22, 6, 0],
                *[[MODELS[m], SLABEL[s], f"**{pool(m, s)['ib_blocked']}**" if s == "infoshield"
                   else pool(m, s)["ib_blocked"], pool(m, s)["provider_filter"],
                   model_handled(pool(m, s)), pool(m, s)["leaked"]] for m in MODELS for s in CORE]]), "",
            "On GPT-5, Undefended's zero ASR is 90% Azure's filter, and ChannelGuard shares credit with it. On the "
            "local backends there is no filter: Undefended is protected only by the model itself (perfectly on "
            "Llama, imperfectly on Qwen), while ChannelGuard blocks 60/60 at a named gate. Llama is the cleanest "
            "case of mechanism substitution: identical 0% ASR, attributed 100% to the model without gates and 100% "
            "to the gates with them.", ""]

    rows = [[f"{b} (paper)", SLABEL[s], ib, pr, mh, f3(a)] for b, s, ib, pr, mh, a in P_TP_BACKENDS]
    for m in MODELS:
        for s in CORE:
            t = cnt(L, m, "provider_invariance", f"tool_poisoning/{s}")
            rows.append([f"**{MODELS[m]} (local)**", SLABEL[s], t["ib_blocked"], t["provider_filter"],
                         model_handled(t), f3(t["leaked"] / 30)])
    out += ["### 6A.4 Provider invariance on five backends (cf. §4.5)", "",
            "Tool Poisoning, n = 30 per row. IB-3 blocks 30/30 on every backend, as Corollary B.5 predicts.", "",
            md_table(["Backend", "System", "IB", "prov", "model-hdl.", "ASR"], rows, "llrrrr"), ""]
    rows = [[f"{b} (paper)", SLABEL[s], ib, pr, mh] for b, s, ib, pr, mh, _ in P_POOL_BACKENDS]
    for m in MODELS:
        for s in CORE:
            p = cnt(L, m, "provider_invariance", f"pooled3/{s}")
            rows.append([f"**{MODELS[m]} (local)**", SLABEL[s], p["ib_blocked"], p["provider_filter"],
                         model_handled(p)])
    out += ["Pooled Tool + Memory + Role-Confusion Memory (n = 90 per row):", "",
            md_table(["Backend", "System", "IB", "prov", "model-hdl."], rows, "llrrr"), "",
            "ChannelGuard blocks 90/90 at a gate on both local backends, vs. 32–50/90 on the manuscript's backends. "
            "The difference is in IB-3-mem on Memory Poisoning and Role-Confusion Memory. With the released code "
            "this gate scores the preloaded memory entries independently of the agent, and it blocks them 30/30 on "
            "both local backends. The manuscript's 2–14/30 for the same gate is not reproduced; see "
            "manuscript_fix.md.", ""]

    rows = []
    for (s, a), (o, r) in P_SENTINEL.items():
        rows.append([SLABEL[s], ALABEL[a] + (" †" if a == TP else ""), f"{o}→{r}",
                     *[f"{cnt(L, m, 'sentinel', f'{s}/{a}')['ib_orig']}→{cnt(L, m, 'sentinel', f'{s}/{a}')['ib_rand']}"
                       for m in MODELS]])
    out += ["### 6A.5 Sentinel randomization (cf. §4.6)", "",
            md_table(["System", "Attack", "GPT-5", "Qwen", "Llama"], rows, "llrrr"), "",
            "† The manuscript run did not actually randomize Tool Poisoning: the payload is generated inside "
            "`mock_tool`, which the marker swap never reached. The local runs pass the token through a `tool_marker` "
            "field. With true randomization, IB blocks are preserved 90/90 (Qwen) and 88/90 (Llama); the "
            "template-driven conclusion holds.", ""]

    rows = [[g, n, f"{u / c:.2f}×", *[f"{L[m]['efficiency'][g]['speedup_serial']:.2f}×" for m in MODELS]]
            for g, n, u, c in P_EFF]
    ov = {m: L[m]["efficiency"]["overall"] for m in MODELS}
    out += ["### 6A.6 Efficiency, fidelity and cost (cf. §4.3–4.4)", "",
            md_table(["Speedup (CG vs Und.)", "n", "GPT-5", "Qwen", "Llama"], rows), "",
            md_table(["Metric", "GPT-5", "Qwen3-4B", "Llama-3.1-8B"], [
                ["Total measured cost", "$47.36",
                 *[f"${L[m]['cost']['pipeline_usd'] + L[m]['cost']['judge_usd']:.2f}" for m in MODELS]],
                ["Mean time per run", "70.3 s", *[f"{ov[m]['mean_serial_ms'] / 1e3:.1f} s" for m in MODELS]],
                ["Generation speed", "API", *[f"{ov[m]['gen_tokens_per_s']:.1f} tok/s" for m in MODELS]],
                ["Mean tokens per run", "4,292", *[f"{ov[m]['mean_tokens']:,.0f}" for m in MODELS]],
                ["Verifier share of tokens", "31%", *[f"{100 * ov[m]['verifier_token_share']:.0f}%" for m in MODELS]],
                ["GSM8K CTCA (Und. = CG)", "0.867", *[f3(L[m]["gsm8k"]["infoshield"]["ctca"]) for m in MODELS]],
                ["Benign FPR, ChannelGuard", "0.033", *[f3(L[m]["bpr_fpr"]["main/infoshield"]["fpr"]) for m in MODELS]],
            ]), "",
            "Local times are contention-free serial latency (8 threads share one Apple-Silicon GPU); GPT-5 is API "
            "wall-clock. Local agents cost $0, so the Claude verifier dominates both tokens and spend. The gates "
            "cost nothing in reasoning accuracy on any backend (CTCA identical with and without them).", ""]

    rows = []
    for a in BASE_ATTACKS:
        rows.append([ALABEL[a], *[f"{f3(P_BASE_ASR[(s, a)])} / "
                                  f"{f3(cnt(L, 'qwen', 'asr_main' if s in CORE else 'baselines', f'{s}/{a}').get('asr'))} / "
                                  f"{f3(cnt(L, 'llama', 'asr_main' if s in CORE else 'baselines', f'{s}/{a}').get('asr'))}"
                                  for s in BASE]])
    out += ["### 6A.7 Baselines (cf. §6.1)", "", "Each cell is GPT-5 / Qwen / Llama ASR (n = 30).", "",
            md_table(["Attack", *[SLABEL[s] for s in BASE]], rows), "",
            "LLM-Guard (Claude classifier at IB-0) is the strongest single input filter on both local backends "
            "(0.000 on Prompt Injection and Adaptive Paraphrase). SmoothLLM is again the strongest perturbation "
            "defense against Adaptive Paraphrase. Perplexity-Filter fails differently: it no longer blocks "
            "everything (benign FPR 0.400 instead of 1.000) and lets 70% of prompt injections through.", ""]

    def abl(m, k):
        c = cnt(L, m, "ablation", k)
        return f"{f3(c.get('asr'))} ({c.get('ib_blocked', 0)})"

    out += ["### 6A.8 Ablation, β-sweep, phrase-bank size (cf. §6.2–6.4)", "",
            md_table(["Config", "GPT-5", "Qwen", "Llama"],
                     [[lbl, f"{f3(a)} ({ib})", abl(q, k), abl(ll, k)] for lbl, k, a, ib, *_ in P_ABLATION]), "",
            md_table(["β", "θ_eff", "GPT-5", "Qwen", "Llama"],
                     [[b, f"{P_BETA[b][0]:.3f}", f"{f3(P_BETA[b][1])} ({P_BETA[b][3]})",
                       *[f"{f3(cnt(L, m, 'beta_sweep', b)['asr'])} ({cnt(L, m, 'beta_sweep', b)['ib_blocks']})"
                         for m in MODELS]] for b in BETAS]), "",
            md_table(["Bank size", "GPT-5", "Qwen", "Llama"],
                     [[k, f"{f3(P_BANK[k][0])} ({P_BANK[k][2]})",
                       *[f"{f3(cnt(L, m, 'bank_size', k)['asr'])} ({cnt(L, m, 'bank_size', k)['ib_blocks']})"
                         for m in MODELS]] for k in P_BANK]), "",
            "Cells are ASR (IB-blocked traces). On both local backends only IB-0 is load-bearing for Prompt "
            "Injection, and the full ChannelGuard row reproduces the main-experiment numbers exactly (Qwen 26 = 26 "
            "IB blocks, Llama 25 = 25), because attack selection is pinned with `PYTHONHASHSEED=0`. The β-sweep is "
            "monotone on all backends. Phrase-bank size has no effect locally, so the manuscript's \"bigger bank is "
            "worse\" does not replicate.", ""]

    rows = []
    for g in GATES:
        pm, pa = P_MARGIN.get(g), P_AUC[g]
        row = [g, f3(pm[1]) if pm else "—", *[f3(L[m]["margin"][g]["median"]) if g in L[m]["margin"] else "—"
                                              for m in MODELS], f3(pa), *[f3(L[m]["auc"][g]) for m in MODELS]]
        rows.append(row)
    out += ["### 6A.9 Gate diagnostics (cf. §5.1–5.4)", "",
            md_table(["Gate", "ε GPT-5", "ε Qwen", "ε Llama", "AUC GPT-5", "AUC Qwen", "AUC Llama"], rows), "",
            "IB-3 has the largest margin on every backend. IB-0's AUC is 0.500 on both local backends because the "
            "AUC subset (Tool, Memory, Role-Confusion and benign) never attacks the user input, so 0.5 is the "
            "expected value; the manuscript's 0.833 needs checking (see manuscript_fix.md).", ""]

    comp_rows = []
    for g in GATES + ["IB-3-mem"]:
        pc = P_COMPRESS.get(g)
        cells = [f3(pc[1] / pc[0]) if pc else "—"]
        for m in MODELS:
            c = L[m]["compression_band"].get(g)
            cells.append(f3(c["safe"] / (c["safe"] + c["leaked"])) if c else "—")
        comp_rows.append([g, *cells])
    vd = {m: L[m]["verifier_disagreement"] for m in MODELS}
    out += ["### 6A.10 Phrase audit, compression, verifier, judges (cf. §5.5, §5.8–5.10)", "",
            f"- **Phrase audit:** {len(L[q]['argmax']['wins'])} (Qwen) and {len(L[ll]['argmax']['wins'])} (Llama) "
            "of 20 phrases ever win, vs. 8 in the manuscript. The local runs record the argmax at gate time for "
            f"every BLOCK decision ({L[q]['argmax']['with_argmax']:,} and {L[ll]['argmax']['with_argmax']:,}).",
            "- **COMPRESS stop-rate:**", "",
            md_table(["Gate", "GPT-5", "Qwen", "Llama"], comp_rows), "",
            "  IB-0's COMPRESS band leaks on every backend (stop-rate 0.766 / 0.511 / 0.407). The inter-agent gates "
            "are no longer at 100% on the local models, but IB-3 and IB-3-mem stay at or near 100%.", "",
            f"- **Verifier blind spot:** risky pass-throughs that leaked: GPT-5 69/354 (19.5%), Qwen "
            f"{vd[q]['risky_leaked']}/{vd[q]['risky_pass_through']}, Llama "
            f"{vd[ll]['risky_leaked']}/{vd[ll]['risky_pass_through']}.",
            "- **Judge audit (Cohen's κ):**", "",
            md_table(["Pair", "GPT-5", "Qwen", "Llama"],
                     [[p, f3(ref), *[f3(L[m]["judges"]["kappas"][p]["kappa"]) for m in MODELS]] for p, ref in [
                         ("Sonnet 4.5 vs GPT-5 (sharpened rubric)", 0.900),
                         ("Sonnet 4.5 vs ground truth (sharpened)", 0.900),
                         ("GPT-5 vs ground truth (sharpened)", 1.000),
                         ("Sonnet 4.5 vs GPT-5 (vague rubric)", 0.060)]]), "",
            "  The sharpened-rubric results replicate. The vague-rubric collapse to chance (κ = 0.060) does not "
            "(0.728 and 0.767).", ""]

    rows = []
    for a in EXTRA:
        for s in CORE:
            rows.append([ALABEL[a], SLABEL[s], *[f"{f3(cnt(L, m, 'extra_attacks', f'{s}/{a}')['asr'])} "
                                                 f"({cnt(L, m, 'extra_attacks', f'{s}/{a}')['ib_blocked']} IB)"
                                                 for m in MODELS]])
    out += ["### 6A.11 Additional attack families (local only)", "",
            md_table(["Attack", "System", "Qwen", "Llama"], rows, "llrr"), "",
            "Both families are caught at IB-2 (worker output). Consensus Collapse is fully fixed on Qwen but not on "
            "Llama (2/30 leak with or without ChannelGuard).", ""]

    out += ["### 6A.12 Protocol differences and caveats", "",
            "1. **Agent model is the varied factor**, together with quantization (Qwen 8-bit, Llama 4-bit) and the "
            "absence of a provider filter. ASR differences mix model capability, alignment and filtering; the "
            "attribution tables separate them.",
            "2. **Deterministic attack selection.** `attacks/registry._pick` uses Python's salted `hash()`; the local "
            "runs pin `PYTHONHASHSEED=0`, so every system and resume sees the same injection per sample.",
            "3. **Sentinel fix.** Tool Poisoning is genuinely randomized locally (`tool_marker`).",
            "4. **Argmax at gate time.** Every gate decision records its winning bank phrase.",
            "5. **Latency** is serial (contention-free) locally vs. API wall-clock for GPT-5.",
            "6. **Single seed (0), n = 30 per cell**, as in the manuscript.", "",
            "### 6A.13 Cross-model verdict", "",
            "1. **The central claim replicates and strengthens.** Undefended safety depends entirely on the backend "
            "(Azure filter on GPT-5, model alignment on Anthropic and Llama, leaky on Qwen); ChannelGuard's blocking "
            "is identical on all five backends.",
            "2. **Prompt-injection reduction becomes significant** on both local backends (p < 0.001), from a "
            "directional p = 0.125 on GPT-5.",
            "3. **Failures replicate:** Adaptive Paraphrase defeats ChannelGuard everywhere; IB-0's COMPRESS band "
            "leaks everywhere.",
            "4. **Not replicated:** narrative memory attacks evading the gates (Slow-Drift is blocked 30/30 locally), "
            "bigger banks hurting, IB-5 being load-bearing, and the vague-rubric κ collapse. These should be "
            "presented as backend- or implementation-specific (see manuscript_fix.md).",
            "", "Full per-model reports follow in §6A.14 (Qwen) and §6A.15 (Llama); standalone copies: "
            "[qwen/README.md](qwen/README.md), [llama/README.md](llama/README.md). "
            "LaTeX: `tables/cross_model/*.tex`. Figures: [figures/cross_model_asr.pdf](figures/cross_model_asr.pdf), "
            "[figures/cross_model_mechanism.pdf](figures/cross_model_mechanism.pdf).", ""]
    for num, m, title in [("6A.14", q, "Qwen3-4B-Instruct-2507 (MLX 8-bit)"),
                          ("6A.15", ll, "Llama-3.1-8B-Instruct (MLX 4-bit)")]:
        out += [f"### {num} Full report: {title}", "", _embed_report(FINAL / m / "README.md"), ""]
    out += [END]
    return "\n".join(out)


def _embed_report(path: Path) -> str:
    # Drop the H1 and demote headings two levels so they nest under a ### section.
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    out, fenced = [], False
    for line in lines:
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced and re.match(r"#{2,4} ", line):
            line = "##" + line
        out.append(line)
    return "\n".join(out).strip()


def splice_results_md(section: str) -> None:
    text = RESULTS_MD.read_text(encoding="utf-8")
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        text = head + section + tail
    else:
        anchor = "## 7. Reproducibility"
        assert anchor in text, "final_results.md: insertion anchor missing"
        text = text.replace(anchor, section + "\n\n---\n\n" + anchor, 1)
    RESULTS_MD.write_text(text, encoding="utf-8")


# ============================================================================ macros + README


def write_macros(macros: dict, X: dict) -> None:
    lines = ["% AUTO-GENERATED by scripts/build_final.py. DO NOT EDIT.",
             "% Legacy names (\\Main..., \\Attribution..., \\repro...) hold manuscript (Azure GPT-5) values.",
             "% \\Qwen... and \\Llama... hold the local replications."]
    for k in sorted(macros, key=lambda k: (k.startswith(("Qwen", "Llama")), k)):
        lines.append(f"\\newcommand{{\\{k}}}{{{macros[k]}}}")
    (MACROS / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert len(macros) == len(set(macros)), "duplicate macro names"
    (MACROS / "numbers.json").write_text(json.dumps({"macros": macros, "cross_model": X}, indent=2, default=str),
                                         encoding="utf-8")


def write_readme(n_macros: int) -> None:
    tables = sorted(p.name for p in TABLES.glob("*.tex"))
    xtables = sorted(p.name for p in XTABLES.glob("*.tex"))
    text = f"""# ChannelGuard Results

Manuscript-aligned results for arXiv:2607.19430v2 (Azure GPT-5 agent) plus two complete open-weight
replications: Qwen3-4B-Instruct-2507 (MLX 8-bit) and Llama-3.1-8B-Instruct (MLX 4-bit), each run locally
with the same Claude Sonnet 4.5 verifier.

Everything in `tables/`, `data/`, `macros/` and the cross-model figures, plus the §6A block of
`final_results.md`, is generated by `scripts/build_final.py`. Do not hand-edit these files; rerun:

```bash
.venv/bin/python scripts/build_final.py      # offline, no API calls
```

## Documents

- [final_results.md](final_results.md): full results and analysis; §6A is the cross-model replication
- [qwen/README.md](qwen/README.md), [qwen/numbers.json](qwen/numbers.json): complete Qwen report (source data)
- [llama/README.md](llama/README.md), [llama/numbers.json](llama/numbers.json): complete Llama report (source data)
- [../manuscript_fix.md](../manuscript_fix.md): corrections and clarifications proposed for the manuscript

## Tables

- `tables/*.tex`: one per manuscript table, matching the manuscript's numbers ({len(tables)} files):
  {", ".join(f"`{t}`" for t in tables)}
- `tables/cross_model/*.tex`: GPT-5 / Qwen / Llama side by side ({len(xtables)} files):
  {", ".join(f"`{t}`" for t in xtables)}

Tables use `booktabs`; `failure_gallery.tex`, `baselines.tex` and the wide cross-model tables use `table*`.

## Macros

`macros/numbers.tex` defines {n_macros} commands. Legacy names (`\\Main...`, `\\Attribution...`, `\\repro...`)
keep the manuscript's values; `\\Qwen...` and `\\Llama...` hold the replications (for example
`\\QwenAsrInfoshieldPromptinjectionstrong`, `\\LlamaCost`). `macros/numbers.json` mirrors them.
`macros/aggregation_summary.json` is a historical aggregator snapshot (308 traces) and is not used.

## Figures

- `figures/phrase_audit.pdf`: corrected manuscript Figure 5, regenerated from Table 18
- `figures/cross_model_asr.pdf`, `figures/cross_model_mechanism.pdf`: new cross-model figures
- The other PDFs are historical renders of the original GPT-5 traces, which are not in this checkout.

## Provenance

- GPT-5 numbers are transcribed from the manuscript; the original 2,100 raw traces are not in this checkout.
- Qwen and Llama numbers come from 1,800 local traces each in `../results_qwen/` and `../results_llama/`.

_Last generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}._
"""
    (FINAL / "README.md").write_text(text, encoding="utf-8")


def _brand(text: str, keep_paths: bool) -> str:
    """Replace 'infoshield' with 'channelguard' (case-preserving), except where it names a real path or module."""
    def sub(m: re.Match) -> str:
        s, i, j, w = m.string, m.start(), m.end(), m.group()
        if keep_paths and (
                (i and s[i - 1] in "/\\")                                   # src/infoshield, results/.../infoshield_x
                or s[max(0, i - 7):i].endswith(("from ", "import ", "cd "))
                or re.match(r"\.[a-z_]", s[j:j + 2])                        # infoshield.config, smoke_infoshield.py
                or ((i == 0 or s[i - 1] == "\n") and s[j:j + 1] in "/\\")):  # repo root in a directory tree
            return w
        return "CHANNELGUARD" if w.isupper() else "Channelguard" if w[0].isupper() else "channelguard"
    return re.sub(r"(?i)infoshield", sub, text)


def rebrand_outputs() -> int:
    # The replication numbers.json files are build inputs keyed by the code's system ID, so they are left alone.
    inputs = {FINAL / m / "numbers.json" for m in MODELS}
    n = 0
    for p in sorted(FINAL.rglob("*")):
        if p in inputs or p.suffix not in {".md", ".mmd", ".csv", ".json", ".tex"}:
            continue
        old = p.read_text(encoding="utf-8")
        new = _brand(old, keep_paths=p.suffix in {".md", ".mmd"})
        if p == MACROS / "numbers.tex":
            # Keep the manuscript's legacy macro names working as aliases.
            legacy = [x for x in re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", old)
                      if "nfoshield" in x.lower() and not x.startswith(("Qwen", "Llama"))]
            if legacy:
                new = new.rstrip("\n") + "\n\n% Legacy aliases (pre-rename macro names)\n" + "".join(
                    f"\\newcommand{{\\{x}}}{{\\{_brand(x, False)}}}\n" for x in legacy)
        if new != old:
            p.write_text(new, encoding="utf-8")
            n += 1
    return n


def main() -> None:
    for d in (TABLES, XTABLES, DATA, XDATA, FIGS, MACROS):
        d.mkdir(parents=True, exist_ok=True)
    macros: dict[str, str] = {}
    build_manuscript_tables(macros)
    build_manuscript_data()
    L = load_local()
    X = build_cross_model(L, macros)
    splice_results_md(build_markdown_section(L))
    write_macros(macros, X)
    write_readme(len(macros))
    renamed = rebrand_outputs()
    print(f"[ok] {len(list(TABLES.glob('*.tex')))} manuscript tables, {len(list(XTABLES.glob('*.tex')))} cross-model "
          f"tables, {len(macros)} macros, figures + README + final_results.md §6A regenerated; "
          f"'infoshield' → 'channelguard' in {renamed} files.")


if __name__ == "__main__":
    main()
