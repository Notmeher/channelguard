# Reproducibility Guide

**Every number, table, and figure in [INFOSHIELD.md](INFOSHIELD.md) and [paper/deliverable/FINAL.md](paper/deliverable/FINAL.md) is regenerable from this repository.** This document tells you exactly how — from a fresh clone to the final artifact — with cost and time estimates for each step.

- **Total spend to reproduce the current artifact from scratch**: ~$105 in Azure OpenAI + Anthropic API calls
- **Total spend to regenerate all reports + figures from the 605 traces already on disk**: **$0** (local-only)
- **Wall-clock end-to-end from a fresh clone (all API batches)**: ~2 hours at 8-way parallelism
- **Wall-clock to regenerate every deliverable from cached traces**: ~2 minutes

Every step below has been executed exactly as documented on the maintainer's machine (macOS 15 arm64, Python 3.12.9, uv, `pipeline.parallel_samples=8`).

---

## 0. Prerequisites

| Requirement | Version tested | Notes |
|---|---|---|
| OS | macOS 15 arm64 (Apple Silicon) | Also expected to work on Linux; Windows via WSL |
| Python | 3.12.9 | 3.10–3.12 all supported |
| [`uv`](https://github.com/astral-sh/uv) | ≥ 0.4 | Optional (any `pip`/`venv` works) |
| Disk | ~2 GB free | 500 MB for `gpt2` local model, ~500 MB HF datasets cache, ~200 MB for `results/` traces |
| RAM | 4 GB free | Sentence-Transformers + tokenizer models are small |
| Network | Required for API + HF downloads | Not required for offline test suite or report regeneration from cached traces |

**Accounts you must have:**

- Azure OpenAI resource with a `gpt-5` deployment and a `text-embedding-3-large` deployment (for the encoder-comparison experiment; local MiniLM is the default)
- Anthropic API key (Claude Sonnet 4.5)
- HuggingFace account/token (for MiniLM + gpt2 + dataset downloads; free tier is sufficient)

---

## 1. One-time setup

```bash
git clone https://github.com/Notmeher/infoshield.git
cd infoshield

# Create venv and install (uv is fastest; pip works identically)
uv venv --python 3.12
source .venv/bin/activate       # Windows: .venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"
```

Create `.env` in the repo root (**never committed**):

```dotenv
# --- Azure OpenAI (planner / worker / synthesizer) ---
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-5

# --- Azure embeddings (only needed for encoder_comparison experiment) ---
AZURE_OPENAI_EMBED_API_KEY=...
AZURE_OPENAI_EMBED_API_VERSION=2024-12-01-preview
AZURE_OPENAI_EMBED_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-large
AZURE_OPENAI_EMBED_DIMENSIONS=1536

# --- Anthropic (verifier + LLM judges) ---
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929

# --- HuggingFace (MiniLM + gpt2 + datasets) ---
Hf_api_key=hf_...
```

**Sanity check** (no API calls, ~15 s):

```bash
python -m pytest -m "not network" -q
# Expected: 56 passed, 2 skipped, 4 deselected
```

**With-network check** (~30 s, ~$0.02 in real API pings):

```bash
python -m pytest --run-network -q
# Expected: 56 passed, 8 network passed
```

If either fails, do not proceed with the paid experiments — investigate `AZURE_OPENAI_*` / `ANTHROPIC_API_KEY` first.

---

## 2. The zero-cost path: regenerate everything from cached traces

If you have the 605 traces already on disk under `results/` (which is the state of the maintainer's tree), every table and figure in the paper is regenerable **without any API calls**.

```bash
# 1. Refresh paper/deliverable/FINAL.md (main safety table + attribution)
python scripts/run_final.py --report-only
# → wall-clock: ~5 s, cost: $0

# 2. Refresh all matplotlib figures + CSVs
python scripts/consolidate.py
# → wall-clock: ~20 s, cost: $0

# 3. Refresh the five "free-analysis" artifacts (Tier D + novelty tier)
python scripts/analyze_gates.py                  # ROC/AUC + threshold sensitivity
python scripts/analyze_certified_radius.py       # first-order local margin (embedding space)
python scripts/analyze_argmax_phrases.py         # which phrase caught what
python scripts/analyze_gate_pareto.py            # cost-conditional Pareto
python figures/fig8_coverage_matrix.py           # system × attack heatmap
# → wall-clock: ~30 s, cost: $0
```

After running these, everything under `paper/deliverable/` will exactly match what is described in [INFOSHIELD.md](INFOSHIELD.md).

---

## 3. The full-cost path: re-produce the 605 traces from scratch

⚠ Only do this if you are validating the paper's numbers on your own API accounts, or you have deleted `results/`.

The full artifact was produced in **ten sequential batches**. Each batch is idempotent (resumes from cached traces on disk) and each script accepts `--dry-run` to preview cost.

| # | Script | Runs | Est. cost | Est. wall-clock | Purpose |
|---:|---|---:|---:|---:|---|
| 1 | `run_final.py` (initial + Tier 1) | 238 | ~$50 | ~35 min | Main safety table + ablation + β sweep |
| 2 | `run_baselines.py --attacks prompt_injection_strong adaptive_paraphrase` | 36 | ~$9 | ~8 min | 3 baselines on input-facing attacks |
| 3 | `run_baselines.py --attacks tool_poisoning` | 30 | ~$7.50 | ~7 min | 3 baselines on `tool_poisoning` (mechanism-substitution Ⅰ evidence) |
| 4 | `run_missing_attacks.py` | 36 | ~$9 | ~30 min | memory/reasoning_state/consensus (mechanism-substitution Ⅱ) |
| 5 | `run_judges.py` v1 (Sonnet-only) | ~30 judge calls | ~$4 | ~7 min | Deprecated by v2 |
| 6 | `run_multiturn_attacks.py` | 36 | ~$9 | ~10 min | 3 stateful-memory attacks (mechanism-substitution Ⅲ) |
| 7 | `run_judges.py` v2 (F2 fix: gpt-5 as primary Judge C) | ~30 judge calls | ~$4 | ~10 min | Cross-family Cohen's κ over 30 borderline traces |
| 8 | `run_sentinel_check.py` (F3 fix: per-attack random tokens) | 36 | ~$9 | ~10 min | Sentinel-randomization robustness check — see §6.15 |
| 9 | Development smoke traces | ~8 | ~$2 | negligible | Ignored in results (kept for provenance) |
| 10 | `run_gsm8k.py` (already ships in the 238 above) | 30 | included | included | Reasoning fidelity |
| **Totals** | | **~608** | **~$105** | **~2–2.5 h** | |

Run in order:

```bash
python scripts/run_final.py --dry-run                       # preview cost
python scripts/run_final.py                                 # ~35 min, ~$50

python scripts/run_baselines.py --dry-run                   # preview
python scripts/run_baselines.py                             # ~8 min, ~$9
python scripts/run_baselines.py --attacks tool_poisoning    # ~7 min, ~$7.50

python scripts/run_missing_attacks.py --dry-run
python scripts/run_missing_attacks.py                       # ~30 min, ~$9

python scripts/run_multiturn_attacks.py --dry-run
python scripts/run_multiturn_attacks.py                     # ~10 min, ~$9

python scripts/run_judges.py                                # ~10 min, ~$4  (v2 cross-family judge)

python scripts/run_sentinel_check.py                        # ~10 min, ~$9  (F3 sentinel randomization)
```

Every script auto-invokes `run_final.py --report-only` at the end, so `paper/deliverable/FINAL.md` is refreshed after each batch.

After the full sequence, run §2 (zero-cost) once more to generate the novelty-tier analyses (first-order local margin, argmax, Pareto, coverage matrix).

---

## 4. What is deterministic vs stochastic

| Component | Determinism |
|---|---|
| IB gate scoring | **Deterministic** (fixed encoder, fixed phrase bank, fixed θ) — same input → same block/pass verdict |
| Sample selection | **Deterministic** (seeded by `cfg.seed = 0`; the same 30 HotpotQA IDs are reused across systems and attacks) |
| gpt-5 planner / worker / synthesizer | **Non-deterministic** (temperature=0 but Azure gpt-5 still has ε-level nondeterminism) |
| Claude verifier (5-sample MC at T=1) | **Non-deterministic** (that's the point) |
| Azure content filter | **Non-deterministic** (opaque, provider-side) |
| Attribution classification | **Deterministic** given a trace on disk |

**Reproducibility target**: your ASR / attribution counts should match this report's within ±1 count per cell (≤3 pp deviation) after a full re-run. If you see larger deviations, the most common causes are:

- Azure API version drift (`AZURE_OPENAI_API_VERSION` changed → different filter behavior)
- Anthropic model deprecation (a Sonnet version was retired → judgment shift)
- HotpotQA dataset revision (unlikely; pinned to distractor validation)
- Phrase-bank size mismatch (`ib.phrase_bank_size` in [config.yaml](config.yaml))

---

## 5. Configuration reference

The **single source of truth** for every knob is [config.yaml](config.yaml). No hyperparameters are hard-coded in source. Key knobs and their defaults:

```yaml
seed: 0

ib:
  beta: 1.0
  threshold: 0.60           # IB gate BLOCK cutoff on max cosine similarity
  compress_ratio: 0.70      # COMPRESS band = [0.70 * threshold, threshold)
  compress_sentences: 2
  phrase_bank_size: 20      # 5 / 20 / 50 supported
  active_layers: [0, 1, 2, 3, 4, 5]

verifier:
  unsafe_threshold: 0.5
  uncertainty_threshold: 0.3
  consensus_collapse_std: 0.15
  samples: 5                # 5-sample Monte Carlo per worker output

pipeline:
  n_workers: 3
  parallel_samples: 8       # Thread-pool size per (system, attack) cell
```

Overrides at command time: any script accepts `--config path/to/other.yaml`.

---

## 6. Directory layout of the artifact

```
infoshield/
├── config.yaml                      # ← single source of truth
├── .env                             # ← API keys, git-ignored
├── INFOSHIELD.md                    # ← 1,252-line research write-up
├── README.md                        # ← project entry point
├── REPRODUCIBILITY.md               # ← this file
│
├── src/infoshield/                  # Library
│   ├── attacks/                     # 9 attack families (including memory_multiturn.py)
│   ├── baselines/                   # ibprotector, llm_guard, perplexity_filter, infoshield, ...
│   ├── ib/                          # IB gate layer + phrase bank
│   ├── llm/                         # Azure OpenAI + Anthropic wrappers
│   ├── memory/                      # MemoryController (with IB-3-mem rescore)
│   ├── metrics/                     # ASR, CTCA, UPS, ILR, TMR, CCR, statistical, llm_eval, ...
│   ├── pipeline/                    # LangGraph pipeline assembly
│   └── tools/                       # Mock tool
│
├── scripts/
│   ├── run_final.py                 # Main orchestrator + FINAL.md writer
│   ├── run_baselines.py             # Adds 3 competitive baselines
│   ├── run_missing_attacks.py       # memory/reasoning_state/consensus
│   ├── run_multiturn_attacks.py     # 3 stateful-memory attacks (single-turn simulation)
│   ├── run_judges.py                # LLM-judge Cohen's κ
│   ├── analyze_gates.py             # ROC/AUC + threshold sensitivity (free)
│   ├── analyze_certified_radius.py  # First-order local margin in embedding space (free)
│   ├── analyze_argmax_phrases.py    # Which phrase caught what (free)
│   ├── analyze_gate_pareto.py       # Cost-conditional Pareto frontier (free)
│   ├── aggregate_results.py         # results/ → paper/numbers.json + CSVs
│   └── consolidate.py               # Aggregator + all figure scripts
│   └── run_sentinel_check.py        # F3 sentinel-randomization runner (~$9)
│
├── figures/                         # matplotlib scripts (fig1..fig8)
│
├── results/                         # git-ignored — 605 trace JSONs
│   └── main/<system>/<attack>/seed_0/<sample_id>.json
│
├── paper/
│   └── deliverable/                 # ← every published number lives here
│       ├── FINAL.md                 # Auto-generated headline tables
│       ├── numbers.json             # Machine-readable
│       ├── judges_report.json       # LLM-judge raw results
│       ├── traces_index.json        # Per-cell coverage
│       ├── progress.jsonl           # One line per sample event
│       └── figures/                 # 12 PDFs + matching CSVs (PDFs git-ignored)
│
└── tests/                           # 56 offline + 8 network tests
```

---

## 7. Reproducing a specific claim

Each numbered claim in [INFOSHIELD.md §6.2](INFOSHIELD.md) is traceable to specific commands:

| Claim | Regenerate with |
|---|---|
| Main ASR table (§6.1) | `python scripts/run_final.py --report-only` → `paper/deliverable/FINAL.md` section 3 |
| Attribution table (§6.2) | Same command → `paper/deliverable/FINAL.md` section 4 |
| **Mechanism substitution Ⅰ** — tool_poisoning (§6.2 claim #2) | `grep tool_poisoning paper/deliverable/FINAL.md` |
| **Mechanism substitution Ⅱ** — memory_poisoning (§6.2 claim #7) | `grep memory_poisoning paper/deliverable/FINAL.md` |
| **Mechanism substitution Ⅲ** — mem_role_confusion (§6.14) | `grep mem_role_confusion paper/deliverable/FINAL.md` |
| GSM8K CTCA delta (§6.3) | `python scripts/aggregate_results.py` → `paper/numbers.json["gsm8k"]` |
| Per-gate AUC (§6.7) | `python scripts/analyze_gates.py` → `paper/numbers.json["roc_auc"]` |
| LLM-judge cross-family κ (§6.8) | `cat paper/deliverable/judges_report.json` (regenerate: `python scripts/run_judges.py`, ~$4) |
| First-order margin (§6.10) | `python scripts/analyze_certified_radius.py` → `certified_radius_per_gate.csv` |
| Argmax phrases (§6.11) | `python scripts/analyze_argmax_phrases.py` → `argmax_phrase_heatmap.csv` |
| Gate Pareto (§6.12) | `python scripts/analyze_gate_pareto.py` → `gate_pareto.csv` |
| Coverage matrix (§6.13) | `python figures/fig8_coverage_matrix.py` → `fig8_coverage_matrix.pdf` |
| Sentinel-randomization check (§6.15) | `python scripts/run_sentinel_check.py` → `sentinel_check_report.json` (~$9) |

---

## 8. Cost calibration on your own account

Actual per-run cost varies with provider pricing. Empirically at 2026-Q3 Azure gpt-5 + Claude Sonnet pricing:

- Mean tokens per run: **~4,082** (planner 500 + workers 3×800 + verifier 5×90 + synth 700 ≈ 4,000)
- Cost per run: **~$0.25**
- Verifier is **29.8 %** of total tokens

To calibrate on your account without spending real money on the full matrix, use:

```bash
python scripts/run_final.py --dry-run --n 3
# Prints estimated run count × $0.25 per run
```

Then run one small cell to measure your real per-run cost:

```bash
python scripts/run_final.py --n 3 --systems undefended --attacks none
# Check the resulting trace JSON's llm_calls[].{prompt_tokens,completion_tokens}
```

Multiply your per-run number by 536 to project full-matrix cost.

---

## 9. Common problems and fixes

| Symptom | Cause | Fix |
|---|---|---|
| `pytest -m "not network"` shows fewer than 56 passed | Missing dev deps | `uv pip install -e ".[dev]"` |
| Azure returns HTTP 401 | `AZURE_OPENAI_API_KEY` wrong or endpoint mismatch | Check `.env`; regenerate key in Azure portal |
| Azure returns HTTP 404 on `claude-opus-4-20250514` | Model deprecated on your endpoint | Change `models.judge_hallucination.model` in [config.yaml](config.yaml) to `claude-sonnet-4-5-20250929` — this is the current workaround, disclosed in [INFOSHIELD.md §6.8](INFOSHIELD.md) |
| `analyze_gates.py` crashes with `AttributeError: module 'numpy' has no attribute 'trapz'` | NumPy 2.x removed `np.trapz` | Already handled: `analyze_gates.py` uses `getattr(np, "trapezoid", None) or np.trapz`. If you see this, `git pull` |
| `run_baselines.py` hangs on first run | HuggingFace downloading gpt-2 (~500 MB) for `perplexity_filter` | Wait 1–2 minutes; subsequent runs use the local cache at `~/.cache/huggingface/hub/models--gpt2` |
| Massive spend spike | Someone bumped `pipeline.parallel_samples` and `--n` at the same time | Use `--dry-run` religiously; run one small `--n 3` cell first |
| Tests pass but ASR differs from paper | Azure API version drift or seed drift | Diff your `config.yaml` against `git show origin/main:config.yaml`; check `AZURE_OPENAI_API_VERSION` |

---

## 10. What is **not** reproducible from this repo

Honest disclosure:

- **The Azure content filter's exact behavior.** Provider filters are opaque and update over time. If Azure changes what it catches, some `azure_filter`-attributed traces would shift to `verifier_unsafe` or `leaked` under a re-run. The paper's arguments (attribution structure, INFOSHIELD's IB blocks) are robust to this; the absolute ASR numbers for `undefended` may drift by a few percentage points.
- **The specific gpt-5 completions.** gpt-5 output varies run-to-run even at `temperature=0` (Azure's server-side sampler is not fully deterministic). Aggregate metrics are stable; individual trace outputs are not.
- **Prices.** All `~$X` figures are 2026-Q3 estimates. Multiply by your account's actual per-token cost for a current estimate.

Everything **else** — attribution logic, IB gate decisions, certified radii, argmax phrases, Pareto subsets, LLM-judge κ methodology, per-attack coverage — is deterministic given the trace JSONs on disk and reproducible bit-for-bit.

---

## 11. Contact / provenance

- **Repository**: https://github.com/Notmeher/infoshield
- **Latest commit at time of this document**: run `git log -1 --oneline` in the repo root
- **Full research write-up**: [INFOSHIELD.md](INFOSHIELD.md)
- **Auto-generated headline numbers**: [paper/deliverable/FINAL.md](paper/deliverable/FINAL.md)

If any command in this document fails on your machine, please open an issue with the exact command, your OS, Python version, and the full traceback.
