# INFOSHIELD

Safety-constrained information flow for trustworthy multi-agent LLM systems.
Six information-bottleneck gates wired across a `planner → workers → verifier → synthesizer`
pipeline; five attacks, seven baselines, two encoder backends, ten experiments.

## Quickstart

```powershell
# From project root, with .venv already created and populated
uv pip install -e ".[dev]"

# Offline tests (no API calls; ~25s)
pytest -m "not network"

# Full test suite incl. real Azure OpenAI + Anthropic + embedding pings
pytest --run-network
```

## Repository layout

| Path | Purpose |
|---|---|
| `config.yaml` | Single source of truth for models, β, θ, seeds, sample counts, encoder |
| `.env` | API keys (git-ignored) |
| `src/infoshield/` | Library code (config, llm wrappers, IB layer, agents, attacks, baselines, metrics) |
| `scripts/` | Experiment runners + `aggregate_results.py` |
| `results/` | One JSON per pipeline sample (git-ignored, regenerable) |
| `tests/` | 56 offline + 8 network tests |
| `paper/` | Auto-generated `numbers.json`, figure CSVs, and rendered PDFs |
| `figures/` | matplotlib scripts that consume paper/figures/*.csv |

## Two encoder backends

The IB gate scores incoming text against an adversarial phrase bank. Two backends:

| Backend | Config | Notes |
|---|---|---|
| **Local MiniLM** (default) | `embedding.provider: local` | free, offline, ~5 ms/gate |
| **Azure text-embedding-3-large** | `embedding.provider: azure` | sharper (blocks subtler injections), ~150 ms/gate, per-call cost |

Registered as two separate systems (`infoshield` and `infoshield_azure`), so a single
experiment run can compare them side-by-side. See `scripts/run_encoder_comparison.py`.

## Running experiments

Every runner accepts `--n_samples N --seeds 0 1 2`. All output goes to
`results/<experiment>/<system>/<attack>/seed_<n>/<sample_id>.json` and re-runs skip cells
that already exist (resume-from-checkpoint).

**Tiny smoke (validates the whole stack, ~2 min, <$1):**
```powershell
python scripts\smoke_baseline.py --n_samples 2
```

**Full plan (see cost estimate below):**
```powershell
python scripts\run_all.py                    # everything, full samples
python scripts\run_all.py --smoke            # 2 samples per experiment (~15 min, ~$5)
python scripts\run_all.py --only run_main    # subset
```

**Individual experiments:**
```powershell
python scripts\run_main.py --n_samples 100 --seeds 0 1 2
python scripts\run_encoder_comparison.py --n_samples 100 --seeds 0
python scripts\run_ablation.py --n_samples 50 --seeds 0 1 2
python scripts\run_beta_sweep.py --n_samples 50 --seed 0
python scripts\run_adaptive.py --n_samples 100
```

## Aggregating results

The **only** path from `results/` to the paper is `scripts/aggregate_results.py`:

```powershell
python scripts\aggregate_results.py
```

This walks every JSON in `results/`, re-computes every metric, and writes:

- `paper/numbers.json` — machine-readable record of every headline number
- `paper/figures/*.csv` — data files consumed by `figures/*.py`
- `paper/figures/failure_gallery.json` — 10 correct blocks / 10 false positives / 10 misses
- `paper/aggregation_summary.json` — coverage report (missing experiments, under-filled cells, trace count)

Run `scripts/consolidate.py` to bundle everything (plus rendered figures and a
human-readable `RESULTS.md`) into `paper/deliverable/`.

## Generating figures

```powershell
python figures\fig1_asr_bar.py             # main ASR bar chart
python figures\fig2_beta_pareto.py         # beta Pareto curve
python figures\fig3_ablation_heatmap.py    # per-layer ablation
python figures\fig5_gate_latency.py        # gate latency histogram
python figures\fig6_failure_gallery.py     # Markdown failure-case gallery
```

Fig 4 (architecture diagram) is drawn manually in draw.io / Excalidraw and exported as PDF.

## Experiment matrix (from `config.yaml`)

| Experiment | Systems | Attacks | Samples | Seeds | Rough cost |
|---|---|---|---|---|---|
| `main` | 6 | 5 | 300 | 3 | ~$1.5k |
| `advbench` | 6 | none | 300 | 3 | ~$500 |
| `gsm8k` | 2 | none | 300 | 3 | ~$100 |
| `injecagent` | 6 | 3 | 200 | 3 | ~$500 |
| `adaptive` | 6 | adaptive | 200 | 3 | ~$300 |
| `ablation` | 7 configs | 1 | 100 | 3 | ~$200 |
| `beta_sweep` | 6 β | 1 | 100 | 1 | ~$60 |
| `multi_framework` | 2 | 3 | 100 | 3 | ~$150 |
| `encoder_comparison` | 2 | 5 | 100 | 3 | ~$200 (embed ~$0.40; rest is agents) |
| `phrase_bank_ablation` | 3 | 1 | 100 | 3 | ~$150 |

**Total budget: ~$2.5–4.5k API cost, ~14–20 days wall-clock (8-way parallel).**

## Design invariants

1. Every module is Pydantic-in / Pydantic-out — no untyped dicts across boundaries
2. Every decision writes one record to a `Trace`
3. Every runner writes exactly one JSON per sample under
   `results/<experiment>/<system>/<attack>/seed_<n>/<sample_id>.json`
4. `scripts/aggregate_results.py` is the only path from `results/` to the paper
5. `config.yaml` is the only source of truth; no hard-coded knobs in source code

## Model roster

| Component | Model |
|---|---|
| IB gate embedding | MiniLM-L6-v2 (default) OR Azure text-embedding-3-large |
| Planner / Worker / Synthesizer | gpt-5 (Azure) |
| Uncertainty-aware Verifier (5-sample MC) | claude-sonnet-4-5 |
| Judge — hallucination | claude-opus |
| Judge — multi-step consistency | claude-sonnet-4-5 |
