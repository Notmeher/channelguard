# INFOSHIELD

**Safety-constrained information flow for trustworthy multi-agent LLM systems.**

Six information-bottleneck (IB) gates wired across a
`planner → workers → verifier → synthesizer` pipeline, tested against
prompt-injection, tool-poisoning, and adaptive white-box paraphrase attacks.

For the full research write-up (motivation, design, results, discussion) see
[paper/deliverable/FINAL.md](paper/deliverable/FINAL.md).

## Quickstart

```bash
# Create the venv (Python 3.10-3.12) and install
uv venv --python 3.12
source .venv/bin/activate         # Windows: .venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"

# Offline regression tests (~15s, $0)
python -m pytest -m "not network"

# Full suite incl. real Azure OpenAI + Anthropic pings (~30s, ~$0.02)
python -m pytest --run-network
```

Fill in `.env` at project root:

```dotenv
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-5

AZURE_OPENAI_EMBED_API_KEY=...
AZURE_OPENAI_EMBED_API_VERSION=2024-12-01-preview
AZURE_OPENAI_EMBED_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-large
AZURE_OPENAI_EMBED_DIMENSIONS=1536

ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929

Hf_api_key=hf_...                 # for HuggingFace dataset + MiniLM downloads
```

## Running the experiments

The main entry point is `scripts/run_final.py`; four supporting scripts
extend the matrix along different axes.

```bash
# See what would run + estimated cost (no API calls)
python scripts/run_final.py --dry-run

# Run all main-experiment steps (resumes any completed cells from disk)
python scripts/run_final.py

# Only regenerate the report from traces already on disk
python scripts/run_final.py --report-only

# Optional add-ons (each auto-refreshes FINAL.md at the end):
python scripts/run_baselines.py         # ibprotector + llm_guard + perplexity_filter (~\$9 at n=6)
python scripts/run_missing_attacks.py   # memory + reasoning_state + consensus (~\$9 at n=6)
python scripts/run_judges.py            # LLM-judge Cohen's \u03ba on borderline traces (~\$4 for n=30)
python scripts/analyze_gates.py         # ROC/AUC + threshold sweep (free, disk-only)
```

Wall-clock: **~20 min** for `run_final.py` at `pipeline.parallel_samples=8`.
Cumulative spend to produce the current 535-trace artifact: **~\$88**.

Output goes to [paper/deliverable/FINAL.md](paper/deliverable/FINAL.md).

## Repository layout

| Path | Purpose |
|---|---|
| [config.yaml](config.yaml) | Single source of truth: models, β, θ, seeds, sample counts |
| `.env` | API keys (git-ignored) |
| `src/infoshield/` | Library code — pipeline, IB layer, agents, attacks, baselines, metrics |
| [scripts/run_final.py](scripts/run_final.py) | Main experiment orchestrator + FINAL.md writer |
| [scripts/run_baselines.py](scripts/run_baselines.py) | Adds `ibprotector` / `llm_guard` / `perplexity_filter` rows to the main table |
| [scripts/run_missing_attacks.py](scripts/run_missing_attacks.py) | Runs `memory_poisoning` / `reasoning_state_injection` / `consensus_collapse` |
| [scripts/run_judges.py](scripts/run_judges.py) | LLM-judge eval: hallucination + consistency + two-judge Cohen's κ |
| [scripts/analyze_gates.py](scripts/analyze_gates.py) | ROC/AUC per gate + threshold sensitivity (from existing traces, \$0) |
| [scripts/aggregate_results.py](scripts/aggregate_results.py) | Walks `results/` → paper artifacts |
| [scripts/consolidate.py](scripts/consolidate.py) | Aggregator + figures → `paper/deliverable/` |
| `figures/` | matplotlib scripts consuming `paper/figures/*.csv`; `fig7_sankey_attribution.py` renders the attribution flow |
| `results/` | One JSON per pipeline sample (git-ignored, regenerable) |
| `tests/` | 56 offline + 8 network tests |
| [INFOSHIELD.md](INFOSHIELD.md) | ~1070-line research write-up (single-doc paper draft) |
| [paper/deliverable/FINAL.md](paper/deliverable/FINAL.md) | Auto-generated headline tables — single source of truth for every number |

## Pipeline architecture

```
Attack input
    ↓
IB-0  ─── input gate ─────────────────┐
    ↓                                  │
Planner (gpt-5)                        │
    ↓ (subtasks)                       │
IB-1  ─── planner→subtask gate ───────┤
    ↓                                  │
Workers (gpt-5, N in parallel)         │  ← MemoryController
    ↓                                  │    with IB re-score
IB-3  ─── tool output gate ─────      │
IB-2  ─── worker output gate ─────    │
    ↓                                  │
IB-4  ─── pre-verifier gate ──────    │
    ↓                                  │
Verifier (Claude Sonnet-4.5, 5-MC)    │
    ↓                                  │
IB-5  ─── joined output gate ─────    │
    ↓                                  │
Synthesizer (gpt-5) + triple filter    │
    ↓                                  │
Final answer  OR  refusal ─────────────┘
```

- **IB gate mechanism**: per-sentence cosine similarity against a 20-phrase
  adversarial bank (default; `phrase_bank_size` in [config.yaml](config.yaml)).
  `risk ≥ threshold` → BLOCK; `compress_ratio * threshold ≤ risk < threshold`
  → COMPRESS (first N sentences); else PASS.
- **Verifier**: 5-sample Monte Carlo scoring at temperature=1.0. Flags unsafe
  (`mean < 0.5`) and uncertain (`std > 0.3`).
- **Synthesizer triple filter**: worker output must satisfy IB-5 gate AND
  verifier `mean_safety ≥ threshold` AND not (unsafe AND uncertain).

See [src/infoshield/ib/layer.py](src/infoshield/ib/layer.py) and
[src/infoshield/pipeline/langgraph_pipeline.py](src/infoshield/pipeline/langgraph_pipeline.py).

## Model roster

| Component | Model |
|---|---|
| Planner / Workers / Synthesizer | `gpt-5` (Azure OpenAI) |
| Verifier (5-sample MC) | `claude-sonnet-4-5-20250929` |
| IB gate embedding | `all-MiniLM-L6-v2` (local, default) or Azure `text-embedding-3-large` |

## Design invariants

1. Every module is Pydantic-in / Pydantic-out — no untyped dicts across boundaries.
2. Every decision writes one record to a `Trace`
   ([src/infoshield/trace.py](src/infoshield/trace.py)).
3. Every runner writes exactly one JSON per sample under
   `results/<experiment>/<system>/<attack>/seed_<n>/<sample_id>.json`.
4. `scripts/aggregate_results.py` is the only path from `results/` to `paper/`.
5. `config.yaml` is the only source of truth — no hard-coded knobs in source.

## Results

See [paper/deliverable/FINAL.md](paper/deliverable/FINAL.md) for the full
auto-generated report, and [INFOSHIELD.md](INFOSHIELD.md) for the full
research write-up (motivation, design, related work, discussion, ethics).

**One-line summary** (over **535 traces**, ~\$88 total spend, 9 attack
families, 5 systems): INFOSHIELD reduces `prompt_injection_strong` ASR by
33 % (0.300 → 0.200) and provides **two uncontested deterministic wins** —
it blocks **30/30 `tool_poisoning`** attempts at IB-3 and **6/6
`memory_poisoning`** attempts at IB-3-mem, whereas undefended relies on
Azure's stochastic filter for both. GSM8K reasoning accuracy is preserved
(+6.7 pp CTCA), and INFOSHIELD runs **~7× faster** than undefended on
adversarial workloads because IB gates short-circuit the pipeline before
spending expensive gpt-5 / Claude calls.
