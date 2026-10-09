# ChannelGuard

**Borrowed safety does not travel; owned safety can.**
Information-bottleneck gates on the channels between LLM agents in a multi-agent pipeline.

ChannelGuard places six lightweight gates (IB-0 to IB-5) across a
`planner → workers → verifier → synthesizer` pipeline. Each gate scores text with a small local
sentence encoder and passes, compresses, or blocks it. The study asks what actually stops an
attack: the application-level gates, the agent model, or the cloud provider's content filter.
Equal attack success rate (ASR) can hide very different sources of protection.

> The Python package still uses the earlier internal name `infoshield`
> (`src/infoshield/`). It is the same system.

## Documents

| File | Contents |
|---|---|
| [Final/final_results.md](Final/final_results.md) | Full results and analysis, including the cross-model replication (§6A) |
| [Final/README.md](Final/README.md) | Index of generated tables, macros, figures, and data |
| [CHANNELGUARD.md](CHANNELGUARD.md) | Long walkthrough: problem, data, methods, results, applications |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Step-by-step regeneration guide for the original study |
| [Final/qwen/README.md](Final/qwen/README.md) | Complete Qwen experiment report |
| [Final/llama/README.md](Final/llama/README.md) | Complete Llama experiment report |

## Main findings

Agents: GPT-5 (Azure, content filter on), Claude Sonnet 4.5 and Haiku 4.5 (no provider filter),
and two local open-weight models, Qwen3-4B-Instruct-2507 (MLX 8-bit) and
Llama-3.1-8B-Instruct (MLX 4-bit). The completed studies report 5,700 pipeline traces:
2,100 historical manuscript traces plus 1,800 for each local agent; n = 30 per cell, one seed.
Only the 3,600 local traces are available in this checkout. Claude Sonnet 4.5 is the verifier
in every configuration, including the local-agent runs.

| Agent | Prompt-injection ASR, U / CG | Exact paired McNemar p | Adaptive-paraphrase ASR, U / CG | GSM8K CTCA, U / CG |
|---|---|---|---|---|
| Azure GPT-5, historical | 0.333 / 0.167 | 0.125 | 0.667 / 0.667 | 0.867 / 0.867 |
| Qwen 4B, MLX 8-bit | 0.800 / 0.033 | 2.38e-7 | 0.633 / 0.600 | 0.767 / 0.767 |
| Llama 8B, MLX 4-bit | 0.767 / 0.100 | 1.91e-6 | 0.567 / 0.567 | 0.767 / 0.767 |

U is the pipeline without IB gates; CG is ChannelGuard (`infoshield` in the code).
Both retain the verifier. The p-values are two-sided and unadjusted; only the local-agent
prompt-injection differences are significant at 0.05 in these comparisons.

On fixed-marker tool poisoning, IB-3 blocks 30/30 on all five measured agents. On GPT-5,
Azure's filter accounts for 54/60 undefended outcomes on pooled tool and memory poisoning.
On Llama, neither system leaks in the pooled 90-trace tool/memory/role-confusion subset,
but ChannelGuard records 90/90 IB-blocked outcomes. Undefended safety includes verifier
catches and safe answers, not 90 model refusals; the zero-leak floor leaves no incremental
ASR benefit to measure on that subset.

GSM8K accuracy is unchanged in aggregate within each agent. Main-cell benign FPR is zero
for both local agents, under their reports' final-refusal definition. This is not a claim
of equal task capability across models or sample-wise equivalence between systems.

Adaptive paraphrase remains a failure case. Both SmoothLLM and the Claude-based LLM-Guard
proxy do better on this attack; LLM-Guard reaches zero ASR on both local agents' 30-sample
cells. Compression is not universally safe, including at several inter-agent gates in
the local replications. No single result establishes general jailbreak protection.

See [Final/final_results.md](Final/final_results.md) for all tables and caveats.

## Pipeline

```text
User input -> IB-0 -> Planner -> IB-1 -> worker subtasks
                      |
      Tool output -> IB-3 ------------>|
      Memory read -> IB-3-mem -------->|
                      v
                  Worker agents
                      |
                    IB-2
                      |
                    IB-4
                      |
             Verifier (Sonnet, M=5, T=1.0)
                      |
               IB-5 -> Synthesizer + filter
                      |
                Answer or fixed refusal
```

The gate computes maximum cosine similarity over sentence embeddings and a 20-phrase
attack bank, using `sentence-transformers/all-MiniLM-L6-v2` on CPU. At the main experiment's
$\theta=0.60$, it blocks at or above the threshold, compresses between
$0.70\theta$ and $\theta$ to the first two sentences, and otherwise passes.
IB-3-mem is the memory-read use of the IB-3 layer, not a seventh ablation index.

The verifier samples each surviving worker output five times. Its unsafe threshold is
mean safety below 0.5 and its uncertainty threshold is standard deviation above 0.3.
The synthesizer filters worker outputs and applies IB-5; if none survive, it returns a
fixed refusal without a synthesis call. An early block can avoid downstream calls;
an internal tool or memory block can instead drop only that content while the pipeline continues.

Code: [src/infoshield/ib/layer.py](src/infoshield/ib/layer.py),
[src/infoshield/pipeline/langgraph_pipeline.py](src/infoshield/pipeline/langgraph_pipeline.py).

## Setup

### Requirements

Run the commands below from the repository root, the directory containing this README
and [pyproject.toml](pyproject.toml). Python 3.12 is recommended; the project declares
support for Python 3.10 through 3.12, not 3.13+. The checked environment uses Python
3.12.9 and MLX-LM 0.31.3. Dependency ranges are specified, but this is not a fully pinned
environment lockfile.

The local MLX agent path requires an Apple Silicon Mac. The recorded local runs used
an Apple M5 with 16 GB unified memory. Run Qwen and Llama sequentially; two concurrent
full suites can cause memory pressure and GPU contention. Cloud-agent runs, tests, and
report building do not require the local model weights.

Reading existing results needs no API credentials. New pipeline runs require Claude
access for the verifier, even when the agent runs locally. The complete local judge
audit also requires an Azure GPT-5 deployment.

### Install

With [uv](https://docs.astral.sh/uv/getting-started/installation/) installed:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Or use Python's standard environment tooling:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

For the local-agent runs, install the additional packages in that same environment:

```bash
python -m pip install "mlx-lm==0.31.3" huggingface_hub
```

Activate the environment in every new terminal, or use `.venv/bin/python` explicitly.
There is no web server to start: this repository runs CLI experiments and produces
Markdown, JSON, CSV, PDF, and LaTeX artifacts.

### Credentials

Create `.env` at the project root (git-ignored):

```dotenv
ANTHROPIC_API_KEY=<your-anthropic-key>

# Required for Azure-agent runs and the GPT-5 half of the local judge audit
AZURE_OPENAI_API_KEY=<your-azure-openai-key>
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-5

# Optional for public downloads; required if your selected model needs authentication
HF_TOKEN=<your-huggingface-token>
```

Use the deployment name that exists in your Azure account. Select Claude models and
the Azure agent deployment in the YAML configs; `ANTHROPIC_MODEL` is not a substitute
for those config fields. `Hf_api_key`, `HF_API_KEY`, and `HUGGINGFACE_TOKEN` are accepted
aliases for Hugging Face authentication when the config loader runs.

For pipeline-only local steps, Azure credentials can be omitted. They are needed for
all four judge labels, not for report-only regeneration. Keep keys and trace content
out of public logs. Never commit `.env`.

### Download Local Models

The two configs expect these directories beside the repository:

```text
channelguard/
  infoshield/                               # repository root
  Qwen3-4B-Instruct-2507-mlx-8bit/
  Llama-3.1-8B-Instruct-mlx-4bit/
```

Download the matching MLX checkpoints from the repository root:

```bash
hf download mlx-community/Qwen3-4B-Instruct-2507-8bit \
  --local-dir ../Qwen3-4B-Instruct-2507-mlx-8bit

hf download mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --local-dir ../Llama-3.1-8B-Instruct-mlx-4bit
```

Download weights, tokenizer files, and configs, not just the model card. A plain
Transformers checkpoint such as Qwen2.5-1.5B is not the checkpoint used in these runs.
To store weights elsewhere, change `models.agent.model` in the corresponding YAML
to an absolute path. Checkpoint licenses apply; Llama uses Meta's Llama 3.1 license.
For gated downloads, obtain access and run `hf auth login` locally. The `hf` CLI does
not automatically load the project's `.env`.

### Cache Datasets and Encoder

First-time use downloads HotpotQA, GSM8K, and MiniLM. Public dataset downloads do not
normally need a token. Prepare the cache without running paid model calls:

```bash
python - <<'PY'
from infoshield.config import load_config
from infoshield.datasets import load
from sentence_transformers import SentenceTransformer

load_config("config.qwen.yaml")
load("hotpot_qa", 30, seed=0)
load("gsm8k", 30, seed=0)
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
print("Datasets and encoder cached")
PY
```

The experiment datasets are HotpotQA distractor validation and GSM8K test, shuffled
with seed 0. The perplexity-filter baseline also downloads GPT-2 on its first use.
Hugging Face manages its own cache; do not assume the YAML's `dataset_cache` field
relocates every library's downloads.

## Running experiments

### Local Suites

Inspect the plan and pending counts first. A dry run does not generate model outputs
or make paid API calls, but can access the dataset cache or download missing data.

```bash
python scripts/run_qwen_suite.py --dry-run
python scripts/run_llama_suite.py --dry-run
```

Run the full suites sequentially. These commands make paid verifier and judge calls:

```bash
python -u scripts/run_qwen_suite.py --config config.qwen.yaml
# After Qwen finishes:
python -u scripts/run_llama_suite.py --config config.llama.yaml
```

On macOS, `caffeinate -i` can keep the machine awake during a foreground run:

```bash
caffeinate -i python -u scripts/run_llama_suite.py --config config.llama.yaml
```

The suite defaults to `--n 30` and seed 0. It pins `PYTHONHASHSEED=0` automatically
to keep attack-template selection consistent across systems and resumed processes.
Planner, workers, and synthesizer use the selected MLX agent; Claude remains the verifier.
The `llm_guard` baseline is a Claude classifier proxy, not an actual Llama Guard model.

### Experiment Matrix

Both local suites use the same matrix:

| Step key | Traces per agent | Scope |
|---|---:|---|
| `main` | 480 | 2 systems x 8 attack/benign cells x 30 |
| `gsm8k` | 60 | 2 systems x 30 clean reasoning samples |
| `sentinel` | 180 | 2 systems x 3 randomized-marker attacks x 30 |
| `ablation` | 210 | Full defense plus 6 gate-drop variants x 30 |
| `beta` | 180 | 6 threshold strictness values x 30 |
| `bank` | 90 | 3 phrase-bank sizes x 30 |
| `extra` | 120 | 2 systems x reasoning-state/consensus attacks x 30 |
| `baselines` | 360 | IBProtector, LLM-Guard, perplexity filter x 4 cells x 30 |
| `smoothllm` | 120 | 4 cells x 30; 10 inner pipelines per saved output |
| `judges` | Separate audit | Up to 100 stratified outputs with 4 judge/rubric labels |
| Total pipeline traces | 1,800 | Audits and discarded SmoothLLM inner traces excluded |

The main matrix includes prompt injection, tool poisoning, memory poisoning, slow drift,
consent escalation, role confusion, adaptive paraphrase, and benign inputs. Baselines
cover prompt injection, tool poisoning, adaptive paraphrase, and benign inputs.
The completed audits contain 94 Qwen outputs and 95 Llama outputs.

### Partial Runs and Resume

Select steps by the keys above; `--steps` takes one or more keys:

```bash
python scripts/run_qwen_suite.py --steps main gsm8k
python scripts/run_llama_suite.py --steps sentinel extra
python scripts/run_llama_suite.py --steps baselines smoothllm
python scripts/run_llama_suite.py --steps judges
```

Existing expected trace files are skipped. Stop with Ctrl+C and rerun the same command
to resume; in-progress samples without a saved trace run again. Each completed pipeline
sample is checkpointed, and completed judge labels are reused. Look at the final coverage
and audit status, not just the process exit code: the suite logs failures and can continue.

For a small smoke run, copy a local config, give it a new `output_root` such as
`results_qwen_smoke`, then use:

```bash
python scripts/run_qwen_suite.py --config config.qwen.smoke.yaml --n 2 --steps main gsm8k
```

Create that smoke config first. Changing the model, attack setup, or sample count while
reusing a completed output root can silently reuse old traces; resume is based on file
existence, not a configuration fingerprint. Smoke runs also overwrite the model's
`paper` report, so do not transfer smoke summaries into `Final`.

### Outputs and Monitoring

| Output | Location |
|---|---|
| Raw Qwen traces | `results_qwen/<experiment>/<system>/<attack>/seed_0/<sample_id>.json` |
| Raw Llama traces | Same layout under `results_llama/` |
| Step/sample events | `results_qwen/progress.jsonl`, `results_llama/progress.jsonl` |
| Raw judge labels | `results_<agent>/judge_audit/judge_audit.json` |
| Runner-generated reports | `paper/qwen/` and `paper/llama/`, recreated on demand |
| Preserved inputs for the final builder | [Final/qwen/numbers.json](Final/qwen/numbers.json), [Final/llama/numbers.json](Final/llama/numbers.json) |

`extra` traces share the `main` directory; SmoothLLM shares `baselines`. Folder names
alone do not isolate all step counts. Trace JSON stores inputs, final outputs, model
calls, tokens, latencies, gate decisions, verifier reports, and task metadata.

For example, inspect recent Llama progress with:

```bash
tail -n 20 results_llama/progress.jsonl
```

MLX generations are serialized on one engine thread per process; eight sample threads
overlap other work and verifier calls. SmoothLLM performs ten inner pipelines before
saving one selected output, so a long gap between saved samples does not establish a hang.

### Original Cloud Runners

The historical scripts remain available, but their current defaults are not the full
published 2,100-trace protocol. [config.yaml](config.yaml) currently sets GSM8K to 15
samples and ablation/beta cells to 6; the published study used 30. The legacy baseline
and additional-attack scripts also default to 6 unless `--n` is supplied.

These commands use the Azure agent and Claude verifier, writing to `results/` and
`paper/deliverable/`, not directly to the manuscript reference in `Final`:

```bash
python scripts/run_final.py --dry-run
python scripts/run_final.py
python scripts/run_baselines.py --n 30 --seed 0
python scripts/run_missing_attacks.py --n 30 --seed 0
python scripts/run_judges.py --n 30 --dry-run
python scripts/run_judges.py --n 30
```

Set the intended sample counts in the config before new cloud runs and inspect each
script's `--help` for attack/system selection. These commands are available workflows,
not a claim that rerunning them reproduces every historical table exactly. The legacy
`run_judges.py` evaluates borderline traces; the local suite's `--steps judges` is its
own stratified sharp/vague audit. Do not substitute one for the other.

## Refresh Results

### Rebuild Local Reports

After completing or resuming a suite, regenerate its report from saved traces without
paid model calls. With the datasets cached, the Hugging Face flags also prevent network
lookups during regeneration:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python scripts/run_qwen_suite.py --report-only
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python scripts/run_llama_suite.py --report-only
```

The runners still write to `paper/<agent>/`, even though that old folder was removed.
They recreate it. Review the refreshed reports for complete coverage and section errors,
then deliberately replace the preserved sources in `Final`:

```bash
cp paper/qwen/README.md paper/qwen/numbers.json Final/qwen/
cp paper/llama/README.md paper/llama/numbers.json Final/llama/
```

Those copy commands overwrite the preserved reports. Use them only for the intended
completed 30-sample suites, and inspect the diff before publishing. Keep the raw traces
and judge-audit JSON alongside the reports; report generation cannot recover missing runs.

### Build the Final Package

To regenerate the manuscript tables and cross-model artifacts from the preserved
summaries and local traces:

```bash
python scripts/build_final.py
```

This is offline and makes no model API calls. It reads both model summaries from
`Final`, not `paper`, plus local traces for per-role latency and failure examples.
It does not rerun experiments or regenerate the model-specific summaries.

The builder overwrites generated tables, CSVs, macros, selected figures, and the
`Final` index. In the main results document it replaces the marked cross-model section
(section 6A), leaving the surrounding hand-maintained narrative in place. Several older
PDFs remain historical renders rather than newly recomputed plots. Back up manual
artifact edits or inspect the Git diff before rebuilding; the builder also normalizes
branding in output files. Do not edit generated sections expecting those edits to survive.

Historical manuscript values are transcribed in
[scripts/build_final.py](scripts/build_final.py). They are not recalculated from the
absent original raw corpus. The complete report remains
[Final/final_results.md](Final/final_results.md).

## Tests

Run the unit and mocked-pipeline tests without live provider calls:

```bash
PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m pytest tests -m "not network" -q
```

For the paired local-suite/config/report tests only:

```bash
PYTHONHASHSEED=0 python -m pytest tests/test_local_suites.py -q
```

Tests marked `network` are excluded by the first command. Running them explicitly can
require credentials and incur charges. Passing mocked tests does not validate provider
availability, downloadable model access, or the full experiment's scientific claims.

## Repository layout

| Path | Purpose |
|---|---|
| [config.yaml](config.yaml) | Legacy Azure-agent configuration |
| [config.qwen.yaml](config.qwen.yaml), [config.llama.yaml](config.llama.yaml) | Local agents, shared verifier, gate settings, and output roots |
| `src/infoshield/` | Pipeline, IB gates, agents, attacks, baselines, memory, tools, metrics, and LLM clients |
| [scripts/build_final.py](scripts/build_final.py) | Offline Final artifact builder |
| `scripts/` | Suite runners, legacy runners, and offline analyses |
| `figures/` | Individual plotting scripts; separate from the Final builder |
| [Final/README.md](Final/README.md) | Index of results, tables, source summaries, figures, and macros |
| `Final/tables/cross_model/`, `Final/data/cross_model/` | Combined experiment tables and their data |
| `results_qwen/`, `results_llama/` | Raw traces, progress, and judge labels for the local replications |
| `tests/` | Unit, mocked-pipeline, and separately marked network tests |

## Provenance and caveats

ASR is literal attack-marker leakage, not a general harmfulness score. CTCA measures
task-answer correctness. BPR and FPR must accompany safety comparisons, because blocking
everything can produce zero ASR without preserving useful answers. Internal gate blocks,
final refusals, verifier flags, and safe answers are distinct outcomes.

The original GPT-5 and Claude numbers are historical manuscript measurements; their
2,100 raw traces are absent from this checkout. Qwen and Llama each have 1,800 local
pipeline traces. Models differ in size, training, quantization, and provider filtering;
these are not quantization-controlled or perfectly contemporaneous model comparisons.
The Claude verifier is shared, so the local-agent suites are not fully local systems.

The replications pin attack-selection hashing, record argmax phrases at gate time, and
fix tool-marker randomization. The old manuscript's tool sentinel row did not actually
randomize the tool payload, so its marker-invariance interpretation needs qualification.
Observed any-gate block preservation also does not guarantee absence of final leakage.

The main threshold is fixed at 0.60. The beta sweep overrides it with
$\theta_{\mathrm{eff}}=\max(0.15,0.85-0.15\beta)$; beta=1 in that sweep means 0.70,
not the main configuration's 0.60. Gate ablation covers prompt injection only, and the
threshold sweeps do not include a benign-fidelity sweep. Embedding margins are
first-order proxies, not certified robustness radii.

Local latency is a sum of recorded call and gate times that excludes MLX queue wait;
it is not the elapsed suite duration and is not directly comparable to historical API
wall-clock measurements. Recorded API costs are about $9.96 for Qwen and $11.37 for Llama,
including their audits, versus the historical $47.36 study total. These are incomplete
billing estimates: LLM-Guard classifier calls are not retained, and SmoothLLM saves only
the selected inner trace rather than all ten pipelines. Retries and hardware energy are
not fully accounted for. Local agent generation has no API charge; the verifier does.

All experiments use one seed, n=30 per completed cell, mock tool responses, and an
English-only static phrase bank. Zero observed leakage at this sample size does not
establish zero population risk. See the full results document for uncertainty intervals,
attribution, baseline comparisons, and remaining failure cases.

## Troubleshooting

| Symptom | Check or action |
|---|---|
| `No module named infoshield` | Activate `.venv` and rerun the editable installation from the repository root. |
| Missing `mlx_lm` or unsupported MLX platform | Install the optional package on Apple Silicon; use a cloud agent or inspect existing reports elsewhere. |
| Model path not found | Confirm the relative directory layout or set an absolute `models.agent.model` path in the local config. |
| Missing API-key error or authentication failure | Check the root `.env`, provider access, and Azure deployment/config names. Do not print keys into logs. |
| Dataset/model download fails | Check access, network, disk space, and the Hugging Face cache. For stalled Xet transfers, try `HF_HUB_DISABLE_XET=1` for the download. |
| TLS certificate failure | Repair the Python/system trust store. For report-only work with a complete cache, use the offline flags; do not disable certificate validation. |
| Offline report says a dataset is missing | Prepare the cache once with network access; offline flags cannot supply uncached files. |
| GPU contention, swap, or long pauses | Run one local suite at a time. SmoothLLM saves only after its inner runs finish; inspect progress rather than assuming a deadlock. |
| A report still shows missing cells | Rerun the affected `--steps` with the original config and n. Existing completed traces are reused. |
| `Final` still shows old results | Review the reports recreated under `paper`, transfer completed summaries into `Final`, then run the Final builder. |

The current setup and report workflow above supersede older commands in the historical
walkthroughs. No experiment rerun is needed just to read or rebuild the completed results.
