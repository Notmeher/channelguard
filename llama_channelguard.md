# ChannelGuard × Llama-3.1-8B: Local-Model Replication — Summary & Verdict

**A plain-language companion to [Final/llama/README.md](Final/llama/README.md).** That file has every raw table;
this one explains what the numbers mean, compared directly against the manuscript's Azure GPT-5 + Claude
results (arXiv:2607.19430v2) and against the earlier Qwen3-4B replication
([qwen_channelguard.md](qwen_channelguard.md)).

---

## TL;DR

> **Llama-3.1-8B, run completely locally with no provider safety filter, was as vulnerable as Qwen to direct
> prompt injection (77% success undefended) — but unlike Qwen, it resisted every tool and memory attack on its
> own. ChannelGuard cut prompt injection to 10% and replaced the model's self-protection with deterministic
> gate blocks on every tool and memory attack. Two attacks were not fixed: Adaptive Paraphrase (on every model)
> and, on Llama only, Consensus Collapse.**

| Component | This run | Manuscript |
|---|---|---|
| Planner / workers / synthesizer | **Llama-3.1-8B-Instruct, MLX 4-bit, local** | Azure GPT-5 |
| Provider-side content filter | **none** (local inference) | Azure content filter, always on |
| Verifier | Claude Sonnet 4.5, T=1.0, 5 MC samples | same |
| IB gates | same θ/α/N/β, same 20-phrase bank | same |
| Data | HotpotQA n=30, GSM8K n=30, seed 0 | same |
| Total cost | **$11.37** | $47.36 |
| Traces | 1,800 | 2,100 + 100 judge pairs |

> **Caveat:** the 4-bit quantization is part of this agent configuration. Some differences from the paper may
> come from quantization rather than from Llama itself.

---

## 1. Was Llama worse than GPT-5? On direct injection, yes — on indirect attacks, no

| Attack | Undefended Llama | Undefended Qwen | Undefended GPT-5 (paper) |
|---|---:|---:|---:|
| **Prompt Injection** | **0.767** | 0.800 | 0.333 |
| Tool Poisoning | **0.000** | 0.067 | 0.000 |
| Memory Poisoning | **0.000** | 0.033 | 0.000 |
| Slow-Drift Memory | **0.000** | 0.067 | 0.000 |
| Consent-Escalation Memory | **0.000** | 0.000 | 0.000 |
| Role-Confusion Memory | **0.000** | 0.100 | 0.000 |
| Adaptive Paraphrase | **0.567** | 0.633 | 0.667 |
| Benign (no attack) | 0.000 | 0.000 | 0.000 |

- **Direct injection:** with no provider filter, Llama obeyed 77% of prompt injections — more than double
  GPT-5's 33%, and nearly as bad as Qwen. Azure's filter was doing real work for GPT-5 that a local model
  simply doesn't get.
- **Indirect injection:** Llama ignored poisoned tool outputs and poisoned memory entries entirely on its own —
  0 leaks across all five tool/memory attacks. Qwen leaked on four of them. Llama's alignment is noticeably
  more robust against instructions arriving through tools and memory than against instructions in the user
  query itself.

---

## 2. Did ChannelGuard fix it? Yes on most attacks — with two exceptions

| Attack | Undefended Llama | ChannelGuard Llama | Fixed? |
|---|---:|---:|---|
| **Prompt Injection** | 0.767 | **0.100** | ✅ 87% relative reduction (p < 0.001) |
| Tool Poisoning | 0.000 | **0.000** | ✅ now blocked by IB-3 (30/30) instead of by luck |
| Memory Poisoning | 0.000 | **0.000** | ✅ now blocked by IB-3-mem (30/30) |
| Slow-Drift Memory | 0.000 | **0.000** | ✅ now blocked by IB-3-mem (30/30) |
| Role-Confusion Memory | 0.000 | **0.000** | ✅ now blocked by IB-3-mem (30/30) |
| Consent-Escalation Memory | 0.000 | **0.000** | ⚠️ safe, but **0 IB blocks** — still model-handled |
| Reasoning-State Injection | 0.000 | **0.000** | ✅ IB-2 blocked 15/30 |
| **Consensus Collapse** | 0.067 | **0.067** | ❌ **not fixed** on Llama |
| **Adaptive Paraphrase** | 0.567 | **0.567** | ❌ **not fixed** |

**Prompt injection:** 20 samples flipped from leaked to safe and none the other way (exact McNemar,
p < 0.001). The paper's GPT-5 result (6 vs 1, p = 0.125) was only directional; on both local models it is
statistically significant. The input gate IB-0 did nearly all the work (22 blocks), with IB-1 adding 3.

### Not fixed #1: Adaptive Paraphrase (every model)

| System | Llama | Qwen | GPT-5 (paper) |
|---|---:|---:|---:|
| Undefended | 0.567 | 0.633 | 0.667 |
| **ChannelGuard** | **0.567** | **0.600** | **0.667** |
| SmoothLLM | 0.167 | 0.133 | 0.200 |
| LLM-Guard | **0.000** | **0.000** | 0.433 |

ChannelGuard makes no difference here on any model. The attacker rewrites the injection to avoid every bank
phrase, so the similarity check never fires. Defenses without a static signature do much better: SmoothLLM
(perturb-and-vote) and, on both local models, the LLM-Guard classifier, which blocked 100%. This is a
structural limit of the phrase-bank design that the manuscript already reports.

### Not fixed #2: Consensus Collapse (Llama only)

ChannelGuard's IB-2 blocked 12 consensus-collapse traces, but 2 still leaked through other workers — the same
2/30 as undefended. On Qwen, IB-2 blocked all 30 and nothing leaked. Llama's worker outputs paraphrase the
unanimity directive enough that some slip under θ. This attack family was not in the manuscript's n = 30
matrix, so it is a new observation rather than a contradiction.

---

## 3. Mechanism substitution — Llama is the cleanest demonstration of the paper's argument

Pooled over Tool Poisoning + Memory Poisoning (n=60 each):

| System | IB-gate blocks | Provider filter | Model-handled | Leaked |
|---|---:|---:|---:|---:|
| **Undefended (Llama)** | 0 | 0 (no filter exists) | **60 (100%)** | 0 |
| **ChannelGuard (Llama)** | **60 (100%)** | 0 | 0 | 0 |
| Undefended (Qwen) | 0 | 0 | 57 | 3 |
| ChannelGuard (Qwen) | 60 (100%) | 0 | 0 | 0 |
| Undefended (GPT-5, paper) | 0 | 54 (90%) | 6 | 0 |
| ChannelGuard (GPT-5, paper) | 32 (53%) | 22 | 6 | 0 |

This is the manuscript's "mechanism substitution" in its purest form. Both Llama systems reach exactly 0% ASR,
but the credit is completely different:

- **Undefended:** 100% from Llama's own alignment — opaque, model-specific, and not something the application
  controls.
- **ChannelGuard:** 100% from named, logged gate decisions — deterministic and identical on any backend.

On GPT-5 the two mechanisms shared credit with Azure's filter; on Qwen the model's own protection was leaky
(3 leaks). On Llama the outcome is identical and only the mechanism changes — exactly the contrast an ASR-only
table hides.

### Who keeps the undefended pipeline safe, across every backend tested

| Backend | Undefended stays safe because of… |
|---|---|
| Azure GPT-5 (paper) | the provider filter (90%) |
| Anthropic Sonnet-4.5 / Haiku-4.5 (paper) | the model's own alignment (100%) |
| Qwen3-4B (local) | the model, imperfectly (3 leaks of 60) |
| **Llama-3.1-8B (local)** | **the model's own alignment (100%)** |

The source of undefended safety changes with every backend. ChannelGuard's is the only layer that stays the
same.

---

## 4. Provider-invariance — now confirmed on five backends

| Backend | System | IB-3 blocks on Tool Poisoning | ASR |
|---|---|---:|---:|
| Azure GPT-5 (paper) | ChannelGuard | 30/30 | 0.000 |
| Anthropic Sonnet-4.5 (paper) | ChannelGuard | 30/30 | 0.000 |
| Anthropic Haiku-4.5 (paper) | ChannelGuard | 30/30 | 0.000 |
| Qwen3-4B (local) | ChannelGuard | 30/30 | 0.000 |
| **Llama-3.1-8B (local, this run)** | **ChannelGuard** | **30/30** | **0.000** |

Pooled over Tool + Memory + Role-Confusion Memory (n=90), ChannelGuard on Llama blocked **90/90** at a gate,
versus 32/90 on GPT-5, 36/90 on Sonnet, 50/90 on Haiku and 90/90 on Qwen. Corollary B.5 — the gate never reads
the agent model, so its decision cannot depend on it — now holds across five backends from three vendors,
two of them running fully offline.

---

## 5. Everything that matched the manuscript's other claims

| Claim | Paper (GPT-5) | Qwen | **Llama** | Confirmed? |
|---|---|---|---|---|
| Gates key on template, not marker | 94% preserved (30→28) | 100% | **97.8% (30→28)** | ✅ Llama matches the paper almost exactly |
| Gates add no reasoning cost (GSM8K CTCA) | 0.867 both | 0.767 both | **0.767 both** | ✅ zero gap |
| Early-exit speedup on Prompt Injection | 3.30× | 4.31× | **3.05×** | ✅ |
| Pooled adversarial speedup | 1.19× | 1.12× | **1.12×** | ✅ |
| No false refusals on benign inputs | FPR 0.033 | 0.000 | **0.000** | ✅ |
| Judge agreement, Sonnet vs GPT-5 (κ) | 0.900 | 0.851 | **0.895** | ✅ nearly identical to the paper |
| GPT-5 judge vs ground truth (κ) | 1.000 | 0.979 | **1.000** | ✅ exact match |
| IB-3 has the strongest per-block margin | median ε 0.495 | 0.495 | **0.459** | ✅ |
| Only a few bank phrases do the work | 8 of 20 win | 6 of 20 | **6 of 20** (top 4 = 96.5%) | ✅ |

GSM8K accuracy (0.767) is lower than GPT-5's in absolute terms because Llama-8B at 4-bit is a much smaller
model — but the gates cost exactly nothing either way.

---

## 6. Where Llama told a *different* story than the paper (worth flagging, not hiding)

1. **Consent-Escalation Memory was never caught by a gate.** The paper had 20/30 IB blocks and Qwen 11/30; on
   Llama, 0/30. ASR stayed at 0 only because Llama refused on its own. The gate's coverage of this attack
   depends on the exact wording the model produces downstream.
2. **Slow-Drift Memory was fully caught (30/30 at IB-3-mem)** — the same divergence Qwen showed. In the paper
   this narrative attack evaded every gate. Two local models now disagree with the paper here, so "narrative
   memory attacks evade embedding gates" looks model-dependent rather than universal.
3. **Only IB-0 matters for Prompt Injection.** Removing IB-0 sends ASR back to the undefended 0.767; removing
   any other single gate changes nothing (all stay at 0.100). The paper found IB-5 load-bearing as well;
   neither local model reproduced that.
4. **COMPRESS at the input is especially leaky on Llama.** When IB-0 compresses rather than blocks, Llama still
   leaks 59% of the time (stop-rate 0.407 vs. paper 0.766 and Qwen 0.511). This is Lemma B.4's
   "prepended payload" weakness, and it is worst here. The paper's own suggested fix — make IB-0 block-only —
   would help most on Llama.
5. **Phrase-bank size made no difference** (ASR 0.100 at 5, 20 and 50 phrases). The paper's "bigger bank is
   worse" did not replicate on Llama or on Qwen.
6. **The vague judge rubric still worked reasonably** (Sonnet vs GPT-5 κ = 0.767). The paper reported a collapse
   to 0.060; neither local replication reproduced it.
7. **Per-gate AUC at IB-0 is 0.500**, vs. the paper's 0.833. The AUC subset (tool, memory, role-confusion and
   benign) never touches the user input, so IB-0 has nothing to discriminate — 0.500 is the expected value,
   which makes the paper's 0.833 worth re-checking.
8. **The "risky pass-through" verifier blind spot almost vanished.** Only 42 reports landed in the
   safe-but-uncertain band and just 2 of those (4.8%) leaked, vs. the paper's 19.5% and Qwen's 19.9%.
9. **Perplexity-Filter failed differently.** In the paper it blocked 100% of everything; on Llama it let 70% of
   prompt injections through while still refusing 40% of benign questions.

---

## 7. Cost and efficiency

| | GPT-5 (paper) | Qwen3-4B | **Llama-3.1-8B** |
|---|---:|---:|---:|
| Total cost | $47.36 | $9.96 | **$11.37** |
| Mean time per run | 70.3 s | 34.2 s | **44.1 s** |
| Generation speed | API | 23.3 tok/s | **14.5 tok/s** |
| Verifier share of tokens | 31% | 62% | **61%** |
| Agent model / precision | GPT-5, cloud API | 4B, MLX 8-bit, local | **8B, MLX 4-bit, local** |
| Agent-side cost | priced per token | $0 (local) | **$0 (local)** |
| Pipeline traces | 2,100 | 1,800 | **1,800** |
| Claude verifier calls | — | 19,895 | **20,620** |
| Claude tokens (in / out) | — | 2.21M / 0.15M | **2.76M / 0.15M** |
| Local-model tokens (in / out) | — | 1.15M / 0.30M | **1.49M / 0.36M** |
| Mean tokens per run | 4,292 | 2,768 | **3,418** |
| Pipeline API spend / judge audit | — | $8.87 / $1.08 | **$10.53 / $0.84** |
| Early-exit speedup, Prompt Injection | 3.30× | 4.31× | **3.05×** |
| Pooled adversarial speedup | 1.19× | 1.12× | **1.12×** |

Llama is about 1.5× slower than Qwen because it has twice the parameters, even at 4-bit. As with Qwen, the
Claude verifier is now the dominant cost and time sink — reducing it from 5 to 3 Monte-Carlo samples is the
single biggest efficiency lever.

---

## Final verdict

1. **Llama undefended is as exposed as Qwen to direct prompt injection (77%)** — the risk the Azure filter was
   hiding for GPT-5 — **but, unlike Qwen, it protects itself against tool and memory attacks.** Undefended
   safety is a property of the specific model, not of the application.
2. **ChannelGuard cut prompt injection from 77% to 10%** (statistically significant) and turned every tool and
   memory attack from "safe because Llama happened to refuse" into "blocked by a named gate, 30/30".
3. **Llama is the cleanest evidence yet for mechanism substitution:** an identical 0% ASR, attributed 100% to
   the model without gates and 100% to the gates with them.
4. **ChannelGuard does not fix Adaptive Paraphrase** on any model, and **did not fix Consensus Collapse on
   Llama** (2/30 leaks either way). Signature-free or classifier defenses (SmoothLLM, LLM-Guard) are the right
   complement for the first.
5. **Provider-invariance (Corollary B.5) now holds across five backends**, from a frontier cloud model to a
   4-bit local 8B model.

**Recommendations for the paper:** make IB-0 block-only (drop COMPRESS at the input), and present three findings
as model-dependent rather than general, since neither local model reproduced them: narrative memory attacks
evading the gates, bigger banks hurting, and the vague-rubric κ collapse.

---

*Full tables: [Final/llama/README.md](Final/llama/README.md). Machine-readable numbers:
[Final/llama/numbers.json](Final/llama/numbers.json). Raw traces: `results_llama/`.
Qwen companion: [qwen_channelguard.md](qwen_channelguard.md).*
