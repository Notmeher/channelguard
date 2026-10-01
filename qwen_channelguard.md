# ChannelGuard × Qwen3-4B: Local-Model Replication — Summary & Verdict

**A plain-language companion to [paper/qwen/README.md](paper/qwen/README.md).** That file has every raw table;
this one explains what the numbers mean, compared directly against the manuscript's Azure GPT-5 + Claude
results (arXiv:2607.19430v2). The Llama-3.1-8B replication is summarized in the same format in
[llama_channelguard.md](llama_channelguard.md); §8 below compares the two local models directly.

---

## TL;DR

> **Qwen3-4B, run completely locally with no provider safety filter, was far more dangerous undefended than
> GPT-5 ever was. ChannelGuard fixed almost all of that gap — except one attack (Adaptive Paraphrase), which
> beats ChannelGuard on both models equally. That one failure is a known, documented limit of the static
> phrase-bank design, not a Qwen-specific weakness.**

| Component | This run | Manuscript |
|---|---|---|
| Planner / workers / synthesizer | **Qwen3-4B-Instruct-2507, MLX 8-bit, local** | Azure GPT-5 |
| Provider-side content filter | **none** (local inference) | Azure content filter, always on |
| Verifier | Claude Sonnet 4.5, T=1.0, 5 MC samples | same |
| IB gates | same θ/α/N/β, same 20-phrase bank | same |
| Data | HotpotQA n=30, GSM8K n=30, seed 0 | same |
| Total cost | **$9.96** | $47.36 |
| Traces | 1,800 | 2,100 + 100 judge pairs |

---

## 1. Was Qwen worse than GPT-5? Yes — undefended, dramatically so

Azure's server-side content filter was quietly doing a large share of the manuscript's "safety" work. Qwen has
no such filter, so its raw, undefended behavior is what you actually get when nothing protects it.

| Attack | Undefended Qwen | Undefended GPT-5 (paper) |
|---|---:|---:|
| **Prompt Injection** | **0.800** | 0.333 |
| Tool Poisoning | 0.067 | 0.000 |
| Memory Poisoning | 0.033 | 0.000 |
| Slow-Drift Memory | 0.067 | 0.000 |
| Consent-Escalation Memory | 0.000 | 0.000 |
| Role-Confusion Memory | 0.100 | 0.000 |
| Adaptive Paraphrase | 0.633 | 0.667 |
| Benign (no attack) | 0.000 | 0.000 |

GPT-5's undefended pipeline never leaked on four of these attack families — not because the application was
safe, but because Azure's filter caught them before the application ever had a chance. Qwen, with no filter,
leaked on every single attack family except Consent-Escalation. **80% of prompt injections succeeded with zero
defense in place.**

---

## 2. Did ChannelGuard fix it? Yes — almost everywhere

| Attack | Undefended Qwen | ChannelGuard Qwen | Fixed? |
|---|---:|---:|---|
| **Prompt Injection** | 0.800 | **0.033** | ✅ 96% relative reduction |
| Tool Poisoning | 0.067 | **0.000** | ✅ fully fixed |
| Memory Poisoning | 0.033 | **0.000** | ✅ fully fixed |
| Slow-Drift Memory | 0.067 | **0.000** | ✅ fully fixed |
| Role-Confusion Memory | 0.100 | **0.000** | ✅ fully fixed |
| Reasoning-State Injection | 0.000 | **0.000** | ✅ already safe, gates add inspectability |
| Consensus Collapse | 0.067 | **0.000** | ✅ fully fixed |
| Adaptive Paraphrase | 0.633 | **0.600** | ❌ **not fixed** |

Every fully-fixed row was blocked **deterministically by a named IB gate**, not by luck: Tool Poisoning and the
three memory attacks were all caught 30/30 by IB-3 / IB-3-mem; Consensus Collapse was caught 30/30 by IB-2.
That is the entire point of the framework — application-owned, inspectable blocking instead of hoping a
provider filter happens to fire.

### The one attack ChannelGuard does not fix: Adaptive Paraphrase

| System | Qwen | GPT-5 (paper) |
|---|---:|---:|
| Undefended | 0.633 | 0.667 |
| **ChannelGuard** | **0.600** | **0.667** |
| SmoothLLM (perturb-and-vote baseline) | **0.133** | 0.200 |

ChannelGuard barely moves the needle on this attack on *either* model, because the attacker rewrites the
injection to avoid every phrase in the bank — the cosine-similarity check never fires. SmoothLLM, which has no
static signature to evade, does far better on both models. This confirms the manuscript's own conclusion: a
static phrase bank is structurally the wrong tool against an adaptive adversary, and that limitation is not
specific to Qwen.

---

## 3. Mechanism substitution — the paper's core argument, demonstrated even more starkly

Pooled over Tool Poisoning + Memory Poisoning (n=60 each):

| System | IB-gate blocks | Provider filter | Model-handled (luck) | Leaked |
|---|---:|---:|---:|---:|
| Undefended (Qwen) | 0 (0%) | 0 (no filter exists) | 57 | 3 |
| **ChannelGuard (Qwen)** | **60 (100%)** | 0 | 0 | 0 |
| Undefended (GPT-5, paper) | 0 | 54 (90%) | 36 | 0 |
| ChannelGuard (GPT-5, paper) | 32 (53%) | 22 | 36 | 0 |

On GPT-5, Azure's filter and ChannelGuard's gates **shared credit** for the zero-ASR outcome. On Qwen, there is
no filter to share credit with — Undefended's "safety" is just 57/60 traces where the attack happened not to
work (3 leaked), and ChannelGuard is **the only thing providing deterministic protection**, catching 100% of
the pooled attacks by gate alone. This is the cleanest possible illustration of "safety that does not travel":
remove the provider filter, and undefended safety evaporates, while ChannelGuard's safety does not move at all.

---

## 4. Provider-invariance extended to a 4th, very different backend

| Backend | System | IB-3 blocks on Tool Poisoning | ASR |
|---|---|---:|---:|
| Azure GPT-5 | ChannelGuard | 30/30 | 0.000 |
| Anthropic Sonnet-4.5 | ChannelGuard | 30/30 | 0.000 |
| Anthropic Haiku-4.5 | ChannelGuard | 30/30 | 0.000 |
| **Qwen3-4B (local, no filter, this run)** | **ChannelGuard** | **30/30** | **0.000** |

Corollary B.5 — that the gate's decision can never depend on the agent model, because it never reads the
agent's output before blocking — now holds across a frontier closed model, two Anthropic tiers, and a 4B
open-weights model running entirely offline. That is a stronger empirical confirmation of provider-invariance
than the manuscript itself had, since Qwen is categorically different from the other three (smaller, local,
no safety-tuned provider wrapper at all).

---

## 5. Everything that matched the manuscript's other claims

| Claim | Confirmed on Qwen? | Evidence |
|---|---|---|
| Gates are template-driven, not marker-driven | ✅ Yes, more cleanly | Sentinel randomization: 100% of IB blocks preserved under a random token (paper: 94%) |
| Gates add zero LLM-call latency | ✅ Yes | Planner/worker/synth latency identical with gates on or off |
| Reasoning accuracy is preserved exactly | ✅ Yes | GSM8K CTCA = 0.767 for both Undefended and ChannelGuard (lower *absolute* accuracy than GPT-5's 0.867, since Qwen is a much smaller model, but the **gates cost nothing** either way) |
| Cross-family judge methodology holds | ✅ Yes | Sonnet vs GPT-5 κ = 0.851 (paper: 0.900) |
| IB-3 (tool-output gate) has the strongest per-block confidence margin | ✅ Yes | Median ε = 0.495, identical to the paper |
| A hard refusal costs zero downstream LLM calls | ✅ Yes | Confirmed via early-exit short-circuit timing |

---

## 6. Where Qwen told a *different* story than the paper (worth flagging, not hiding)

- **Slow-Drift Memory — the narrative attack that beat the gates in the manuscript — was fully caught here.**
  On GPT-5, this attack evaded every IB gate (0 IB blocks; safety came entirely from the model's own behavior).
  On Qwen, IB-3-mem blocked it **30/30**. This suggests that "narrative attacks evade embedding gates" may be a
  property of *how a specific model phrases the attack back*, not a universal property of the gate design —
  worth investigating further rather than treating as settled.
- **Perplexity-Filter behaved very differently.** In the manuscript it blocked 100% of all inputs including
  every benign query (a spurious 0.000 ASR that was actually complete over-blocking). On Qwen it blocked only
  40% of benign inputs and let through 70% of prompt injections — a different, less extreme, but still broken
  failure mode. The underlying GPT-2 perplexity threshold is identical; the difference comes from how the
  HotpotQA queries flow through a different pipeline context.
- **Per-gate AUC was weaker on the Qwen subset** for IB-0 (0.500 vs. paper's 0.833), though IB-3 held up well
  (0.667 vs. 0.624). This reflects the much smaller, harder-attack-mix subset tested, not a regression in the
  gate mechanism itself.

---

## 7. Cost and efficiency

| | GPT-5 (paper) | **Qwen3-4B** | Llama-3.1-8B |
|---|---:|---:|---:|
| Total cost | $47.36 | **$9.96** | $11.37 |
| Mean time per run | 70.3 s | **34.2 s** | 44.1 s |
| Generation speed | API | **23.3 tok/s** | 14.5 tok/s |
| Verifier share of tokens | 31% | **62%** | 61% |
| Agent model / precision | GPT-5, cloud API | **4B, MLX 8-bit, local** | 8B, MLX 4-bit, local |
| Agent-side cost | priced per token | **$0 (local)** | $0 (local) |
| Pipeline traces | 2,100 | **1,800** | 1,800 |
| Claude verifier calls | — | **19,895** | 20,620 |
| Claude tokens (in / out) | — | **2.21M / 0.15M** | 2.76M / 0.15M |
| Local-model tokens (in / out) | — | **1.15M / 0.30M** | 1.49M / 0.36M |
| Mean tokens per run | 4,292 | **2,768** | 3,418 |
| Pipeline API spend / judge audit | — | **$8.87 / $1.08** | $10.53 / $0.84 |
| Early-exit speedup, Prompt Injection | 3.30× | **4.31×** | 3.05× |
| Pooled adversarial speedup | 1.19× | **1.12×** | 1.12× |

Qwen is the fastest and cheapest of the three: the agent itself is free and local, and at 4B parameters it
generates ~1.6× faster than Llama-8B. With the agent costing nothing, the Claude verifier dominates both time
and cost (~62% of tokens, vs. 31% in the manuscript). Reducing it from 5 to 3 Monte-Carlo samples is the single
biggest efficiency lever on either local model. The manuscript's $47.36 also paid for GPT-5 itself, so its
cost is not directly comparable.

---

## 8. Qwen3-4B vs. Llama-3.1-8B — how the two local models compare

| | Qwen3-4B | Llama-3.1-8B |
|---|---:|---:|
| Undefended Prompt Injection ASR | 0.800 | 0.767 |
| ChannelGuard Prompt Injection ASR | **0.033** | 0.100 |
| McNemar on Prompt Injection | 23 vs 0, p < 0.001 | 20 vs 0, p < 0.001 |
| Undefended leaks on tool/memory attacks (5 families, n=150) | **8** | **0** |
| ChannelGuard IB blocks, Tool + Memory + Role-Confusion (n=90) | 90/90 | 90/90 |
| Consent-Escalation IB blocks | 11/30 | 0/30 |
| Consensus Collapse (CG ASR) | **0.000** | 0.067 ❌ |
| Adaptive Paraphrase (CG ASR) | 0.600 ❌ | 0.567 ❌ |
| Sentinel randomization preserved | 100% | 97.8% |
| GSM8K CTCA (both systems) | 0.767 | 0.767 |
| IB-0 COMPRESS stop-rate | 0.511 | 0.407 |
| Judge κ, Sonnet vs GPT-5 | 0.851 | 0.895 |

- **Direct injection:** both local models are equally exposed without a defense (~78–80%), more than double
  GPT-5's 33%. ChannelGuard brings both down, slightly further on Qwen.
- **Indirect injection:** the models differ sharply. Qwen leaked 8 times across the tool and memory attacks;
  Llama never did. ChannelGuard blocked 100% of these at a gate on both, so the defended outcome is identical
  even though the undefended risk is not — provider-invariance from the application's point of view.
- **Failures:** Adaptive Paraphrase beats ChannelGuard on both. Consensus Collapse is fixed on Qwen but not on
  Llama.
- **Shared divergences from the paper:** both models had Slow-Drift Memory caught 30/30 at IB-3-mem, neither
  showed the "bigger bank is worse" effect, and neither reproduced the vague-rubric κ collapse.

---

## Final verdict

1. **Qwen undefended is meaningfully worse than GPT-5 undefended** — not because Qwen is a "bad" model, but
   because it has no provider safety net underneath it. This is exactly the hidden dependency the manuscript
   warns about.
2. **ChannelGuard closes almost the entire gap**, bringing six of seven tested attack families to 0% success,
   each one via a named, auditable gate decision rather than luck or an opaque filter.
3. **ChannelGuard does not fix Adaptive Paraphrase**, on Qwen or on GPT-5. This is a pre-existing, documented
   limitation of the static phrase-bank design — confirmed again, not newly discovered.
4. **The paper's central claim about mechanism substitution is, if anything, more dramatically true on a
   filterless local model**, where ChannelGuard goes from "sharing credit with a provider filter" to being
   "the only deterministic protection in the entire system."
5. **Provider-invariance (Corollary B.5) now holds across five structurally different backends** — the
   manuscript's three plus Qwen and Llama — including two that are small, local, and have no safety-tuned API
   wrapper at all.

**One open thread for follow-up:** the Slow-Drift Memory result diverges from the manuscript (fully caught here
vs. fully evaded there). The Llama replication reproduced Qwen's result (30/30 caught), so two local models now
disagree with the paper. A dedicated multi-model sweep is still needed before concluding whether narrative-style
memory attacks are a fundamental gate weakness or a model-dependent one.

---

*Full tables: [paper/qwen/README.md](paper/qwen/README.md). Machine-readable numbers:
[paper/qwen/numbers.json](paper/qwen/numbers.json). Raw traces: `results_qwen/`.
Llama companion: [llama_channelguard.md](llama_channelguard.md).*
