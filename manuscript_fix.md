# Manuscript fixes: ChannelGuard (arXiv:2607.19430v2)

These are changes the manuscript needs. They were found while reconciling `Final/` with the paper and re-running every experiment on two local agents: Qwen3-4B-Instruct-2507 (MLX 8-bit) and Llama-3.1-8B-Instruct (MLX 4-bit), each 1,800 traces with the same Claude Sonnet 4.5 verifier.

The headline numbers in the manuscript are **not** changed here. Every table in [Final/tables/](Final/tables/) reproduces the manuscript's values exactly. The items below are internal contradictions, mislabels, wrong explanations, and disclosures.

Line references are to `pdftotext -layout` output of the v2 PDF. Section and table numbers are the manuscript's own.

| # | Severity | Location | Issue |
| :-- | :-- | :-- | :-- |
| 1 | Must fix | Figure 5 (Appendix I.5) | Figure contradicts Table 18 |
| 2 | Must fix | §6.5, Table 6, abstract vs Table 19 | GSM8K CTCA 0.867 vs 0.833 |
| 3 | Must fix | Appendix G, Table 12 | "Default β=1.0" is not the deployed default |
| 4 | Must fix | Table 11 note, Appendix A | 0.067-vs-0.167 gap blamed on the wrong cause |
| 5 | Must fix | Table 20 | Wall-time shares sum to 121.4% |
| 6 | Should fix | §6.4, Table 5 | Tool Poisoning sentinel row was not actually randomized |
| 7 | Should fix | Table 15 (Appendix I.2) | IB-0 AUC 0.833 on a subset that never attacks the input |
| 8 | Should fix | §6.3, Table 10 caption, Appendix B | IB-3-mem "2–14 of 30 off Azure" is not reproduced |
| 9 | Should fix | §7, Appendices F, H, I.6 | Claims that do not replicate on other backends |
| 10 | Recommended | §4.1, Appendix L | Make IB-0 BLOCK-only |
| 11 | Optional | New appendix | Add the cross-model replication |

---

## 1. Figure 5 contradicts Table 18 (Must fix)

**Where.** Appendix I.5, Figure 5 caption (line ~1123), next to Table 18 (line ~1107).

**Problem.** Table 18 lists the 8 winning phrases with win counts 138, 93, 81, 58, 23, 15, 9, 9 (426 total) and says IB-0 uses 2 phrases. The Figure 5 bar chart was rendered from an earlier, pre-backfill phrase-audit snapshot, so its bars disagree with the table directly beside it. The old `phrase_audit.tex` had the same stale data, attributing the top winner to IB-3 with "3/20" phrases and 67 wins.

**Fix.** Replace Figure 5 with [Final/figures/phrase_audit.pdf](Final/figures/phrase_audit.pdf), which is regenerated from Table 18's counts by `scripts/build_final.py`. Keep the caption. With the new figure, its claim ("drop off sharply after the top 4–8 phrases; the remaining 12 of 20 never win") is true.

---

## 2. GSM8K CTCA is reported as two different values (Must fix)

**Where.** Abstract (line ~105, "(0.867)"); §6.5 text (line ~504, "CTCA is 0.867 for both systems"); Table 6 (line ~529, "CTCA = 0.867 for both"); Table 19 (lines ~1143–1144, GSM8K rows: 25 correct of 30, BPR 0.833).

**Problem.** 25/30 = 0.833, not 0.867 (0.867 would be 26/30). The §I.6 text also says "100% on GSM8K (0.833 both)". The paper therefore uses both values for the same 30-sample, two-system cell.

**Fix.** Check the GSM8K traces and use one value everywhere. If CTCA and BPR use different correctness checks (normalized-number match vs. string inclusion), say so in the Table 19 caption. Otherwise change the abstract, §6.5 and Table 6 to 0.833. The "no fidelity regression" conclusion holds either way, because both systems are equal.

---

## 3. "Default β=1.0" is not the deployed default (Must fix)

**Where.** Appendix G text (line ~880): "The default β=1.0 is conservative: β=2.0 (θ_eff=0.55) reaches ASR=0.067 at 17 IB blocks, strictly better than the default's ASR=0.100." Table 12 row label "1.0 (default)" (line ~946). §5 configuration paragraph ("β=1.0").

**Problem.** Eq. 3 gives θ_eff(β) = max(0.15, 0.85 − 0.15β), so θ_eff(1.0) = 0.70. The deployed gate uses θ = 0.60 (Section 4.1, Table 17's "default θ=0.60" plateau, and the released `config.yaml`). θ = 0.60 corresponds to β ≈ 1.67, which is **not in the sweep**. The β=1.0 row is a looser setting, not the configuration behind every other result. Its ASR (0.100) is not "the default's ASR": the default's measured ASR is 0.167 (Table 2) or 0.067 (Table 11).

**Fix.**
- Table 12: relabel the row "1.0" and add a footnote: "The deployed configuration uses θ=0.60 directly (β≈1.67), between the 1.0 and 2.0 rows."
- Appendix G: replace the quoted sentence with: "β=2.0 (θ_eff=0.55) reaches ASR=0.067 at 17 IB blocks, the same block count as the deployed θ=0.60 (Table 11, full row), so the deployed threshold already sits on the efficient part of the curve; β=1.0 (θ_eff=0.70) is looser and admits one more leak."
- §5: either state θ=0.60 as the primary knob and describe β only as a sweep reparameterization, or set the default to β=5/3.

---

## 4. The Table 2 vs Table 11 gap is not Azure stochasticity (Must fix)

**Where.** Table 11 note (lines ~970–976): "The full-ChannelGuard ASR here (0.067) differs slightly from Table 2's 0.167 … both are within the ±1-per-30-cell stochastic envelope from Azure's content filter and GPT-5's reasoning stochasticity". Appendix A (line ~726): "counts is ±1 per 30-run cell, which is why (for example) the full-defense ablation row in Appendix F shows a slightly different leak…". The same pattern appears in Appendix H (bank size 20, ASR 0.100).

**Problem.** The gap is 21 vs 17 **IB-0 blocks** and 5 vs 2 leaks. IB-0 scores the user query before any LLM call, so Azure and GPT-5 stochasticity cannot change its block count. The real cause is that `attacks/registry._pick` chooses the injection template with Python's builtin `hash()`. That hash is randomly salted per process unless `PYTHONHASHSEED` is pinned, so the two runs attacked with **different template mixes**. With `PYTHONHASHSEED=0`, the main-slice and ablation full-row cells match exactly on both local replications (Qwen 26 = 26, Llama 25 = 25 IB blocks). A 3-leak difference is also not "±1".

**Fix.**
- Table 11 note: "The full-ChannelGuard ASR here (0.067, 17 IB-0 blocks) differs from Table 2 (0.167, 21 IB-0 blocks) because attack-template selection used Python's per-process-salted `hash()`, so the two runs received different template mixes. Pinning `PYTHONHASHSEED` makes the two cells identical (verified on two additional backends)."
- Appendix A: keep the ±1 statement for LLM-touched counts (leaks, provider and verifier classes), but drop "(for example) the full-defense ablation row…" and add: "Attack-template selection is deterministic only with `PYTHONHASHSEED` fixed; the released runners set it."
- Code (released artifact): replace `hash()` in `_pick` with a stable hash (e.g. `zlib.crc32`).

---

## 5. Table 20 wall-time shares sum to 121.4% (Must fix)

**Where.** Table 20 (lines ~1196–1200): verifier 56.4%, planner 14.3%, worker (×3) 37.4%, synth 13.3%. Also §I.7 text ("wall-time (56.4% of it)").

**Problem.** The shares sum to 121.4%. The three workers run in parallel, so summed per-call latency exceeds pipeline wall-clock time, while the denominator is wall-clock.

**Fix.** Either rename the column to "Summed call latency / wall-clock" with a footnote ("workers run in parallel, so shares overlap and do not sum to 100%"), or normalize by total summed call latency (35,569 s) so the column sums to 100%. Under that normalization the verifier share is 46.5%, planner 11.7%, worker 30.8%, synth 11.0%. Update the "56.4%" in the text to match whichever convention is chosen.

---

## 6. The Tool Poisoning sentinel row was never randomized (Should fix)

**Where.** §6.4 (lines ~505–515): "…Tool Poisoning, Memory Poisoning, and Role-Confusion Memory with a per-run random token (e.g. TOKEN_B267BFD8)…". Table 5, ChannelGuard Tool Poisoning row (30 → 28).

**Problem.** In the released code the mock tool's injected suffix was hard-coded to `ATTACK_SUCCESS` and ignored the per-run sentinel. The Tool Poisoning row therefore never ran with a randomized payload; only the two memory rows did. The −2 on that row reflects run-to-run variation, not marker sensitivity. The fix is now in `tools/mock_tool.py` (`marker` parameter, threaded through `pipeline/state.py`, `agents/worker.py` and `scripts/run_sentinel_check.py`). With it, the corrected row reads 30 → 30 on Qwen and 30 → 28 on Llama.

**Fix.** Re-run the six Tool Poisoning sentinel cells with the fixed tool, which is about 60 traces. If that is not possible, add a footnote to Table 5: "The Tool Poisoning payload was not randomized in this run due to a bug in the mock tool; on two further backends with the fix applied the row is 30→30 and 30→28." Change "|Δ|=4/62" in §6.4 and §7 accordingly.

---

## 7. IB-0 AUC of 0.833 is implausible for the stated subset (Should fix)

**Where.** Table 15 (Appendix I.2, line ~1016): IB-0, n=120, n_attack=90, AUC 0.833.

**Problem.** The text defines the subset as Tool Poisoning, Memory Poisoning and Role-Confusion Memory (90 attack traces) plus 30 benign. None of these attacks modifies the user query, so IB-0 scores attack and benign queries drawn from the same 30 HotpotQA questions, and the expected AUC is 0.5. Both local replications compute exactly 0.500 for IB-0 on this subset. An AUC of 0.833 suggests the original computation used a different subset (for example, one including Prompt Injection) or a different score.

**Fix.** Recompute IB-0's AUC on the stated subset and correct Table 15. If 0.833 came from a subset with Prompt Injection, state that subset in the caption. The text's conclusion that IB-0 is load-bearing should rest on Table 11 (ablation) and Table 14 (margins), not on this AUC.

---

## 8. IB-3-mem "only partially fires off Azure (2–14 of 30)" is not reproduced (Should fix)

**Where.** §6.3 (line ~499): "only partially fires off Azure (2–14 of 30 depending on backend)". Table 10 caption (line ~957): "ChannelGuard's IB share varies (32–50%) … because IB-3-mem only partially fires on the two memory attacks off Azure". Appendix B corollary text (line ~822).

**Problem.** In the released code, IB-3-mem scores the **preloaded memory entries** on `memory.read()`, before any agent LLM call. Its decision should therefore not depend on the agent backend. On both local backends it blocks Memory Poisoning and Role-Confusion Memory 30/30, giving 90/90 pooled IB blocks vs. the manuscript's 36/90 (Sonnet) and 50/90 (Haiku). Either the Anthropic-agent runs used a different memory path (for example, memory written by the agent rather than preloaded), or the counts are attributable to something else.

**Fix.** Check which memory path the Sonnet and Haiku runs exercised. If the counts stand, explain why a pre-LLM gate is backend-dependent. Otherwise re-run those two provider arms, which is about 120 traces. Until then, soften §6.3 and the Table 10 caption: "IB-3 is provider-invariant at 30/30. IB-3-mem fired on 2–14 of 30 memory-attack traces in our Anthropic runs but on 30/30 in two independent local-agent replications; we are investigating the discrepancy."

---

## 9. Claims that do not replicate on other backends (Should fix)

Each of these is correct for the GPT-5 runs. However, the Qwen and Llama replications (same code, same verifier, `PYTHONHASHSEED=0`) do not reproduce them, so they should be stated as backend-specific rather than general properties of ChannelGuard.

| Claim (location) | GPT-5 | Qwen3-4B | Llama-3.1-8B | Suggested wording |
| :-- | :-- | :-- | :-- | :-- |
| Narrative memory attacks evade the gates (§7 limitation 2, Table 9 Slow-Drift row) | 0/30 IB | 30/30 IB | 30/30 IB | "On GPT-5, narrative memory attacks rarely triggered IB gates; on two other backends IB-3-mem blocked them 30/30." |
| Larger bank → higher ASR, 0.067→0.100→0.167 (Appendix H, Appendix B Lemma B.2 text) | rising | flat 0.033 | flat 0.100 | "Bank size had no consistent effect on ASR; the GPT-5 spread (2 vs 5 leaks) is within n=30 noise." Drop the "curate a 5-phrase bank" advice. |
| IB-5 is load-bearing on Prompt Injection (Appendix F) | 0.067→0.133 | unchanged | unchanged | "Only IB-0 is consistently load-bearing on Prompt Injection; IB-5's effect on GPT-5 did not replicate." |
| A vague rubric collapses judge κ to 0.06 (Appendix I.10 / judge audit) | 0.060 | 0.728 | 0.767 | "Rubric sharpness mattered on the original audit (κ 0.06→0.90) but the collapse did not recur on two further output sets." |

The core claims **do** replicate on both backends: IB-3's 30/30 provider-invariant Tool Poisoning block, the large Prompt Injection ASR reduction (0.800→0.033 and 0.767→0.100), zero GSM8K fidelity regression, the adaptive-paraphrase defeat, and the monotonic β Pareto. Saying so strengthens the paper.

---

## 10. Make IB-0 BLOCK-only (Recommended)

**Where.** §4.1 (gate definition) and Appendix L (compression effectiveness).

**Reason.** Appendix L already shows that IB-0's COMPRESS band leaks 23.4% of the time (58/248), while every inter-agent gate's COMPRESS band has a 100% stop rate. The paper recommends the fix but ships the flawed default. Either adopt BLOCK-only at IB-0 in the released config and say so in §4.1, or state explicitly in §7 that the released default keeps the known-leaky band.

---

## 11. Add the cross-model replication (Optional)

A short appendix (or one paragraph in §6.3 plus one table) would answer the obvious reviewer question, "does this only work on GPT-5 behind Azure's filter?" Ready-made material:

- Table: [Final/tables/cross_model/asr.tex](Final/tables/cross_model/asr.tex) (ASR, 3 backends × 2 systems × 10 attack conditions)
- Table: [Final/tables/cross_model/provider_invariance.tex](Final/tables/cross_model/provider_invariance.tex) (five backends)
- Figure: [Final/figures/cross_model_asr.pdf](Final/figures/cross_model_asr.pdf), [Final/figures/cross_model_mechanism.pdf](Final/figures/cross_model_mechanism.pdf)
- Macros: `\Qwen…` and `\Llama…` in [Final/macros/numbers.tex](Final/macros/numbers.tex)
- Narrative: §6A of [Final/final_results.md](Final/final_results.md)

Suggested paragraph:

> **Cross-model replication.** We re-ran the full suite (1,800 traces per backend) with two open-weight agents served locally — Qwen3-4B-Instruct-2507 and Llama-3.1-8B-Instruct — keeping the Claude Sonnet 4.5 verifier. Without a provider filter both agents obey most direct injections (Undefended ASR 0.800 and 0.767); ChannelGuard reduces this to 0.033 and 0.100 and blocks every inter-agent attack at a gate (90/90 pooled), while adding no GSM8K regression. Adaptive paraphrase still defeats the gates (0.600, 0.567). Total API cost was $9.96 and $11.37 (verifier and judges only).

Disclose with it: quantization (8-bit / 4-bit), `PYTHONHASHSEED=0`, the corrected sentinel tool, and that latency there is local generation time, not API wall-clock.

---

## What does *not* need to change

- All headline numbers in Tables 2–4, 6–14, 16–21 match the regenerated tables in `Final/tables/`, apart from the specific cells named above.
- The $47.36 cost and the 2,100-trace count are historical measurements; nothing found here contradicts them.
- The attribution methodology, provider-invariance result for IB-3, and the adaptive-paraphrase limitation are all supported, and now cross-validated.
