# Step 3 — gsm8k_fidelity

GSM8K under no attack for both systems. Proves INFOSHIELD does not degrade multi-step reasoning (CTCA fidelity).

- **Experiment**: `gsm8k`
- **Systems**: `undefended`, `infoshield`
- **Attacks**: `none`
- **n_samples**: 15  |  **seeds**: [0]

## Runtime

- runs done: **30**
- runs skipped (resumed): 0
- runs failed: 0
- wall-clock: **0 s** (0.0 min)
- estimated cost: **$0.00**  _(at $0.25/run)_

## Per-cell metrics (all traces on disk for this experiment)

| System | Attack | Seed | n | ASR | CTCA | Mean latency (ms) | Mean tokens |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `infoshield` | `none` | 0 | 15 | 0.000 | 0.933 | 59970 | 4137 |
| `undefended` | `none` | 0 | 15 | 0.000 | 0.867 | 58729 | 4134 |

_Written at 2026-07-03T05:04:29.480144+00:00._
