"""Phase 9 smoke: 2 HotpotQA samples through the full INFOSHIELD pipeline.

Runs THREE cells to prove the whole stack works:
  1. undefended baseline, no attack        -> should answer correctly
  2. infoshield, no attack                  -> should also answer correctly (no CTCA loss)
  3. infoshield, prompt_injection_strong    -> should refuse or scrub the injection

Verifies that:
  * IB-0..IB-5 gates fire (visible in trace.gate_decisions)
  * Verifier runs 5-sample MC (visible in trace.verifier_reports[*].samples)
  * Synthesizer applies triple-filter
  * ATTACK_SUCCESS marker does NOT appear in the infoshield final output
"""
from __future__ import annotations

import argparse

from infoshield.attacks import ATTACKS
from infoshield.attacks.registry import ATTACK_MARKER
from infoshield.baselines import SYSTEMS
from infoshield.config import load_config
from infoshield.datasets import load as load_dataset


def _run_cell(system: str, attack: str, samples, cfg):
    print(f"\n===== {system} / {attack} =====")
    pipeline = SYSTEMS[system](cfg)
    atk_fn = ATTACKS[attack]
    for s in samples:
        atk_kwargs = atk_fn(s["input"], s.get("meta", {}))
        trace = pipeline(
            sample_id=s["sample_id"],
            ground_truth=s["ground_truth"],
            experiment="phase9_smoke",
            system=system,
            attack=attack,
            seed=0,
            dataset="hotpot_qa",
            **atk_kwargs,
        )
        path = trace.write(cfg.output_root)
        leaked = ATTACK_MARKER in (trace.final_output or "")
        n_gates = len(trace.gate_decisions)
        n_blocks = trace.n_blocked()
        n_verif = len(trace.verifier_reports)
        marker = "!! LEAKED" if leaked else ("[refused]" if trace.refused else "[ok]")
        print(f"  {s['sample_id']}  gates={n_gates}  blocks={n_blocks}  verifs={n_verif}  {marker}")
        print(f"     final: {(trace.final_output or '')[:120]}")
        print(f"     -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    samples = load_dataset("hotpot_qa", n_samples=args.n_samples, seed=args.seed)

    _run_cell("undefended", "none", samples, cfg)
    _run_cell("infoshield", "none", samples, cfg)
    _run_cell("infoshield", "prompt_injection_strong", samples, cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()
