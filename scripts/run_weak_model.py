"""Weak-model appendix run — same matrix but with Llama-3.1-8B via Ollama.

Requires `ollama serve` running and the model pulled:
    ollama pull llama3.1:8b-instruct-q4_K_M

Swaps `cfg.models.agent` for `cfg.models.weak_model` so `build_chat_client`
routes agent calls (planner / workers / synthesizer) to the local Ollama
endpoint. The verifier + judges keep their Anthropic models.
"""
from __future__ import annotations

import argparse
import copy

import httpx

from infoshield.config import load_config
from infoshield.datasets import load as load_dataset
from infoshield.runner import run_matrix

SYSTEMS = ["undefended", "infoshield"]
ATTACKS = [
    "prompt_injection_strong",
    "tool_poisoning",
    "memory_poisoning",
    "reasoning_state_injection",
    "consensus_collapse",
]


def _ollama_up(url: str = "http://localhost:11434") -> bool:
    try:
        r = httpx.get(f"{url}/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    args = ap.parse_args()

    cfg = load_config()
    if not _ollama_up():
        print(
            "[skip] Ollama not reachable at http://localhost:11434 — "
            "the weak-model appendix run needs 'ollama serve' + "
            "`ollama pull llama3.1:8b-instruct-q4_K_M`."
        )
        return

    # Actually swap the agent model to the configured weak model (e.g. Llama-3.1
    # via Ollama). The pipeline's build_chat_client will pick the right provider.
    cfg_weak = copy.deepcopy(cfg)
    cfg_weak.models.agent = cfg.models.weak_model

    ecfg = cfg.experiments["weak_model"]
    samples = load_dataset(ecfg.dataset, n_samples=args.n_samples or ecfg.n_samples, seed=(args.seeds or ecfg.seeds)[0])

    stats = run_matrix(
        cfg=cfg_weak,
        experiment="weak_model",
        dataset_name=ecfg.dataset,
        samples=samples,
        systems=SYSTEMS,
        attacks=ATTACKS,
        seeds=args.seeds or ecfg.seeds,
        progress_cb=print,
    )
    print(f"weak_model: {stats}")


if __name__ == "__main__":
    main()
