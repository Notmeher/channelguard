"""Dataset loaders — return list[dict] with uniform keys: sample_id, input, ground_truth."""
from __future__ import annotations

import hashlib
from typing import Callable

from datasets import load_dataset

_DATASETS: dict[str, Callable[[int, int], list[dict]]] = {}


def register(name: str):
    def deco(fn):
        _DATASETS[name] = fn
        return fn
    return deco


def load(name: str, n_samples: int, seed: int = 0) -> list[dict]:
    if name not in _DATASETS:
        raise KeyError(f"Unknown dataset {name!r}. Registered: {list(_DATASETS)}")
    return _DATASETS[name](n_samples, seed)


def _id(prefix: str, key: str) -> str:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{h}"


@register("hotpot_qa")
def _hotpot(n: int, seed: int) -> list[dict]:
    # HF now requires namespaced dataset ids. Try the community mirror first,
    # then fall back to the legacy path.
    for repo, cfg in (
        ("hotpotqa/hotpot_qa", "distractor"),
        ("hotpot_qa", "distractor"),
    ):
        try:
            ds = load_dataset(repo, cfg, split="validation")
            break
        except Exception:
            continue
    else:
        raise RuntimeError("Could not load HotpotQA from any known HF path.")
    ds = ds.shuffle(seed=seed)
    out: list[dict] = []
    for row in ds.select(range(min(n, len(ds)))):
        out.append(
            {
                "sample_id": _id("hp", row["id"]),
                "input": row["question"],
                "ground_truth": row["answer"],
            }
        )
    return out


@register("gsm8k")
def _gsm8k(n: int, seed: int) -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=seed)
    out: list[dict] = []
    for row in ds.select(range(min(n, len(ds)))):
        out.append(
            {
                "sample_id": _id("gsm", row["question"][:64]),
                "input": row["question"],
                # gsm8k answers are like "72 <newline>#### 72"; strip after ####
                "ground_truth": row["answer"].split("####")[-1].strip(),
            }
        )
    return out


@register("walledai/AdvBench")
def _advbench(n: int, seed: int) -> list[dict]:
    ds = load_dataset("walledai/AdvBench", split="train")
    ds = ds.shuffle(seed=seed)
    out: list[dict] = []
    for row in ds.select(range(min(n, len(ds)))):
        # For AdvBench, ground truth is "refuse" — every prompt is a harmful request.
        out.append(
            {
                "sample_id": _id("adv", row["prompt"][:64]),
                "input": row["prompt"],
                "ground_truth": "REFUSAL",
            }
        )
    return out


@register("uiuc-kang-lab/InjecAgent")
def _injecagent(n: int, seed: int) -> list[dict]:
    # InjecAgent has multiple configs. Use 'test_cases_direct_harm' as default;
    # falls back to loading whatever is available if that split is missing.
    try:
        ds = load_dataset(
            "uiuc-kang-lab/InjecAgent",
            "test_cases_dh_base",
            split="test",
        )
    except Exception:
        # Best-effort fallback
        ds = load_dataset("uiuc-kang-lab/InjecAgent", split="train")
    ds = ds.shuffle(seed=seed)
    out: list[dict] = []
    for row in ds.select(range(min(n, len(ds)))):
        # InjecAgent samples typically have a "User Instruction" and "Attacker Instruction"
        # We treat User Instruction as input; ground truth is that the agent should ignore
        # the Attacker Instruction embedded in the tool output.
        user_instr = row.get("User Instruction") or row.get("user_instruction") or str(row)
        attacker = row.get("Attacker Instruction") or row.get("attacker_instruction") or ""
        out.append(
            {
                "sample_id": _id("inj", str(user_instr)[:64]),
                "input": user_instr,
                "ground_truth": "IGNORE_ATTACKER",
                "meta": {"attacker_instruction": attacker},
            }
        )
    return out
