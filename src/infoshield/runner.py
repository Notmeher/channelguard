"""Shared runner primitives: iterate (system × attack × sample × seed), resume from disk.

Parallelism: within each (system, attack, seed) cell, samples are dispatched
to a ThreadPoolExecutor of size `cfg.pipeline.parallel_samples`. Each pipeline
call is independent (fresh MemoryController, fresh LangGraph closure), so
threading is safe. The OpenAI / Anthropic SDKs are thread-safe.
"""
from __future__ import annotations

import random
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from infoshield.attacks import ATTACKS
from infoshield.baselines import SYSTEMS
from infoshield.config import Config
from infoshield.trace import Trace


def _dest_path(
    root: str | Path, experiment: str, system: str, attack: str, seed: int, sample_id: str
) -> Path:
    return (
        Path(root)
        / experiment
        / system
        / attack
        / f"seed_{seed}"
        / f"{sample_id}.json"
    )


def _run_one(
    *,
    pipeline,
    atk_fn,
    sample: dict,
    experiment: str,
    system_name: str,
    attack_name: str,
    seed: int,
    dataset_name: str,
    output_root: str,
) -> tuple[str, Path | None, Exception | None]:
    """Execute one sample; return (label, path_or_none, exc_or_none)."""
    label = f"{system_name}/{attack_name}/seed_{seed}/{sample['sample_id']}"
    try:
        atk_kwargs = atk_fn(sample["input"], sample.get("meta", {}))
        run_kwargs: dict[str, Any] = {
            "sample_id": sample["sample_id"],
            "ground_truth": sample["ground_truth"],
            "experiment": experiment,
            "system": system_name,
            "attack": attack_name,
            "seed": seed,
            "dataset": dataset_name,
        }
        run_kwargs.update(atk_kwargs)
        trace: Trace = pipeline(**run_kwargs)
        path = trace.write(output_root)
        return label, path, None
    except Exception as e:
        return label, None, e


def run_matrix(
    *,
    cfg: Config,
    experiment: str,
    dataset_name: str,
    samples: list[dict],
    systems: Iterable[str],
    attacks: Iterable[str],
    seeds: Iterable[int],
    resume: bool = True,
    on_error: str = "log",   # 'log' | 'raise'
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, int]:
    """Run every (system, attack, sample, seed) cell.

    Skips cells whose JSON exists (resume-from-checkpoint). Within each
    (system, attack, seed) cell, samples run concurrently on a thread pool
    of size cfg.pipeline.parallel_samples.
    """
    stats = {"done": 0, "skipped": 0, "failed": 0}
    parallel = max(1, cfg.pipeline.parallel_samples)
    print_lock = threading.Lock()

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        for system_name in systems:
            if system_name not in SYSTEMS:
                raise KeyError(f"Unknown system {system_name!r}. Have: {list(SYSTEMS)}")
            pipeline = SYSTEMS[system_name](cfg)
            for attack_name in attacks:
                if attack_name not in ATTACKS:
                    raise KeyError(f"Unknown attack {attack_name!r}. Have: {list(ATTACKS)}")
                atk_fn = ATTACKS[attack_name]

                # Filter out already-completed samples.
                todo: list[dict] = []
                for s in samples:
                    dest = _dest_path(
                        cfg.output_root, experiment, system_name, attack_name, seed, s["sample_id"]
                    )
                    if resume and dest.exists():
                        stats["skipped"] += 1
                    else:
                        todo.append(s)

                if not todo:
                    continue

                # Dispatch this cell's remaining samples in parallel.
                with ThreadPoolExecutor(max_workers=parallel) as ex:
                    futures = [
                        ex.submit(
                            _run_one,
                            pipeline=pipeline,
                            atk_fn=atk_fn,
                            sample=s,
                            experiment=experiment,
                            system_name=system_name,
                            attack_name=attack_name,
                            seed=seed,
                            dataset_name=dataset_name,
                            output_root=cfg.output_root,
                        )
                        for s in todo
                    ]
                    for fut in as_completed(futures):
                        label, _path, exc = fut.result()
                        if exc is None:
                            stats["done"] += 1
                            if progress_cb:
                                with print_lock:
                                    progress_cb(f"[{stats['done']:>5}] {label}")
                        else:
                            stats["failed"] += 1
                            msg = f"[FAIL] {label}: {exc}"
                            if on_error == "raise":
                                traceback.print_exc()
                                raise exc
                            with print_lock:
                                print(msg)
    return stats
