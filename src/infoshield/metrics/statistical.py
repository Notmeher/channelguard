"""Statistical helpers: bootstrap CIs, paired significance tests."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the given statistic."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    boots = np.array(
        [statistic(rng.choice(arr, size=arr.size, replace=True)) for _ in range(n_resamples)]
    )
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (lo, hi)


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 5000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap: p-value that mean(a) != mean(b) given per-sample paired values.

    Returns {'diff': mean(a) - mean(b), 'ci_low': ..., 'ci_high': ..., 'p_value': ...}.
    """
    aa, bb = np.asarray(list(a)), np.asarray(list(b))
    if aa.size == 0 or aa.size != bb.size:
        return {"diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0}
    rng = np.random.default_rng(seed)
    diffs = aa - bb
    observed = float(diffs.mean())
    boots = np.array(
        [rng.choice(diffs, size=diffs.size, replace=True).mean() for _ in range(n_resamples)]
    )
    p = float(2 * min(np.mean(boots >= 0), np.mean(boots <= 0)))
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    return {"diff": observed, "ci_low": lo, "ci_high": hi, "p_value": p}


def cohens_kappa(labels_a: Sequence[int], labels_b: Sequence[int]) -> float:
    """Two-annotator Cohen's kappa for binary or categorical labels."""
    a, b = np.asarray(list(labels_a)), np.asarray(list(labels_b))
    if a.size == 0:
        return 0.0
    categories = sorted(set(a.tolist()) | set(b.tolist()))
    if len(categories) < 2:
        return 1.0 if np.array_equal(a, b) else 0.0
    n = a.size
    po = float(np.mean(a == b))
    pe = 0.0
    for c in categories:
        p_a = np.mean(a == c)
        p_b = np.mean(b == c)
        pe += p_a * p_b
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)
