"""Trace loader: walk results/ and yield Trace objects filtered by experiment/system/attack/seed."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from infoshield.trace import Trace


def iter_traces(
    root: str | Path,
    experiment: str | None = None,
    system: str | None = None,
    attack: str | None = None,
    seed: int | None = None,
) -> Iterator[Trace]:
    """Yield Trace objects matching the (optional) filters."""
    root = Path(root)
    if not root.exists():
        return
    exp_dirs = [root / experiment] if experiment else [p for p in root.iterdir() if p.is_dir()]
    for edir in exp_dirs:
        if not edir.exists():
            continue
        sys_dirs = [edir / system] if system else [p for p in edir.iterdir() if p.is_dir()]
        for sdir in sys_dirs:
            if not sdir.exists():
                continue
            atk_dirs = [sdir / attack] if attack else [p for p in sdir.iterdir() if p.is_dir()]
            for adir in atk_dirs:
                if not adir.exists():
                    continue
                seed_dirs = (
                    [adir / f"seed_{seed}"]
                    if seed is not None
                    else [p for p in adir.iterdir() if p.is_dir() and p.name.startswith("seed_")]
                )
                for seed_dir in seed_dirs:
                    if not seed_dir.exists():
                        continue
                    for jf in seed_dir.glob("*.json"):
                        try:
                            yield Trace.read(jf)
                        except Exception as e:
                            print(f"[warn] failed to read {jf}: {e}")


def collect(*args, **kwargs) -> list[Trace]:
    return list(iter_traces(*args, **kwargs))
