"""Per-gate latency histogram data. Consumed by figures/fig5."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from infoshield.loader import iter_traces


def build(root: str, experiment: str = "main", out_csv: str | Path | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    for t in iter_traces(root, experiment=experiment):
        for g in t.gate_decisions:
            rows.append(
                {
                    "gate_id": g.gate_id,
                    "system": t.system,
                    "attack": t.attack,
                    "risk": g.risk,
                    "status": g.status,
                    "latency_ms": g.latency_ms,
                    "text_len_chars": g.text_len_chars,
                }
            )
    df = pd.DataFrame(rows)
    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df
