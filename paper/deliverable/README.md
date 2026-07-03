# INFOSHIELD deliverable folder

This folder is entirely auto-generated from `results/*.json` by:

```bash
python scripts/aggregate_results.py
python scripts/consolidate.py
```

Nothing here is hand-written. To regenerate after new experiments:

1. Run more samples: `python scripts/run_proof.py` (or any other runner)
2. Aggregate: `python scripts/aggregate_results.py`
3. Consolidate: `python scripts/consolidate.py`

See `RESULTS.md` for a human-readable summary of the current numbers.
