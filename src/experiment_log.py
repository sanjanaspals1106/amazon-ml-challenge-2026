"""Append-only experiment log (experiments/experiment_log.csv), columns fixed by the team plan (section 16)."""

import csv
from pathlib import Path
from typing import Dict

COLUMNS = [
    "experiment_id", "date", "owner", "blocking", "K", "normalization", "features", "model", "threshold",
    "validation_f05", "precision", "recall", "US_f05", "India_f05", "S2_f05", "S3_f05", "notes",
]


def append_experiment(row: Dict[str, object], path: Path) -> Path:
    """Append one row (creating the file with the header if needed). Unknown keys are rejected."""
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise KeyError(f"Unknown experiment-log columns: {sorted(unknown)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not path.exists()) or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in COLUMNS})
    return path
