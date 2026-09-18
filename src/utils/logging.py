"""Run artifacts isolated from training logic.

Every record is appended to ``train.log`` (JSON lines) and the per-epoch metric
table to ``metrics.csv``.  Nothing is printed here: the terminal belongs to the
callers, which emit the test-facing blocks.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, TextIO

METRIC_COLUMNS = (
    "epoch",
    "learning_rate",
    "train_mse",
    "train_rmse",
    "train_relative_l2",
    "train_seconds",
    "validation_mse",
    "validation_rmse",
    "validation_relative_l2",
    "validation_seconds",
)


class RunLogger:
    """Append-only writer for ``train.log`` and ``metrics.csv``."""

    def __init__(self, run_dir: Optional[Path]) -> None:
        self.run_dir = run_dir
        self._log: Optional[TextIO] = None
        self._metrics: Optional[TextIO] = None
        self._writer: Optional[csv.DictWriter] = None
        self._wrote_header = False
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            self._log = (run_dir / "train.log").open("a", encoding="utf-8")
            self._metrics = (run_dir / "metrics.csv").open(
                "w", encoding="utf-8", newline=""
            )

    def log(self, payload: Mapping[str, Any]) -> None:
        if self._log is not None:
            self._log.write(json.dumps(payload, sort_keys=True) + "\n")
            self._log.flush()

    def log_metrics(self, record: Mapping[str, Any]) -> None:
        row: Dict[str, Any] = {
            "epoch": record["epoch"],
            "learning_rate": record["learning_rate"],
        }
        for phase in ("train", "validation"):
            for key, value in record.get(phase, {}).items():
                row[f"{phase}_{key}"] = value
        if self._metrics is not None:
            if not self._wrote_header:
                self._writer = csv.DictWriter(self._metrics, fieldnames=METRIC_COLUMNS)
                self._writer.writeheader()
                self._wrote_header = True
            assert self._writer is not None
            self._writer.writerow(row)
            self._metrics.flush()
        self.log(record)

    def close(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None
        if self._metrics is not None:
            self._metrics.close()
            self._metrics = None

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
