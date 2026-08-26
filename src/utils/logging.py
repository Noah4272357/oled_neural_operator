"""Experiment logging isolated from training logic."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, TextIO


class ExperimentLogger:
    def __init__(self, run_dir: Optional[Path]) -> None:
        self.run_dir = run_dir
        self._log: Optional[TextIO] = None
        self._metrics: Optional[TextIO] = None
        self._writer: Optional[csv.DictWriter] = None
        if run_dir is not None:
            self._log = (run_dir / "train.log").open("a", encoding="utf-8")
            self._metrics = (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="")

    def log(self, payload: Mapping[str, Any]) -> None:
        line = json.dumps(payload, sort_keys=True)
        print(line, flush=True)
        if self._log is not None:
            self._log.write(line + "\n")
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
            if self._writer is None:
                fieldnames = [
                    "epoch", "learning_rate", "train_mse", "train_rmse",
                    "train_relative_l2", "train_seconds", "validation_mse",
                    "validation_rmse", "validation_relative_l2", "validation_seconds",
                ]
                self._writer = csv.DictWriter(self._metrics, fieldnames=fieldnames)
                self._writer.writeheader()
            self._writer.writerow(row)
            self._metrics.flush()
        self.log(record)

    def close(self) -> None:
        if self._log is not None:
            self._log.close()
        if self._metrics is not None:
            self._metrics.close()

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
