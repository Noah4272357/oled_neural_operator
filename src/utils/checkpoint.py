"""Backward-compatible checkpoint serialization and restoration."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from torch.optim import Optimizer


def _rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    epoch: int,
    validation: Mapping[str, float],
    best_metric: float,
    config: Mapping[str, Any],
    legacy_arguments: Mapping[str, Any],
) -> None:
    torch.save(
        {
            "format_version": 2,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "validation": dict(validation),
            "best_metric": best_metric,
            "config": dict(config),
            # Preserve the key consumed by the existing evaluation programs.
            "args": dict(legacy_arguments),
            "rng_state": _rng_state(),
        },
        path,
    )


def load_checkpoint(path: Path, map_location: Any = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def restore_training_state(
    checkpoint: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: Optional[Optimizer] = None,
) -> int:
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["epoch"])
