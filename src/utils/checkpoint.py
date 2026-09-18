"""Checkpoint serialization.

A checkpoint is self-describing: it carries the resolved config that produced
it, so ``scripts/evaluate.py`` can rebuild the exact model and data pipeline
without being told anything but the file path.  There is no resume machinery --
the acceptance pipeline always trains from a fresh initialization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import torch
from torch.optim import Optimizer


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
) -> None:
    torch.save(
        {
            "format_version": 3,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "validation": dict(validation),
            "best_metric": best_metric,
            "config": dict(config),
        },
        path,
    )


def load_checkpoint(path: Path, map_location: Any = "cpu") -> Dict[str, Any]:
    """Load a checkpoint.

    ``weights_only=True`` is safe here and is the default for this pipeline:
    a checkpoint holds only tensors, numpy-free primitives and the
    JSON-compatible config mapping, so unpickling cannot execute code from the
    file.  Historical campaign checkpoints (``format_version`` 2) additionally
    carried pickled RNG state and are not readable this way -- they are not
    part of the acceptance pipeline.
    """
    return torch.load(path, map_location=map_location, weights_only=True)


def load_model_state(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    state = checkpoint.get("model_state_dict")
    if state is None:
        raise ValueError("Checkpoint has no 'model_state_dict'; it is not a model checkpoint.")
    return dict(state)
