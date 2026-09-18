"""One epoch of model optimization."""

from __future__ import annotations

import time
from typing import Dict, Optional

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .factories import LossFunction
from .metrics import RegressionMetrics


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: LossFunction,
    device: torch.device,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    """Run one full pass over the training split with gradient descent."""
    model.train()
    metrics = RegressionMetrics()
    start = time.perf_counter()
    for batch in dataloader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        grid = batch["grid"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs, grid)
        loss = criterion(predictions, targets)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        metrics.update(predictions, targets)
    return metrics.compute(time.perf_counter() - start)
