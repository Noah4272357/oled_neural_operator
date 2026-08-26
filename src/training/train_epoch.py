"""One epoch of model optimization."""

from __future__ import annotations

import contextlib
import time
from typing import Dict, Optional

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .factories import LossFunction
from .metrics import RegressionMetrics


_AMP_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def _autocast_context(amp: Optional[str], device: torch.device):
    """Return an autocast context when AMP is enabled on a CUDA device."""
    dtype = _AMP_DTYPES.get(amp) if amp else None
    if dtype is None or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=dtype)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: LossFunction,
    device: torch.device,
    grad_clip: Optional[float] = None,
    amp: Optional[str] = None,
) -> Dict[str, float]:
    model.train()
    metrics = RegressionMetrics()
    start = time.perf_counter()
    autocast = _autocast_context(amp, device)
    for batch in dataloader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        grid = batch["grid"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast:
            predictions = model(inputs, grid)
            loss = criterion(predictions, targets)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        metrics.update(predictions, targets)
    return metrics.compute(time.perf_counter() - start)
