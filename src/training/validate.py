"""Evaluation without optimization."""

from __future__ import annotations

import contextlib
import time
from typing import Dict

import torch
from torch.utils.data import DataLoader

from .factories import LossFunction
from .metrics import RegressionMetrics


@contextlib.contextmanager
def _full_precision():
    # Training runs with TF32 ("high") for throughput; validation/test metrics
    # must reflect full fp32 precision ("highest") or TF32 noise inflates them
    # (~2x on this workload). Restore the previous setting afterwards so the
    # training loop keeps TF32.
    if torch.cuda.is_available() and hasattr(torch, "set_float32_matmul_precision"):
        previous = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        try:
            yield
        finally:
            torch.set_float32_matmul_precision(previous)
    else:
        yield


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: LossFunction,
    device: torch.device,
) -> Dict[str, float]:
    # Criterion is accepted for a stable validation interface and to catch
    # invalid numerical behavior, while reported metrics retain legacy global
    # aggregation semantics.
    model.eval()
    metrics = RegressionMetrics()
    start = time.perf_counter()
    with _full_precision():
        for batch in dataloader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            grid = batch["grid"].to(device, non_blocking=True)
            predictions = model(inputs, grid)
            loss = criterion(predictions, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite validation loss encountered.")
            metrics.update(predictions, targets)
    return metrics.compute(time.perf_counter() - start)
