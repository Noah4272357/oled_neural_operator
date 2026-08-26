"""Streaming aggregate regression metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class RegressionMetrics:
    """Streaming aggregation with GPU-async accumulators.

    Accumulators stay on-device (no per-step ``.item()`` synchronization)
    so small-batch training keeps the GPU pipeline unbroken; ``compute``
    performs the single final sync. ``element_count`` is a Python int.
    """

    squared_error: float = 0.0
    squared_target: float = 0.0
    element_count: int = 0
    _squared_error_tensor: Optional[torch.Tensor] = None
    _squared_target_tensor: Optional[torch.Tensor] = None

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        # Promote to float32 for metric stability across AMP dtypes (bf16
        # training still reports fp32-equivalent metrics).
        difference = (prediction.detach().float() - target.float())
        if self._squared_error_tensor is None:
            self._squared_error_tensor = difference.square().sum()
            self._squared_target_tensor = target.float().square().sum()
        else:
            self._squared_error_tensor += difference.square().sum()
            self._squared_target_tensor += target.float().square().sum()
        self.element_count += target.numel()

    def _snapshot(self) -> Tuple[float, float]:
        if self._squared_error_tensor is None:
            return 0.0, 0.0
        error, target = self._squared_error_tensor.detach(), self._squared_target_tensor.detach()
        if error.is_cuda:
            # Single sync point at epoch end; both values from one event.
            error_cpu, target_cpu = error.cpu(), target.cpu()
            return float(error_cpu), float(target_cpu)
        return float(error), float(target)

    def compute(self, seconds: float) -> Dict[str, float]:
        squared_error, squared_target = self._snapshot()
        mse = squared_error / max(self.element_count, 1)
        return {
            "mse": mse,
            "rmse": math.sqrt(mse),
            "relative_l2": math.sqrt(
                squared_error / max(squared_target, 1e-30)
            ),
            "seconds": seconds,
        }
