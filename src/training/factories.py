"""Optimization components for the test pipeline.

One final recipe: plain SGD on the whitened quadratic, cosine-annealed over the
run.  Nothing here is selected at run time beyond what the test config
sets, so the components are built directly rather than looked up in a registry
of abandoned alternatives.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR


LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

SUPPORTED_LOSSES = ("mse", "relative_l2")
SUPPORTED_OPTIMIZERS = ("sgd",)
SUPPORTED_SCHEDULERS = ("cosine",)


def build_loss(config: Mapping[str, Any]) -> LossFunction:
    name = str(config["name"]).lower()
    if name == "mse":
        return F.mse_loss
    if name == "relative_l2":

        def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            denominator = torch.linalg.vector_norm(target).clamp_min(
                torch.finfo(target.dtype).tiny
            )
            return torch.linalg.vector_norm(prediction - target) / denominator

        return relative_l2
    raise ValueError(f"Unsupported loss {name!r}; expected one of {SUPPORTED_LOSSES}.")


def build_optimizer(config: Mapping[str, Any], model: torch.nn.Module) -> Optimizer:
    name = str(config["name"]).lower()
    if name != "sgd":
        raise ValueError(
            f"Unsupported optimizer {name!r}; this pipeline ships only "
            f"{list(SUPPORTED_OPTIMIZERS)}."
        )
    return SGD(
        model.parameters(),
        lr=float(config["lr"]),
        momentum=float(config.get("momentum", 0.0)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )


def build_scheduler(config: Mapping[str, Any], optimizer: Optimizer, epochs: int) -> Any:
    name = str(config["name"]).lower()
    if name != "cosine":
        raise ValueError(
            f"Unsupported scheduler {name!r}; this pipeline ships only "
            f"{list(SUPPORTED_SCHEDULERS)}."
        )
    return CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(config.get("eta_min", 0.0))
    )
