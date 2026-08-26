"""Factories for optimization components."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW, Optimizer, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    MultiStepLR,
    ReduceLROnPlateau,
)


LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


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
    if name == "spectral_rl2":
        from .spectral_loss import SpectralRelativeL2

        return SpectralRelativeL2(
            bands=list(config.get("bands", [])),
            default=float(config.get("band_default", 1.0)),
        )
    if name == "band_rl2":
        from .spectral_loss import BandRelativeL2

        return BandRelativeL2(
            bands=list(config.get("bands", [])),
            default=float(config.get("band_default", 1.0)),
        )
    raise ValueError(f"Unsupported loss: {name!r}")


def build_optimizer(config: Mapping[str, Any], model: torch.nn.Module) -> Optimizer:
    name = str(config["name"]).lower()
    optimizer_types = {
        "adam": Adam,
        "adamw": AdamW,
        "sgd": SGD,
    }
    try:
        optimizer_type = optimizer_types[name]
    except KeyError as error:
        raise ValueError(f"Unsupported optimizer: {name!r}") from error
    kwargs = {}
    if name == "sgd":
        # M6 pure-spectral recipe: SGD + momentum 0.9 (spike S5a).
        kwargs["momentum"] = float(config.get("momentum", 0.9))
    return optimizer_type(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
        **kwargs,
    )


def build_scheduler(
    config: Mapping[str, Any], optimizer: Optimizer, epochs: int
) -> Any:
    name = str(config["name"]).lower()
    if name == "cosine":
        return CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=float(config["eta_min"])
        )
    if name == "multistep":
        # M6 pure-spectral recipe: lr step decay 0.1 -> 0.01 -> 0.001
        # (Ruling M6-P.2); stepped once per epoch by the trainer, like cosine.
        return MultiStepLR(
            optimizer,
            milestones=[int(milestone) for milestone in config["milestones"]],
            gamma=float(config["gamma"]),
        )
    if name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(config["factor"]),
            patience=int(config["patience"]),
            threshold=float(config["threshold"]),
            threshold_mode=str(config["threshold_mode"]),
            cooldown=int(config["cooldown"]),
            min_lr=float(config["eta_min"]),
        )
    raise ValueError(f"Unsupported scheduler: {name!r}")
