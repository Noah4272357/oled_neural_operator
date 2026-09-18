"""Model construction for the acceptance pipeline.

One architecture: the dense cross-frequency spectral map.  The factory exists
so that the training and evaluation entry points stay free of model wiring and
so the basis/model channel contract is checked in exactly one place.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from .spectral_dense import GridAdapter, SpectralDenseMap

SUPPORTED_MODELS = ("spectral_dense",)


def build_model(
    config: Mapping[str, Any], input_channels: int, target_channels: int
) -> nn.Module:
    """Build the final model from a resolved config.

    ``config`` carries ``name``, ``basis_path`` and ``init_scale``; the basis
    file must have been fitted on the train split of the same dataset with the
    same preprocessing, and its channel counts are validated against the
    dataset's in :class:`SpectralDenseMap`.
    """
    name = str(config.get("name", "spectral_dense")).lower()
    if name != "spectral_dense":
        raise ValueError(
            f"Unsupported model {name!r}; this pipeline ships only "
            f"{list(SUPPORTED_MODELS)}."
        )
    if not config.get("basis_path"):
        raise ValueError(
            "model.basis_path is required: fit the train-only basis with "
            "scripts/fit_basis.py before building the model."
        )
    return GridAdapter(
        SpectralDenseMap(
            in_channels=input_channels,
            out_channels=target_channels,
            basis_path=str(config["basis_path"]),
            init_scale=float(config.get("init_scale", 1e-3)),
        ),
        in_dtype=torch.float64,
    )
