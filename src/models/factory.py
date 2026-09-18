"""Model factory decoupling orchestration from concrete architectures."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from .fno import FNO1d, RefinedFNO1d
from .spectral import GridAdapter, SpectralMap
from .spectral_dense import SpectralDenseMap


def build_model(
    config: Mapping[str, Any], input_channels: int, target_channels: int
) -> nn.Module:
    name = str(config.get("name", "fno1d")).lower()
    if name == "spectral":
        # M6 pure-spectral model: shared complex-linear map over rfft bins
        # with a fixed whitening buffer (scripts/analysis/spectral_whiten.py).
        # The branch reads only its own keys -- the FNO hyperparameters
        # (embed_dim/modes/width/...) are not required here.
        return GridAdapter(
            SpectralMap(
                in_channels=input_channels,
                out_channels=target_channels,
                whiten_path=str(config["whiten_path"]),
            )
        )
    if name == "spectral_dense":
        # Dense cross-frequency map on the flattened spectrum.  Class ceilings
        # (16ch float64, same conditions): shared 1.0e-02, per-bin on a
        # truncated spectrum 1.4e-03, per-bin full spectrum 2.5e-05, dense
        # 5.2e-08.  The per-bin class does clear 1e-4 given the full spectrum;
        # the dense class wins by ~470x on the boundary correction.
        # in_dtype is explicit: W is real, so GridAdapter cannot infer it.
        return GridAdapter(
            SpectralDenseMap(
                in_channels=input_channels,
                out_channels=target_channels,
                basis_path=str(config["basis_path"]),
                init_scale=float(config.get("init_scale", 1e-3)),
            ),
            in_dtype=torch.float64,
        )
    common = dict(
        in_channels=input_channels + 1,
        out_channels=target_channels,
        embed_dim=int(config["embed_dim"]),
        modes=int(config["modes"]),
        width=int(config["width"]),
        lift_dim=int(config["lift_dim"]),
        num_blocks=int(config["num_blocks"]),
        activation=str(config.get("activation", "gelu")),
    )
    if name == "fno1d":
        return FNO1d(**common)
    if name == "fno1d_refined":
        # M5-3: trained FNO1d + closed-form joint-LTI residual head.  The
        # checkpoint config carries model.name = "fno1d_refined" so the
        # unmodified evaluate.py builds the wrapper and loads its state_dict
        # (base keys + the lti_map buffer) with the same strict protocol.
        return RefinedFNO1d(**common, refine_n_bin=int(config.get("refine_n_bin", 251)))
    raise ValueError(f"Unsupported model: {name!r}")
