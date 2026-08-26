"""Deterministic preprocessing construction.

The default transforms are the identity: the raw physical values are fed to
the network.  ``second_derivative`` differentiates selected input channels
twice in time, which annihilates the encoder's start-up drift ramp exactly
while keeping the 8-14 Hz disturbance tones at their own DFT bins (see
``scripts/analysis/analyze_dataset_fourier.py`` for the dataset-intrinsic
justification and the measured per-mode least-squares floors).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np
import torch


Sample = Dict[str, Any]


def build_preprocessor(config: Mapping[str, Any]) -> Optional[Callable[[Sample], Sample]]:
    """Build the configured deterministic sample transform."""
    name = str(config.get("name", "none")).lower()
    if name in {"none", "identity"}:
        return None
    if name == "second_derivative":
        return _make_second_derivative(config)
    if name == "diff_features":
        return _make_diff_features(config)
    raise ValueError(f"Unsupported preprocessing method: {name!r}")


def _make_second_derivative(config: Mapping[str, Any]) -> Callable[[Sample], Sample]:
    """Transform differentiating selected input channels twice in time.

    Config keys:
        dt:          sample spacing in seconds (after any time_stride), e.g.
                     0.001 for the default 1 kHz grid.
        channels:    integer channel indices of the concatenated input tensor
                     to differentiate, e.g. [4, 5, 6, 7] for
                     ``encoder_displacement`` following ``force`` (4 ch).

    Uses ``np.gradient`` with ``edge_order=2`` twice: central differences in
    the interior (exact per-bin scaling for discrete tones) and one-sided
    estimates at the edges (exact on linear functions, so the encoder drift
    ramp is annihilated everywhere).
    """
    dt = float(config["dt"])
    if dt <= 0.0:
        raise ValueError("preprocessing dt must be positive.")
    channels = tuple(int(c) for c in config["channels"])
    if not channels:
        raise ValueError("preprocessing channels must be non-empty.")

    def transform(sample: Sample) -> Sample:
        inputs = sample["input"]
        if inputs.ndim != 2:
            raise ValueError(
                f"second_derivative expects [time, channels] inputs, got {inputs.shape}."
            )
        width = inputs.shape[1]
        if any(c < 0 or c >= width for c in channels):
            raise ValueError(
                f"second_derivative channels {channels} out of range for input "
                f"width {width}."
            )
        array = inputs.numpy()
        differentiated = np.gradient(
            np.gradient(array[:, channels], dt, axis=0, edge_order=2),
            dt,
            axis=0,
            edge_order=2,
        )
        result = array.copy()
        result[:, channels] = differentiated
        sample["input"] = torch.as_tensor(result, dtype=inputs.dtype)
        return sample

    return transform


def _make_diff_features(config: Mapping[str, Any]) -> Callable[[Sample], Sample]:
    """Append first and second central-difference features of selected input
    channels, growing the input tensor to ``C + 2 * len(channels)`` channels.

    This is the 8ch nonlinear-campaign feature set (m62): for the encoder
    displacement channels the appended d1/d2 rows are the finite-difference
    acceleration estimates, whose window-boundary error (``q[-1]`` is
    outside the window) is the dominant residual of the per-bin linear map
    (see scripts/analysis/ for the boundary-localization analysis).  The
    stencils here MUST match the closed-form diagnostics:

        3pt second diff  (x[n+1] - 2x[n] + x[n-1]) / dt^2, edges copied from
        the inner stencil (y[0]=y[2], y[-1]=y[-3]);
        3pt first diff   (x[n+1] - x[n-1]) / (2 dt), same edge convention.

    Config keys (same as ``second_derivative``):
        dt:          sample spacing in seconds (after any time_stride), e.g.
                     0.001 for the default 1 kHz grid.
        channels:    integer channel indices of the concatenated input tensor
                     to differentiate, e.g. [4, 5, 6, 7] for
                     ``encoder_displacement`` following ``force`` (4 ch).
    """
    dt = float(config["dt"])
    if dt <= 0.0:
        raise ValueError("preprocessing dt must be positive.")
    channels = tuple(int(c) for c in config["channels"])
    if not channels:
        raise ValueError("preprocessing channels must be non-empty.")

    def transform(sample: Sample) -> Sample:
        inputs = sample["input"]
        if inputs.ndim != 2:
            raise ValueError(
                f"diff_features expects [time, channels] inputs, got {inputs.shape}."
            )
        width = inputs.shape[1]
        if any(c < 0 or c >= width for c in channels):
            raise ValueError(
                f"diff_features channels {channels} out of range for input "
                f"width {width}."
            )
        x = inputs.numpy()[:, channels]
        d1 = np.zeros_like(x)
        d1[1:-1, :] = (x[2:, :] - x[:-2, :]) / (2.0 * dt)
        d1[0, :] = d1[2, :]
        d1[-1, :] = d1[-3, :]
        d2 = np.zeros_like(x)
        d2[1:-1, :] = (x[2:, :] - 2.0 * x[1:-1, :] + x[:-2, :]) / (dt * dt)
        d2[0, :] = d2[2, :]
        d2[-1, :] = d2[-3, :]
        result = np.concatenate([inputs.numpy(), d1, d2], axis=1)
        sample["input"] = torch.as_tensor(result, dtype=inputs.dtype)
        return sample

    return transform
