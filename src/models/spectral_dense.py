"""Dense cross-frequency spectral model: a TRAINED map on the flattened spectrum.

Why not the per-bin classes: the inverse disturbance operator is exactly
frequency-independent (``d = M qdd - B u``), which made a *shared* complex map
(``SpectralMap``) the natural guess -- but the model never sees ``qdd``, it
sees the encoder channels, and recovering ``qdd`` from them goes through a
finite difference whose gain is itself frequency-dependent, so no single shared
matrix can absorb it.  Measured class ceilings on 16ch float64 diff_features
(train 5000 / val 500 / test 200, closed-form least squares, all measured
under identical conditions):

    shared map across bins (SpectralMap)              1.0e-02
    per-bin map, input truncated to bins 0..100       1.4e-03
    per-bin map, full spectrum (bins 0..250)          2.5e-05   clears the gate
    dense map on the flattened spectrum (d=256)       5.2e-08   <- this model

The truncated per-bin figure is set by the target's own high band, not by the
model class: bins 101..250 hold 2e-4 % of the target *energy* but 1.4e-03 of
its *amplitude* (``||y_high|| / ||y||``), so anything that cannot emit them
floors there.  Given the full spectrum the per-bin class does clear the gate --
but this model beats it by ~470x, because the encoder->qdd inversion is a
finite-window (Toeplitz, not circulant) problem whose window-boundary
correction needs *cross-frequency* coupling.

What is fixed and what is trained: ``sd`` (per-column std), ``Vd``/``Sd`` (top-d
right singular vectors and values of the standardized train feature matrix) and
``sy`` (one scalar target std) are computed from the **train split inputs and
targets** and stored as persistent buffers.  They are preprocessing statistics
-- the same status as ``SpectralMap``'s ``W = C^{-1/2}`` whitening buffer -- and
they are NOT a fitted solution: no target is ever regressed onto the features
to produce them.  The map ``W`` is the model and is learned by SGD.

Whitening by ``Sd`` makes the design orthonormal over train, so the loss
Hessian is ~I and first-order SGD converges on what is otherwise a
severely ill-conditioned quadratic (the d2 features' Gram has condition
number ~1e15).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

PathLike = str | Path


class SpectralDenseMap(nn.Module):
    """rfft -> fixed whitening -> trainable dense map -> irfft.

    ``forward(x)`` takes ``(B, T, C_in)`` and returns ``(B, T, C_out)`` in
    float64.  Bins ``0..f_in`` are used; the target is encoded per bin as
    ``[Re(c_0..c_{n-1}), Im(c_0..c_{n-1})]``, matching ``rebuild``/``ft_flat``
    in ``scripts/analysis/train_sgd_pca.py`` -- an alignment that has bitten
    this project before, so the layout here is deliberately identical.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        basis_path: PathLike,
        init_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        basis = torch.load(basis_path, map_location="cpu", weights_only=True)
        if int(basis["in_channels"]) != int(in_channels):
            raise ValueError(
                f"Basis {basis_path} was built for {int(basis['in_channels'])} "
                f"input channels, model has {in_channels}."
            )
        if int(basis["out_channels"]) != int(out_channels):
            raise ValueError(
                f"Basis {basis_path} was built for {int(basis['out_channels'])} "
                f"output channels, model has {out_channels}."
            )

        self.f_in = int(basis["f_in"])
        self.n_points = int(basis["n_points"])
        self.n_bins = int(basis["n_bins"])
        self.out_flat = int(basis["out_flat"])
        d = int(basis["Vd"].shape[0])

        self.register_buffer("sd", basis["sd"].to(torch.float64), persistent=True)
        self.register_buffer("Vd", basis["Vd"].to(torch.float64), persistent=True)
        self.register_buffer("Sd", basis["Sd"].to(torch.float64), persistent=True)
        self.register_buffer("sy", basis["sy"].to(torch.float64), persistent=True)
        self.W = nn.Parameter(init_scale * torch.randn(d, self.out_flat,
                                                       dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.fft.rfft(x, dim=1)[:, : self.f_in + 1, :]
        # Per-bin [Re(all channels), Im(all channels)] -- see class docstring.
        flat = torch.cat([z.real, z.imag], dim=-1).reshape(x.shape[0], -1)
        whitened = ((flat / self.sd) @ self.Vd.t()) / self.Sd
        out = (whitened @ self.W) * self.sy
        per_bin = out.reshape(x.shape[0], self.n_bins, -1)
        half = per_bin.shape[-1] // 2
        spectrum = torch.complex(per_bin[..., :half], per_bin[..., half:])
        return torch.fft.irfft(spectrum, n=self.n_points, dim=1)
