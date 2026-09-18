"""Dense cross-frequency spectral model: a TRAINED map on the flattened spectrum.

The inverse disturbance operator is exactly frequency-independent
(``d = M qdd - B u``), which makes a *shared* complex map the natural guess --
but the model never sees ``qdd``, it sees the encoder channels, and recovering
``qdd`` from them goes through a finite difference whose gain is itself
frequency-dependent.  No single shared matrix can absorb that, so the final
model is a dense map on the flattened per-bin spectrum.

What is fixed and what is trained: ``sd`` (per-column std), ``Vd``/``Sd`` (top-d
right singular vectors and values of the standardized train feature matrix) and
``sy`` (one scalar target std) are computed from the **train split inputs and
targets** and stored as persistent buffers.  They are preprocessing statistics:
no target is ever regressed onto the features to produce them.  The map ``W``
is the model and is learned by gradient descent.

Whitening by ``Sd`` makes the design orthonormal over train, so the loss Hessian
is ~I and first-order SGD converges on what is otherwise a severely
ill-conditioned quadratic (the differenced features' Gram has condition number
~1e15).
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
    ``[Re(c_0..c_{n-1}), Im(c_0..c_{n-1})]``.

    The only trainable parameter is ``W`` of shape ``(d, out_flat)``.
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
        # Per-bin [Re(all channels), Im(all channels)].
        flat = torch.cat([z.real, z.imag], dim=-1).reshape(x.shape[0], -1)
        whitened = ((flat / self.sd) @ self.Vd.t()) / self.Sd
        out = (whitened @ self.W) * self.sy
        per_bin = out.reshape(x.shape[0], self.n_bins, -1)
        half = per_bin.shape[-1] // 2
        spectrum = torch.complex(per_bin[..., :half], per_bin[..., half:])
        return torch.fft.irfft(spectrum, n=self.n_points, dim=1)


class GridAdapter(nn.Module):
    """Adapt the trainer/evaluate call ``model(inputs, grid)`` to ``model(inputs)``.

    Casts inputs to the model's working dtype -- float64 for this model, whose
    parameters and buffers are float64 -- so what the official metric measures
    is exactly what was trained.  The grid argument is ignored.
    """

    def __init__(self, model: nn.Module, in_dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        self.model = model
        self.in_dtype = in_dtype

    def forward(self, inputs: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return self.model(inputs.to(self.in_dtype))
