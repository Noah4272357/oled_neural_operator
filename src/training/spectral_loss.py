"""Frequency-domain relative-L2 losses with per-band weights.

Two flavors:
- SpectralRelativeL2: weighted-norm ratio in the rfft domain (absolute-energy
  weighting; failed in every configuration tried).
- BandRelativeL2: per-band relative L2, each band normalized by its *own*
  target energy. This balances gradient allocation across bands regardless
  of absolute target energy — the fix for structural underfitting of
  low-energy high-frequency bands under global relative L2.
"""

from __future__ import annotations

import torch


def _band_weights(n_freq: int, bands, default: float = 1.0) -> torch.Tensor:
    """Build a (n_freq,) weight vector from (lo_hz, hi_hz, weight) tuples.

    Band boundaries are half-open: [lo_hz, hi_hz). With no bands the weights
    are all ones, making the loss Parseval-equivalent to the global
    time-domain relative L2.
    """
    weights = torch.full((n_freq,), float(default))
    for lo_hz, hi_hz, weight in bands:
        lo_bin = int(round(lo_hz * n_freq * 2 / 1000))  # rfft bin per 1000Hz
        hi_bin = min(n_freq, int(round(hi_hz * n_freq * 2 / 1000)))
        weights[lo_bin:hi_bin] = float(weight)
    return weights


class SpectralRelativeL2(torch.nn.Module):
    """Relative L2 in the rfft domain along the time axis (dim=1).

    pred/target: (batch, time, channels). Signals are demeaned per-sample
    along time before the transform (the DC bin carries no useful energy for
    this task and would otherwise dominate).

    weights: list of (lo_hz, hi_hz, weight) tuples; default weight 1.0.
    Uniform weights reproduce the global relative L2 (Parseval).
    """

    def __init__(self, bands=(), default: float = 1.0) -> None:
        super().__init__()
        self.bands = list(bands)
        self.default = float(default)
        self._weights: torch.Tensor | None = None
        self._n_freq: int | None = None

    def _ensure_weights(self, n_freq: int, device: torch.device) -> torch.Tensor:
        if self._weights is None or self._n_freq != n_freq:
            self._weights = _band_weights(n_freq, self.bands, self.default)
            self._n_freq = n_freq
        return self._weights.to(device)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = prediction - prediction.mean(dim=1, keepdim=True)
        t = target - target.mean(dim=1, keepdim=True)
        n_freq = p.size(1) // 2 + 1
        weights = self._ensure_weights(n_freq, prediction.device).unsqueeze(-1)
        p_ft = torch.fft.rfft(p, dim=1)
        t_ft = torch.fft.rfft(t, dim=1)
        err = torch.sqrt((weights * (p_ft - t_ft).abs().square()).sum())
        denominator = torch.sqrt((weights * t_ft.abs().square()).sum()).clamp_min(
            torch.finfo(t_ft.real.dtype).tiny
        )
        return err / denominator


def _freq_bins(lo_hz: float, hi_hz: float, n_freq: int, sample_rate: float = 1000.0):
    """(lo, hi) rfft bin indices for a half-open [lo_hz, hi_hz) band."""
    lo_bin = int(round(lo_hz * n_freq * 2 / sample_rate))
    hi_bin = min(n_freq, int(round(hi_hz * n_freq * 2 / sample_rate)))
    return lo_bin, hi_bin


class BandRelativeL2(torch.nn.Module):
    """Per-band relative L2 along the time axis (dim=1).

    loss = sum_band w_b * ||(p - t)_band|| / ||t_band||

    Each band's error is normalized by that band's own target energy, so a
    band holding 0.004% of target energy (e.g. 18-250 Hz here) contributes
    gradient on equal footing with the dominant bands. With a single
    [0, Nyquist] band this reduces to the global relative L2 (Parseval).

    Signals are NOT demeaned: the DC bin (0 Hz) is included in the first
    band, so any DC bias in the prediction inflates that band's error while
    the target's own (small) DC energy sits in the denominator — DC is
    penalized automatically. (Demeaning would remove DC entirely and leave
    the model free to output an arbitrary bias.)

    bands: list of (lo_hz, hi_hz, weight) tuples; default weight 1.0.
    """

    def __init__(self, bands=(), default: float = 1.0) -> None:
        super().__init__()
        if not bands:
            bands = [(0.0, 1000.0, 1.0)]
        self.bands = [(float(lo), float(hi), float(w)) for lo, hi, w in bands]
        self.default = float(default)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_freq = prediction.size(1) // 2 + 1
        p_ft = torch.fft.rfft(prediction, dim=1)
        t_ft = torch.fft.rfft(target, dim=1)
        err_e = (p_ft - t_ft).abs().square()
        tgt_e = t_ft.abs().square()
        tiny = torch.finfo(t_ft.real.dtype).tiny
        total = torch.zeros((), device=p_ft.device, dtype=p_ft.real.dtype)
        for lo_hz, hi_hz, weight in self.bands:
            lo_bin, hi_bin = _freq_bins(lo_hz, hi_hz, n_freq)
            err_b = torch.sqrt(err_e[:, lo_bin:hi_bin].sum())
            tgt_b = torch.sqrt(tgt_e[:, lo_bin:hi_bin].sum()).clamp_min(tiny)
            total = total + weight * err_b / tgt_b
        return total
