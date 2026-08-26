"""Diagnose WHERE the m65/m67 residual r = dist - head(x16) energy lives.

m65/m67a both plateau at ~5.4e-4 (capacity hypothesis falsified: 1536->2048
width changed best by 0.7%).  The residual signal r itself determines the
information limit of the head: r is what the MLP must recover.  Its spectral
structure decides the next design:

  - if r's energy is ~99.9% in tone bins 4-7: head misses a nonlinear tone
    component; the MLP bottleneck is in-band -> needs per-bin nonlinearity
    or higher-order features (tone amplitudes couple to the window FFT).
  - if r's energy is mostly out-of-band (the broadband tail): the in-band
    tone map is nearly perfect and the tail is the target; but the tail is
    already 82-99% linearly predictable, so per-bin LS should have caught
    it -- a contradiction worth measuring directly.

Also reports the per-bin error spectrum of the trained m65 net vs the
per-bin r energy (which bins does the residual MLP fail to recover?).

Usage:
  .venv/bin/python scripts/analysis/diag_residual_spectrum.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import build_dataloaders

N_POINTS = 501
N_BIN = 251

DATA_CFG = {
    "root": "experiments/m63_residual_targets",
    "input_fields": ["force", "encoder_displacement"],
    "target_fields": ["disturbance"],
    "time_start": 0,
    "time_stop": None,
    "time_stride": 1,
    "preprocessing": {"name": "diff_features", "dt": 0.001, "channels": [4, 5, 6, 7]},
    "memory_cache": False,
    "batch_size": 128,
    "eval_batch_size": 128,
    "num_workers": 0,
    "max_train_samples": None,
    "max_val_samples": None,
    "max_test_samples": None,
}


def main() -> None:
    head = torch.load("outputs/lti_head_16ch.pt", map_location="cpu", weights_only=True)
    assert head.shape == (N_BIN, 2, 16), head.shape
    loaders = build_dataloaders(DATA_CFG, seed=20260810)

    # per-bin energy of: dist, r (= target stored in dataset), hp = head(x16)
    Eb_d = torch.zeros(N_BIN, dtype=torch.float64)
    Eb_r = torch.zeros(N_BIN, dtype=torch.float64)
    Eb_h = torch.zeros(N_BIN, dtype=torch.float64)
    n = 0
    for b in loaders["test"]:
        xx = b["input"].double()
        rr = b["target"].double()
        z = torch.fft.rfft(xx, dim=1)
        hp = torch.fft.irfft(
            torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
            n=N_POINTS, dim=1)
        dist = rr + hp
        Df = torch.fft.rfft(dist, dim=1)
        Rf = torch.fft.rfft(rr, dim=1)
        Hf = torch.fft.rfft(hp, dim=1)
        Eb_d += Df.abs().square().sum((0, 2))
        Eb_r += Rf.abs().square().sum((0, 2))
        Eb_h += Hf.abs().square().sum((0, 2))
        n += len(b["input"])
    tot_d = Eb_d.sum().item()
    print(f"[n] test samples {n}")
    print(f"[tot] dist {tot_d:.3e} | r {Eb_r.sum().item():.3e} "
          f"({Eb_r.sum().item()/tot_d:.3e} of dist) | "
          f"hp {Eb_h.sum().item():.3e}")
    print("[per-bin] k : dist_share | r_share(dist) | r_share(r) | hp_share(dist)")
    # first 60 bins + a tail window
    for k in list(range(0, 20)) + [25, 30, 40, 50, 64, 100, 128, 160, 200, 250]:
        sd = Eb_d[k].item() / tot_d
        sr = Eb_r[k].item() / tot_d
        srr = Eb_r[k].item() / Eb_r.sum().item()
        sh = Eb_h[k].item() / tot_d
        print(f"  {k:3d}: {sd:.6e} | {sr:.6e} | {srr:.6e} | {sh:.6e}")
    print(f"[cumulative r] bins 0-3: {Eb_r[:4].sum().item()/Eb_r.sum().item():.3e} | "
          f"bins 4-7: {Eb_r[4:8].sum().item()/Eb_r.sum().item():.3e} | "
          f"bins 8-17: {Eb_r[8:18].sum().item()/Eb_r.sum().item():.3e} | "
          f"bins 18-250: {Eb_r[18:].sum().item()/Eb_r.sum().item():.3e} | "
          f"bins 8-250: {Eb_r[8:].sum().item()/Eb_r.sum().item():.3e}")
    print(f"[cumulative dist] bins 4-7: {Eb_d[4:8].sum().item()/tot_d:.3e} | "
          f"bins 8+: {Eb_d[8:].sum().item()/tot_d:.3e}")


if __name__ == "__main__":
    main()
