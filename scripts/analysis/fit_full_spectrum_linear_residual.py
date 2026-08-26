"""m67b: closed-form FULL-SPECTRUM linear residual head.

Diagnosis that motivated this (diag_r_predictability.py):
  - r = dist - head(x16) has 88.2% of its energy in bins 8-250 (broadband
    tail + intermodulation), only 4.9% in the tone bins 4-7.
  - per-bin linear R^2 of r_b ~ z_bins(0-100) is >= 0.997 for nearly every
    bin: the residual is almost entirely a LINEAR function of the FULL input
    spectrum (cross-bin coupling), not a per-bin one.
  - head (per-bin LS) cannot express cross-bin coupling -> 1.52e-3.
  - the m65/m67a GELU MLPs recovered only ~87% of r (5.4e-4) despite 27M
    params; the linear map's fitted limit on test is 9.9e-5 < 1e-4.
  - fitted on 4000 train samples, evaluated on test: the linear map is
    therefore a *learning target that beats the target gate*, reachable in
    closed form.

This script fits the closed-form map and reports the official metric:
    pred = head(x16) + r_lin(rfft(x16))        head frozen, W from lstsq
    test rl2 = ||pred - dist|| / ||dist||

Both the fit (train 4000) and the evaluation (full test) run on CPU in
about a minute.  The fitted W is saved for the follow-up training campaign
(m67c: SGD from random init on the same parameterization).

Usage (project .venv, CPU):
  .venv/bin/python scripts/analysis/fit_full_spectrum_linear_residual.py \
      [--fit-bins 100] [--n-samples 4000] [--out outputs/m67b_linear_residual]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-bins", type=int, default=100, metavar="K",
                        help="input feature bins 0..K (default 100 = 0-200 Hz)")
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--head", type=Path, default="outputs/lti_head_16ch.pt")
    parser.add_argument("--out", type=Path, default="outputs/m67b_linear_residual")
    args = parser.parse_args()
    F = args.fit_bins

    t0 = time.perf_counter()
    head = torch.load(args.head, map_location="cpu", weights_only=True)  # complex128
    loaders = build_dataloaders(DATA_CFG, seed=20260810)

    def feats(batch) -> torch.Tensor:
        z = torch.fft.rfft(batch["input"], dim=1)[:, : F + 1, :]
        return torch.cat([z.real, z.imag], dim=-1).reshape(len(batch["input"]), -1)

    # ---- closed-form fit: r_b = W_b @ x, complex lstsq over all bins at once ----
    Xtr, Rtr = [], []
    n = 0
    for b in loaders["train"]:
        Xtr.append(feats(b).double())
        Rtr.append(torch.fft.rfft(b["target"].double(), dim=1))
        n += len(b["input"])
        if n >= args.n_samples:
            break
    X = torch.cat(Xtr)  # (n, (F+1)*16*2)
    R = torch.cat(Rtr)  # (n, 251, 2) complex128
    W = torch.linalg.pinv(X.to(torch.complex128)) @ R.permute(1, 0, 2)
    print(f"[m67b] fit {n} samples, input bins 0-{F} (feat {X.shape[1]}) -> "
          f"W {(N_BIN, 2, X.shape[1])}  ({time.perf_counter()-t0:.0f}s)")

    # ---- official evaluation on test: pred = head + linear residual ----
    num = den = num_h = 0.0
    for b in loaders["test"]:
        xx = b["input"].double()
        rr = b["target"].double()
        z = torch.fft.rfft(xx, dim=1)
        hp = torch.fft.irfft(
            torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
            n=N_POINTS, dim=1)
        dist = rr + hp
        rhat_ft = torch.einsum("bi,kio->bko", feats(b).double().to(torch.complex128), W)
        rhat = torch.fft.irfft(rhat_ft, n=N_POINTS, dim=1)
        pred = hp + rhat
        num += (pred - dist).square().sum().item()
        den += dist.square().sum().item()
        num_h += (hp - dist).square().sum().item()
    rl2 = (num / den) ** 0.5
    rl2_h = (num_h / den) ** 0.5
    print(f"[m67b] test rl2 = {rl2:.6e} (head-only {rl2_h:.6e}) "
          f"({time.perf_counter()-t0:.0f}s)")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"W": W, "fit_bins": F, "n_samples": n, "head": head},
               args.out / "W_full_spectrum.pt")
    with open(args.out / "summary.json", "w") as f:
        json.dump({"test_rl2": rl2, "head_only_rl2": rl2_h, "fit_bins": F,
                   "n_samples": n, "feat_dim": X.shape[1]}, f, indent=1)
    print(f"[m67b] saved {args.out / 'W_full_spectrum.pt'}")


if __name__ == "__main__":
    main()
