"""Prefit the joint per-bin LTI head for the 8ch nonlinear campaign (m62/63).

Fits the per-bin complex least-squares map

    M[k] = argmin_M || z_ft[n,k] M - t_ft[n,k] ||_2^2           (k = 0..250)

where z_ft are the rfft features of the ``diff_features`` input (16ch:
force + encoder_displacement + d1 + d2 of the encoder) and t_ft the rfft of
the disturbance target, accumulated over the whole train split.  This is the
8ch linear information limit (test rl2 ~1.4e-3, residual concentrated at the
window-boundary points, see the campaign diagnostics); the map is frozen as
the head of ``RefinedFNO1d`` (config key ``lti_map_path``) so the FNO trains
on the residual the closed-form head cannot see.

Deterministic: one ordered pass over the train split, no RNG.  Output is a
(251, out_channels, 16) complex128 tensor, saved with ``torch.save``.

Usage (project .venv, CPU):
  .venv/bin/python scripts/analysis/fit_lti_head_16ch.py \
      [--data-root ~/data/neural_operator_3] \
      [--output outputs/lti_head_16ch.pt]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import build_dataloaders


def build_config(data_root: str) -> dict:
    return {
        "root": data_root,
        "input_fields": ["force", "encoder_displacement"],
        "target_fields": ["disturbance"],
        "time_start": 0,
        "time_stop": None,
        "time_stride": 1,
        "preprocessing": {"name": "diff_features", "dt": 0.001, "channels": [4, 5, 6, 7]},
        "memory_cache": False,
        "batch_size": 256,
        "eval_batch_size": 256,
        "num_workers": 0,
        "max_train_samples": None,
        "max_val_samples": None,
        "max_test_samples": None,
    }


def fit_map(loaders, split: str) -> torch.Tensor:
    zHt = None
    xx = None
    for b in loaders[split]:
        zft = torch.fft.rfft(b["input"].double(), dim=1).to(torch.complex128)
        tft = torch.fft.rfft(b["target"].double(), dim=1).to(torch.complex128)
        zf = zft.permute(1, 0, 2)
        tt = tft.permute(1, 0, 2)
        if zHt is None:
            n_bin, in_ch = zf.shape[0], zf.shape[2]
            out_ch = tt.shape[2]
            zHt = torch.zeros((n_bin, in_ch, out_ch), dtype=torch.complex128)
            xx = torch.zeros((n_bin, in_ch, in_ch), dtype=torch.complex128)
        zHt += torch.einsum("kbi,kbj->kij", zf.conj(), tt)
        xx += torch.einsum("kbi,kbj->kij", zf.conj(), zf)
    return torch.linalg.pinv(xx) @ zHt  # (n_bin, in_ch, out_ch)


def evaluate_map(loaders, split: str, mk: torch.Tensor) -> float:
    # mk is (n_bin, in_ch, out_ch); einsum 'bki,kio->bko' contracts in_ch.
    num = torch.zeros((), dtype=torch.float64)
    den = torch.zeros((), dtype=torch.float64)
    for b in loaders[split]:
        zft = torch.fft.rfft(b["input"].double(), dim=1).to(torch.complex128)
        pred = torch.fft.irfft(
            torch.einsum("bki,kio->bko", zft, mk), n=501, dim=1
        )
        t = b["target"].double()
        num += (pred - t).square().sum()
        den += t.square().sum()
    return (num / den).sqrt().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default="~/data/neural_operator_3")
    parser.add_argument("--output", type=Path, default="outputs/lti_head_16ch.pt")
    args = parser.parse_args()

    root = str(Path(args.data_root).expanduser())
    loaders = build_dataloaders(build_config(root), seed=20260810)
    t0 = time.perf_counter()
    mk = fit_map(loaders, "train")
    train_rl2 = evaluate_map(loaders, "train", mk)
    test_rl2 = evaluate_map(loaders, "test", mk)
    print(
        f"[fit_lti_head] closed-form per-bin head: "
        f"train rl2 {train_rl2:.6e}, test rl2 {test_rl2:.6e} "
        f"({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mk.permute(0, 2, 1).contiguous(), out_path)
    print(f"[fit_lti_head] saved {out_path} shape {tuple(mk.shape)}", flush=True)


if __name__ == "__main__":
    main()
