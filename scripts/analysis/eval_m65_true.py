"""True-value evaluation of m65 (head + residual MLP) vs the ORIGINAL
disturbance, with the complex head kept intact.

The old evaluation chain did `torch.load(head).double()`, which on a
complex128 tensor DISCARDS THE IMAGINARY PART (|imag| = 2.7x |real|).
That corrupted the reconstructed dist (denominator) in every metric.
The residual dataset rr was built with the complex head (build_residual_
targets.py keeps it), so dist = rr + irfft(complex head @ rfft(x16)) is
exact to float32 storage precision (verified: diff rms ~0).

Usage:
  .venv/bin/python scripts/analysis/eval_m65_true.py --checkpoint outputs/m65_residual_mlp/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import build_dataloaders
from train_m65_residual_mlp import ResidualNet

N_POINTS, N_BIN = 501, 251

DATA_CFG = {
    "root": "experiments/m63_residual_targets",
    "input_fields": ["force", "encoder_displacement"],
    "target_fields": ["disturbance"],
    "time_start": 0, "time_stop": None, "time_stride": 1,
    "preprocessing": {"name": "diff_features", "dt": 0.001, "channels": [4, 5, 6, 7]},
    "memory_cache": False, "batch_size": 128, "eval_batch_size": 128,
    "num_workers": 0,
    "max_train_samples": None, "max_val_samples": None, "max_test_samples": None,
}


def feats(batch) -> torch.Tensor:
    z = torch.fft.rfft(batch["input"], dim=1)
    return torch.cat([z.real, z.imag], dim=-1).reshape(len(batch["input"]), -1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--head", type=Path, default="outputs/lti_head_16ch.pt")
    args = parser.parse_args()

    loaders = build_dataloaders(DATA_CFG, seed=20260810)
    head = torch.load(args.head, map_location="cpu", weights_only=True)  # complex128, KEEP
    assert head.is_complex() and head.dtype == torch.complex128, head.dtype

    xs, ys = [], []
    for b in loaders["train"]:
        xs.append(feats(b))
        ys.append(b["target"].reshape(-1, N_POINTS * 2))
    Xtr = torch.cat(xs)
    Ytr = torch.cat(ys)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-12)
    Ymu, Ysd = Ytr.mean(0), Ytr.std(0).clamp_min(1e-12)

    model = ResidualNet(Xtr.shape[1], N_POINTS * 2, width=1536)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    num = torch.zeros((), dtype=torch.float64)
    den = torch.zeros((), dtype=torch.float64)
    num_h = torch.zeros((), dtype=torch.float64)
    den_old = torch.zeros((), dtype=torch.float64)   # old (real-head) denominator for comparison
    with torch.no_grad():
        for b in loaders["test"]:
            xx = b["input"].double()
            rr = b["target"].double()
            z = torch.fft.rfft(xx, dim=1)
            hp = torch.fft.irfft(
                torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
                n=N_POINTS, dim=1)
            dist = rr + hp                          # == original disturbance (exact)
            rhat = (model((feats(b) - mu) / sd) * Ysd + Ymu).reshape(
                len(b["input"]), N_POINTS, 2).double()
            pred = hp + rhat
            num += (pred - dist).square().sum()
            den += dist.square().sum()
            num_h += (hp - dist).square().sum()
            # old denominator: real-only head rebuild
            hp_old = torch.fft.irfft(
                torch.einsum("bki,koi->bko", z, head.real.to(torch.complex128)),
                n=N_POINTS, dim=1)
            den_old += (rr + hp_old).square().sum()
    rl2 = torch.sqrt(num / den).item()
    rl2_h = torch.sqrt(num_h / den).item()
    print(f"[m65 TRUE] head+MLP test rl2 = {rl2:.6e} (head-only {rl2_h:.6e})")
    print(f"[m65 TRUE] old (real-head) denominator rl2 = {torch.sqrt(num / den_old).item():.6e} "
          f"(den ratio {torch.sqrt(den_old / den).item():.4f})")


if __name__ == "__main__":
    main()
