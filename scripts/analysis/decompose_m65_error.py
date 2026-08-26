"""Decompose the m65 residual-MLP error: where does the remaining rl2 live?

Splits the official test error of (head + m65 MLP) into:
  - boundary-8 points vs interior (point groups)
  - spectrum: per-bin error energy vs per-bin target energy (which bands
    still carry error after the MLP?)
This decides the next move: if the error is still boundary-dominated the
boundary net needs more capacity; if interior/white, the interior needs a
different treatment.

Usage:
  .venv/bin/python scripts/analysis/decompose_m65_error.py \
      --checkpoint outputs/m65_residual_mlp/best.pt
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

BOUNDARY = [0, 1, 2, 3, 497, 498, 499, 500]
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--head", type=Path, default="outputs/lti_head_16ch.pt")
    args = parser.parse_args()

    # rebuild the same standardized features as training
    loaders = build_dataloaders(DATA_CFG, seed=20260810)
    head = torch.load(args.head, map_location="cpu", weights_only=True)  # complex128, keep

    def feats(batch) -> torch.Tensor:
        z = torch.fft.rfft(batch["input"], dim=1)
        return torch.cat([z.real, z.imag], dim=-1).reshape(len(batch["input"]), -1)

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

    err = torch.zeros((N_POINTS, 2), dtype=torch.float64)
    tgt = torch.zeros((N_POINTS, 2), dtype=torch.float64)
    rsh = torch.zeros((N_POINTS, 2), dtype=torch.float64)   # residual signal energy
    with torch.no_grad():
        for b in loaders["test"]:
            xx = b["input"].double()
            rr = b["target"].double()
            z = torch.fft.rfft(xx, dim=1)
            hp = torch.fft.irfft(
                torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
                n=N_POINTS, dim=1,
            )
            dist = rr + hp
            rhat = (model((feats(b) - mu) / sd) * Ysd + Ymu).reshape(
                len(b["input"]), N_POINTS, 2).double()
            pred = hp + rhat
            err += (pred - dist).square().mean(0)
            tgt += dist.square().mean(0)
            rsh += rr.square().mean(0)

    g = [("boundary-8", BOUNDARY),
         ("interior", [t for t in range(N_POINTS) if t not in BOUNDARY]),
         ("edge2 (8-11,490-497)", list(range(8, 12)) + list(range(490, 497)))]
    for name, idx in g:
        e = err[idx].sum().item()
        tt = tgt[idx].sum().item()
        print(f"[{name}] error energy {e:.6e} | target energy {tt:.6e} | "
              f"point rl2 {torch.sqrt(err[idx].sum() / tgt[idx].sum()).item():.4f}")
    # spectrum of error vs target (mean over channels, test)
    ef = torch.fft.rfft(err.mean(1), dim=0).abs()
    tf = torch.fft.rfft(tgt.mean(1), dim=0).abs()
    print("[spectrum] per-bin error/target ratio (top-8 bins):")
    ratio = ef / tf.clamp_min(1e-30)
    for k in ratio.topk(8).indices.tolist():
        print(f"    bin {k}: ratio {ratio[k].item():.3f}")
    print(f"[spectrum] ratio mean over bins 0-250 = {ratio.mean().item():.3f}")
    print(f"[residual energy share] r-target {rsh.sum().item():.3e} vs "
          f"dist {tgt.sum().item():.3e} (r/t = {rsh.sum().item()/tgt.sum().item():.4f})")


if __name__ == "__main__":
    main()
