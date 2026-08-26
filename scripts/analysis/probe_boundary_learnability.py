"""Nonlinear learnability probe for the m64 boundary residual.

The linear head (fit_lti_head_16ch) is least-squares optimal, so the
residual r = f_dist - head(x16) is *linearly* orthogonal to the input by
construction -- a linear probe on r cannot see anything.  This script asks
the real question: can a *nonlinear* model recover r at the window boundary
points from the in-window input?

  - features: the 16ch diff_features input on the two boundary
    neighbourhoods [0:16] and [485:501] (B, 32, 16) flattened
  - targets: r at the 8 boundary points {0,1,2,3,497,498,499,500} (B, 8, 2)
  - model: 3-layer MLP (512 -> 512 -> 256 -> 16, GELU), AdamW, ~2 min CPU
  - metric: test rl2 on the probed points, vs the zero-prediction baseline
    (rl2 = 1.0).  <<0.5 means the boundary residual is learnable from the
    window (architecture was the problem); ~1.0 means the boundary value
    depends on out-of-window information and no 16ch in-window model can
    reach 1e-4 (task-definition problem).

Usage (project .venv, CPU):
  .venv/bin/python scripts/analysis/probe_boundary_learnability.py \
      --data-root experiments/m63_residual_targets
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import build_dataloaders

BOUNDARY = [0, 1, 2, 3, 497, 498, 499, 500]
EDGE = [0, 1, 2, 3, 15, 14, 13, 12, 485, 486, 487, 488, 500, 499, 498, 497]


def collect(loaders, split: str, max_n: int) -> tuple:
    Xs, Ys = [], []
    for i, b in enumerate(loaders[split]):
        if i * 64 >= max_n:
            break
        xx = b["input"].double()  # (B, 501, 16)
        rr = b["target"].double()  # (B, 501, 2)
        feats = xx[:, EDGE, :]  # (B, 16, 16) ordered as {0,1,2,3,15,14,13,12,485..,500..}
        Xs.append(feats.reshape(len(xx), -1))
        Ys.append(rr[:, BOUNDARY, :].reshape(len(xx), -1))
    X = torch.cat(Xs)
    Y = torch.cat(Ys)
    # standardize features from train stats
    return X, Y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default="experiments/m63_residual_targets")
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    data_cfg = {
        "root": str(args.data_root),
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
    loaders = build_dataloaders(data_cfg, seed=20260810)
    Xtr, Ytr = collect(loaders, "train", 4000)
    Xte, Yte = collect(loaders, "test", 200)

    # Standardize with train stats.
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-12)
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Ymu, Ysd = Ytr.mean(0), Ytr.std(0).clamp_min(1e-12)
    Ytr = (Ytr - Ymu) / Ysd
    n = Xtr.shape[0]

    mlp = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], 512),
        torch.nn.GELU(),
        torch.nn.Linear(512, 256),
        torch.nn.GELU(),
        torch.nn.Linear(256, Ytr.shape[1]),
    ).double()
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-5)

    t0 = time.perf_counter()
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(7))[: 3000]
    Xb, Yb = Xtr[idx], Ytr[idx]
    for ep in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(mlp(Xb), Yb)
        loss.backward()
        opt.step()
        if (ep + 1) % 50 == 0:
            with torch.no_grad():
                p = mlp(Xte) * Ysd + Ymu
                se = (p - Yte).square().sum()
                st = Yte.square().sum()
            print(f"  ep {ep+1}: train mse {loss.item():.3e}, "
                  f"test probe rl2 {torch.sqrt(se/st).item():.4f} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    with torch.no_grad():
        p = mlp(Xte) * Ysd + Ymu
        se = (p - Yte).square().sum()
        st = Yte.square().sum()
        rl2 = torch.sqrt(se / st).item()
    print(f"[probe] final test rl2 on boundary points = {rl2:.4f} "
          f"(baseline zero-output = 1.0; lower is more learnable)")


if __name__ == "__main__":
    main()
