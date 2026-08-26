"""Full-spectrum learnability probe for the m64 INTERIOR residual.

Boundary correction (probe_boundary_fullspec.py) took official rl2 from
1.50e-3 to 7.78e-4; the remainder is dominated by the interior residual
(~20% of residual energy, flat white spectrum ~1.2-1.6% per bin).  This
probe asks whether the interior residual is learnable from the full-window
spectrum at all -- same MLP recipe, but the target is the residual at 8
interior points spread across the window.

  - features: full rfft (251, 16) real+imag = 8032 dims, standardized
  - targets:  r at {100,200,250,300,350,400,450,490} (B, 16)
  - metric:   test rl2 on probed points (zero-output baseline = 1.0)
  - decision: <<0.8 => interior learnable => full-window residual net is
    the path; ~1.0 => interior is noise-like => 8ch information limit is
    ~5-7e-4 and 1e-4 is unreachable for the 8ch task definition.

Usage (project .venv, CPU):
  .venv/bin/python scripts/analysis/probe_interior_points.py
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

INTERIOR = [100, 200, 250, 300, 350, 400, 450, 490]


def collect(loaders, split: str, max_n: int) -> tuple:
    Xs, Ys = [], []
    for i, b in enumerate(loaders[split]):
        if i * 64 >= max_n:
            break
        xx = b["input"].double()
        rr = b["target"].double()
        z = torch.fft.rfft(xx, dim=1)
        feats = torch.cat([z.real, z.imag], dim=-1).reshape(len(xx), -1)
        Xs.append(feats)
        Ys.append(rr[:, INTERIOR, :].reshape(len(xx), -1))
    return torch.cat(Xs), torch.cat(Ys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=400)
    args = parser.parse_args()

    data_cfg = {
        "root": "experiments/m63_residual_targets",
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
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-12)
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Ymu, Ysd = Ytr.mean(0), Ytr.std(0).clamp_min(1e-12)
    Ytr = (Ytr - Ymu) / Ysd
    mlp = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], 2048), torch.nn.GELU(),
        torch.nn.Linear(2048, 1024), torch.nn.GELU(),
        torch.nn.Linear(1024, 512), torch.nn.GELU(),
        torch.nn.Linear(512, Ytr.shape[1]),
    ).double()
    opt = torch.optim.AdamW(mlp.parameters(), lr=3e-4, weight_decay=1e-5)
    idx = torch.randperm(Xtr.shape[0], generator=torch.Generator().manual_seed(7))[:3000]
    Xb, Yb = Xtr[idx], Ytr[idx]
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(mlp(Xb), Yb)
        loss.backward()
        opt.step()
        if (ep + 1) % 100 == 0:
            with torch.no_grad():
                p = mlp(Xte) * Ysd + Ymu
                rl2 = torch.sqrt((p - Yte).square().sum() / Yte.square().sum()).item()
            print(f"  ep {ep+1}: train mse {loss.item():.3e}, test interior rl2 {rl2:.4f} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    with torch.no_grad():
        p = mlp(Xte) * Ysd + Ymu
        rl2 = torch.sqrt((p - Yte).square().sum() / Yte.square().sum()).item()
    print(f"[probe] final test rl2 on interior points = {rl2:.4f} "
          f"(zero-output baseline = 1.0)")


if __name__ == "__main__":
    main()
