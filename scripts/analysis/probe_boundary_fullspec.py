"""Full-spectrum learnability probe for the m64 boundary residual.

The local-neighbourhood MLP probe (probe_boundary_learnability.py) scored
0.873 -- boundary residual almost orthogonal to the local window.  But the
disturbance tones {8,10,12,14}Hz have global phase information only visible
in the FULL window spectrum.  This probe gives the nonlinear model the whole
rfft of the 16ch input (251 bins, real+imag = 8032 features) and asks again:
can any model recover the residual at the 8 boundary points?

  - features: full rfft (251, 16) -> (251*16*2,) real+imag, standardized
  - targets:  r at {0,1,2,3,497,498,499,500} (B, 16)
  - model: MLP 8032 -> 2048 -> 1024 -> 512 -> 16, GELU, AdamW
  - metric: test rl2 on probed points (baseline zero-output = 1.0)

Plus the global consequence: with the learned boundary corrections placed on
the 8 boundary points of the linear-head prediction, what is the OFFICIAL
global relative_l2 on the original test split (head-only = 1.52e-3 baseline)?

Usage (project .venv, CPU):
  .venv/bin/python scripts/analysis/probe_boundary_fullspec.py \
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
N_BIN = 251


def collect(loaders, split: str, max_n: int) -> tuple:
    Xs, Ys = [], []
    for i, b in enumerate(loaders[split]):
        if i * 64 >= max_n:
            break
        xx = b["input"].double()  # (B, 501, 16)
        rr = b["target"].double()  # (B, 501, 2)
        z = torch.fft.rfft(xx, dim=1)  # (B, 251, 16) complex128
        feats = torch.cat([z.real, z.imag], dim=-1).reshape(len(xx), -1)  # (B, 8032)
        Xs.append(feats)
        Ys.append(rr[:, BOUNDARY, :].reshape(len(xx), -1))
    return torch.cat(Xs), torch.cat(Ys)


def run_probe(Xtr, Ytr, Xte, Yte, epochs: int):
    """Train the MLP; return (mlp, mu, sd, Ymu, Ysd, test_rl2)."""
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-12)
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Ymu, Ysd = Ytr.mean(0), Ytr.std(0).clamp_min(1e-12)
    Ytr = (Ytr - Ymu) / Ysd
    mlp = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], 2048),
        torch.nn.GELU(),
        torch.nn.Linear(2048, 1024),
        torch.nn.GELU(),
        torch.nn.Linear(1024, 512),
        torch.nn.GELU(),
        torch.nn.Linear(512, Ytr.shape[1]),
    ).double()
    opt = torch.optim.AdamW(mlp.parameters(), lr=3e-4, weight_decay=1e-5)
    idx = torch.randperm(Xtr.shape[0], generator=torch.Generator().manual_seed(7))[:3000]
    Xb, Yb = Xtr[idx], Ytr[idx]
    t0 = time.perf_counter()
    for ep in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(mlp(Xb), Yb)
        loss.backward()
        opt.step()
        if (ep + 1) % 100 == 0:
            with torch.no_grad():
                p = mlp(Xte) * Ysd + Ymu
                rl2 = torch.sqrt((p - Yte).square().sum() / Yte.square().sum()).item()
            print(f"  ep {ep+1}: train mse {loss.item():.3e}, "
                  f"test probe rl2 {rl2:.4f} ({time.perf_counter()-t0:.0f}s)",
                  flush=True)
    with torch.no_grad():
        p = mlp(Xte) * Ysd + Ymu
        rl2 = torch.sqrt((p - Yte).square().sum() / Yte.square().sum()).item()
    print(f"[probe] final test rl2 on boundary points = {rl2:.4f} "
          f"(baseline zero-output = 1.0)")
    return mlp, mu, sd, Ymu, Ysd, rl2


def global_check(loaders, mlp, mu, sd, Ymu, Ysd) -> None:
    """Official global rl2 on the ORIGINAL disturbance target.

    The residual dataset stores target = dist - head(x16) (float32), so
    dist = rr + head_pred with head_pred recomputed here from the same
    lti_head_16ch.pt -- consistent with build_residual_targets.py.
    """
    head = torch.load("outputs/lti_head_16ch.pt", map_location="cpu",
                      weights_only=True).double()
    num = torch.zeros((), dtype=torch.float64)
    den = torch.zeros((), dtype=torch.float64)
    num_head = torch.zeros((), dtype=torch.float64)
    with torch.no_grad():
        for b in loaders["test"]:
            xx = b["input"].double()      # (B, 501, 16)
            rr = b["target"].double()     # residual target (B, 501, 2)
            z = torch.fft.rfft(xx, dim=1)
            head_ft = torch.einsum("bki,koi->bko", z, head.to(torch.complex128))
            hp = torch.fft.irfft(head_ft, n=501, dim=1)      # head prediction
            dist = rr + hp                                   # original target
            feats = torch.cat([z.real, z.imag], dim=-1).reshape(len(xx), -1)
            pred = mlp((feats - mu) / sd) * Ysd + Ymu        # (B, 16)
            rp = torch.zeros_like(rr)
            rp[:, BOUNDARY, :] = pred.reshape(len(xx), 8, 2)
            full = hp + rp
            num += (full - dist).square().sum()
            den += dist.square().sum()
            num_head += (hp - dist).square().sum()
    print(f"[global] official rl2  head-only = {torch.sqrt(num_head / den).item():.6e}")
    print(f"[global] official rl2  head + boundary-corr MLP = {torch.sqrt(num / den).item():.6e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default="experiments/m63_residual_targets")
    parser.add_argument("--epochs", type=int, default=400)
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
    print(f"features: {Xtr.shape[1]} dims, train {Xtr.shape[0]} / test {Xte.shape[0]}")
    mlp, mu, sd, Ymu, Ysd, rl2 = run_probe(Xtr, Ytr, Xte, Yte, args.epochs)
    global_check(loaders, mlp, mu, sd, Ymu, Ysd)


if __name__ == "__main__":
    main()
