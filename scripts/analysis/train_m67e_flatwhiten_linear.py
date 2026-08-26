"""m67e: full-spectrum linear residual head, SGD from scratch, GLOBAL whitening.

m67d (per-pair 16ch whitening, the M6 recipe) did NOT fix the frozen
convergence (test rl2 1.46e-3 @500ep vs closed-form ceiling 9.91e-5).
Reason: the M6 spectral map is per-bin (shared 2x13 complex M) so its
Hessian is conditioned by the within-bin 16x16 covariance -- whitened by
C^{-1/2}.  m67c/m67d need CROSS-bin combinations (output bin b uses input
bins 0..100), and the flattened 3232-dim features are strongly correlated
ACROSS bins (smooth broadband spectra, tone-energy coupling between bins
4-7, intermodulation bands).  Full-batch GD convergence rate is set by
the global feature covariance condition number; within-bin whitening does
not touch the cross-bin part.

Fix: whiten the ENTIRE flattened feature vector (3232x3232 real Gram),
which is still a fixed invertible linear data transform (same legal class
as standardization / the M6 whiten buffer) and does not change the
information ceiling of the parameterization (9.91e-5).  After it the
Linear Hessian is ~ identity and SGD converges in tens of epochs.

  z = (X - mu) / sd            column-standardized features
  C = (1/N) z^T z              global Gram
  W = C^{-1/2} (eigh, clip)    fixed preconditioning buffer
  train: SGD momentum 0.9, freq-domain MSE, Linear(3232 -> 1004)

Usage (project .venv, CPU):
  .venv/bin/python scripts/analysis/train_m67e_flatwhiten_linear.py \
      [--epochs 600] [--lr 1e-2] [--fit-bins 100] [--out outputs/m67e_linear_sgd]
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
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--fit-bins", type=int, default=100, metavar="K",
                        help="input feature bins 0..K (default 100 = 0-200 Hz)")
    parser.add_argument("--head", type=Path, default="outputs/lti_head_16ch.pt")
    parser.add_argument("--out", type=Path, default="outputs/m67e_linear_sgd")
    args = parser.parse_args()
    F = args.fit_bins

    torch.manual_seed(20260810)
    loaders = build_dataloaders(DATA_CFG, seed=20260810)
    head = torch.load(args.head, map_location="cpu", weights_only=True)  # complex128

    def feats(batch) -> torch.Tensor:
        z = torch.fft.rfft(batch["input"], dim=1)[:, : F + 1, :]
        return torch.cat([z.real, z.imag], dim=-1).reshape(
            len(batch["input"]), -1).to(torch.float64)

    def ft_targets(batch) -> torch.Tensor:
        z = torch.fft.rfft(batch["target"], dim=1)
        return torch.cat([z.real, z.imag], dim=-1).reshape(len(batch["input"]), -1)

    xs, ys = [], []
    for b in loaders["train"]:
        xs.append(feats(b))
        ys.append(ft_targets(b))
    Xr = torch.cat(xs)   # (5000, 3232) float64
    Ytr = torch.cat(ys)
    mu, sd = Xr.mean(0), Xr.std(0).clamp_min(1e-12)
    Xs = (Xr - mu) / sd

    # ---- global whitening of the flattened features ----
    n = Xs.shape[0]
    C = Xs.T @ Xs / n
    evals, evecs = torch.linalg.eigh(C)
    evals = evals.clamp_min(evals.max().item() * 1e-10)
    W = (evecs * (1.0 / evals.sqrt())) @ evecs.T
    Xtr = (Xs @ W).float()
    print(f"[m67e] global whitening: feat {Xtr.shape}, evals "
          f"{evals.min().item():.3e}..{evals.max().item():.3e} "
          f"(cond {evals.max().item()/evals.max().item()*1e10:.1e})", flush=True)

    Ymu, Ysd = Ytr.mean(0), Ytr.std(0).clamp_min(1e-12)
    Ytr = (Ytr - Ymu) / Ysd
    print(f"[m67e] train features {Xtr.shape} -> freq targets {Ytr.shape} "
          f"(globally whitened, input bins 0-{F})", flush=True)

    model = torch.nn.Linear(Xtr.shape[1], Ytr.shape[1])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[m67e] params {n_params} (linear)", flush=True)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=1e-5)

    def apply_prep(batch) -> torch.Tensor:
        return ((feats(batch) - mu) / sd) @ W

    t0 = time.perf_counter()
    best = {"rl2": float("inf"), "epoch": -1}
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(Xtr), Ytr)
        loss.backward()
        opt.step()
        sched.step()
        if (ep + 1) % 20 == 0:
            with torch.no_grad():
                model.eval()
                num = torch.zeros((), dtype=torch.float64)
                den = torch.zeros((), dtype=torch.float64)
                num_h = torch.zeros((), dtype=torch.float64)
                for b in loaders["test"]:
                    xx = b["input"].double()
                    rr = b["target"].double()
                    z = torch.fft.rfft(xx, dim=1)
                    hp = torch.fft.irfft(
                        torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
                        n=N_POINTS, dim=1)
                    dist = rr + hp
                    rhat_ft = (model(apply_prep(b).float()) * Ysd + Ymu).reshape(
                        len(b["input"]), N_BIN, 2, 2)
                    rhat = torch.fft.irfft(
                        torch.complex(rhat_ft[..., 0].double(), rhat_ft[..., 1].double()),
                        n=N_POINTS, dim=1)
                    pred = hp + rhat
                    num += (pred - dist).square().sum()
                    den += dist.square().sum()
                    num_h += (hp - dist).square().sum()
                rl2 = torch.sqrt(num / den).item()
                rl2_h = torch.sqrt(num_h / den).item()
                if rl2 < best["rl2"]:
                    best = {"rl2": rl2, "epoch": ep + 1}
                    args.out.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), args.out / "best.pt")
                    torch.save({"mu": mu, "sd": sd, "W": W, "Ymu": Ymu, "Ysd": Ysd},
                               args.out / "prep.pt")
                print(f"  ep {ep+1}: train mse {loss.item():.3e} | "
                      f"test rl2 = {rl2:.6e} (head-only {rl2_h:.6e}) "
                      f"[best {best['rl2']:.6e} @ep{best['epoch']}] "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"[m67e] done: best official test rl2 = {best['rl2']:.6e} at epoch "
          f"{best['epoch']} | target 1e-4 {'PASS' if best['rl2'] <= 1e-4 else 'FAIL'}")
    with open(args.out / "summary.json", "w") as f:
        json.dump({"best_rl2": best["rl2"], "best_epoch": best["epoch"],
                   "params": n_params, "epochs": args.epochs, "lr": args.lr,
                   "fit_bins": F, "global_whiten": True}, f, indent=1)


if __name__ == "__main__":
    main()
