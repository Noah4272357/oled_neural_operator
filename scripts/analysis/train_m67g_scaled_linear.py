"""m67g: full-spectrum linear residual head, AdamW, GLOBALLY scaled targets.

Controlled follow-up to m67f.  Failure mode found there: Y per-unit
standardization (Ymu/Ysd) AMPLIFIES the ~502 near-zero-energy output
units of bins 101-250 into unit-variance pure-noise targets; the convex
optimum of that weighted loss overfits train noise -> train mse 0.06,
test rl2 1.45e-3 (vs closed-form 9.91e-5 on the raw-scale loss).

Fix: the convex optimum is invariant only under UNIFORM target scaling.
Scale the raw frequency-domain target by ONE global scalar s = 1/rms so
the loss is O(1) and AdamW's eps is harmless; no per-unit whitening.
Weight of every unit stays equal -> optimum == the closed-form lstsq
solution (test rl2 9.91e-5).

  X: column-standardized (feature preconditioning, does not move the
     linear optimum)
  Y: raw rfft(r) flattened x global scalar s   (uniform, invariant)
  loss: plain MSE  | model: Linear(3232 -> 1004)  | optimizer: AdamW

Usage (project .venv, CPU):
  .venv/bin/python scripts/analysis/train_m67g_scaled_linear.py \
      [--epochs 600] [--lr 3e-3] [--fit-bins 100] [--out outputs/m67g_adamw_scaled]
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
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--fit-bins", type=int, default=100, metavar="K",
                        help="input feature bins 0..K (default 100 = 0-200 Hz)")
    parser.add_argument("--head", type=Path, default="outputs/lti_head_16ch.pt")
    parser.add_argument("--out", type=Path, default="outputs/m67g_adamw_scaled")
    args = parser.parse_args()
    F = args.fit_bins

    torch.manual_seed(20260810)
    loaders = build_dataloaders(DATA_CFG, seed=20260810)
    head = torch.load(args.head, map_location="cpu", weights_only=True)  # complex128

    def feats(batch) -> torch.Tensor:
        z = torch.fft.rfft(batch["input"], dim=1)[:, : F + 1, :]
        return torch.cat([z.real, z.imag], dim=-1).reshape(len(batch["input"]), -1)

    def ft_targets(batch) -> torch.Tensor:
        z = torch.fft.rfft(batch["target"], dim=1)
        return torch.cat([z.real, z.imag], dim=-1).reshape(len(batch["input"]), -1)

    xs, ys = [], []
    for b in loaders["train"]:
        xs.append(feats(b))
        ys.append(ft_targets(b))
    Xtr = torch.cat(xs)
    Ytr = torch.cat(ys)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-12)
    Xtr = (Xtr - mu) / sd
    # ONE global scalar: uniform scaling only (convex optimum unchanged)
    s = 1.0 / Ytr.std().clamp_min(1e-12)
    Ytr = Ytr * s
    print(f"[m67g] train features {Xtr.shape} -> freq targets {Ytr.shape} "
          f"(input bins 0-{F}; global target scale s={s:.3e}, no per-unit "
          f"whitening)", flush=True)

    model = torch.nn.Linear(Xtr.shape[1], Ytr.shape[1])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[m67g] params {n_params} (linear)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=1e-6)

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
                    rhat_ft = (model((feats(b) - mu) / sd) / s).reshape(
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
                print(f"  ep {ep+1}: train mse {loss.item():.3e} | "
                      f"test rl2 = {rl2:.6e} (head-only {rl2_h:.6e}) "
                      f"[best {best['rl2']:.6e} @ep{best['epoch']}] "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"[m67g] done: best official test rl2 = {best['rl2']:.6e} at epoch "
          f"{best['epoch']} | target 1e-4 {'PASS' if best['rl2'] <= 1e-4 else 'FAIL'}")
    with open(args.out / "summary.json", "w") as f:
        json.dump({"best_rl2": best["rl2"], "best_epoch": best["epoch"],
                   "params": n_params, "epochs": args.epochs, "lr": args.lr,
                   "fit_bins": F, "optimizer": "AdamW", "target_scale": float(s)}, f, indent=1)


if __name__ == "__main__":
    main()
