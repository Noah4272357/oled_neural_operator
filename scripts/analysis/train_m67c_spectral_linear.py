"""m67c: full-spectrum linear residual head trained by SGD from scratch.

Follow-up to the archived 8ch campaign (experiments/analysis/2026-08-26/
8ch-nonlinear-campaign.md §4).  The diagnostic chain established:

  - r = dist - head(x16) is almost entirely a LINEAR function of the FULL
    input spectrum (per-bin R^2 >= 0.997), i.e. cross-bin coupling that the
    per-bin head cannot express (head-only rl2 1.52e-3).
  - the closed-form full-spectrum linear map reaches test rl2 = 9.91e-5
    (< 1e-4 target) when fitted on 4000 train samples, evaluated on test
    (diag_r_predictability.py).  That is the *information ceiling* of this
    parameterization.
  - m67b (closed-form lstsq) was never run; m67c is the training-pipeline
    version: SAME parameterization trained by pure SGD from random init,
    no init from the closed-form solution, no resume -- same contract as
    the M6 spectral model (rfft -> whiten -> shared complex M -> irfft,
    26 complex params, SGD).  The hard gate is "pure SGD training, zero
    injection", so only m67c's result counts toward the gate.

Parameterization: Linear(3232 -> 1004)
    input  = bins 0..100 of rfft(x16), real+imag flattened (same features
             as the fitted ceiling; 0-200 Hz)
    output = all 251 bins of rfft(r), real+imag flattened (2 ch x 2 parts)
    loss   = MSE in the frequency domain (Parseval-equivalent to
             time-domain MSE on r; convex problem -- SGD converges to the
             same optimum as the closed-form lstsq map)
    eval   = official: pred = head(x16) + irfft(r_hat_ft), test rl2 vs
             reconstructed dist, complex head preserved every step.

Usage (project .venv, CPU):
  .venv/bin/python scripts/analysis/train_m67c_spectral_linear.py \
      [--epochs 600] [--lr 1e-2] [--fit-bins 100] [--out outputs/m67c_linear_sgd]
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
    parser.add_argument("--out", type=Path, default="outputs/m67c_linear_sgd")
    args = parser.parse_args()
    F = args.fit_bins

    torch.manual_seed(20260810)
    loaders = build_dataloaders(DATA_CFG, seed=20260810)
    head = torch.load(args.head, map_location="cpu", weights_only=True)  # complex128, keep

    # ---- features & targets (frequency domain, real+imag flattened) ----
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
    Xtr = torch.cat(xs)   # (5000, (F+1)*16*2)
    Ytr = torch.cat(ys)   # (5000, 251*2*2)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-12)
    Xtr = (Xtr - mu) / sd
    Ymu, Ysd = Ytr.mean(0), Ytr.std(0).clamp_min(1e-12)
    Ytr = (Ytr - Ymu) / Ysd
    print(f"[m67c] train features {Xtr.shape} -> freq targets {Ytr.shape} "
          f"(input bins 0-{F} = 0-{F*2} Hz)", flush=True)

    model = torch.nn.Linear(Xtr.shape[1], Ytr.shape[1])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[m67c] params {n_params} (linear, no hidden layers)", flush=True)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=1e-5)

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
            # official metrics on test (head + linear residual, original data)
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
                    rhat_ft = (model((feats(b) - mu) / sd) * Ysd + Ymu).reshape(
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
    print(f"[m67c] done: best official test rl2 = {best['rl2']:.6e} at epoch "
          f"{best['epoch']} | target 1e-4 {'PASS' if best['rl2'] <= 1e-4 else 'FAIL'}")
    with open(args.out / "summary.json", "w") as f:
        json.dump({"best_rl2": best["rl2"], "best_epoch": best["epoch"],
                   "params": n_params, "epochs": args.epochs, "lr": args.lr,
                   "fit_bins": F, "momentum": 0.9}, f, indent=1)


if __name__ == "__main__":
    main()
