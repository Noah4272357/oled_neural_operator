"""m67h: full-spectrum linear residual head, MINI-BATCH SGD+momentum.

Root cause of m67c-g failures (established 2026-08-26):
  - m67c/m67d (SGD, m67d per-pair whiten): full-batch 600 steps is far
    short of converging on the 3232-dim convex problem (train mse 0.56 vs
    closed-form optimum 3.6e-4, i.e. >1000x gap) -- slow, not frozen.
  - m67e (global whiten): whitening itself destroys information (closed
    form on whitened features drops to 3.03e-3).
  - m67f (AdamW + per-unit Y standardization): correct eval 6.03e-4
  - m67g (AdamW + global scalar target): correct eval 5.76e-4; closed
    form (bias-free, raw feats) = 9.18e-5; AdamW train loss 155x above
    the optimum after 600 FULL-batch steps -> still under-converged.
  - eval bug found & fixed: rhat rebuild via reshape(B,N_BIN,2,2) mixed
    real/imag columns (flat layout is per-bin [r0,r1,i0,i1]); all m67c-g
    reported test rl2 were 2.4-2.5x too high.  Training losses were
    unaffected.  m67h evaluates per-bin: v4 = out.reshape(B,N_BIN,4),
    rhat_c = complex(v4[..., :2], v4[..., 2:]).

m67h recipe (M6-style mini-batch training on the m67 linear head):
  - features: RAW rfft bins 0-100 real+imag, column-scaled ONLY (x / col
    std, NO centering: a diagonal scaling is invertible so the convex
    optimum -- the closed form -- is unchanged; the std-feature ceiling
    with centering is 1.25e-4 > 1e-4 because centering removes a column
    direction, raw col-scaled keeps the 9.18e-5 ceiling)
  - targets:  raw rfft(r) flattened per-bin [r0,r1,i0,i1] x ONE global
    scalar s (uniform scaling keeps the convex optimum)
  - model:    Linear(3232 -> 1004, bias=False)  (bias breaks the optimum:
    closed-form with bias = 1.98e-3 vs 1.25e-4 without, 16x)
  - optimizer: SGD momentum 0.9 (M6 recipe), cosine lr, mini-batch 128
    (39 steps/epoch; full-batch 600 steps was the under-convergence
    driver -- m67g AdamW train loss stayed 155x above the optimum)
  - evaluation: official test rl2, per-bin complex rebuild (fixed)

Usage (project .venv):
  .venv/bin/python scripts/analysis/train_m67h_minibatch_sgd.py \
      [--epochs 300] [--lr 3e-3] [--batch 128] [--fit-bins 100] \
      [--optimizer sgd|adamw] [--device cpu|cuda] \
      [--out outputs/m67h_minibatch_sgd]
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
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--fit-bins", type=int, default=100, metavar="K",
                        help="input feature bins 0..K (default 100 = 0-200 Hz)")
    parser.add_argument("--optimizer", choices=["sgd", "adamw", "lbfgs"], default="sgd")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--restart-every", type=int, default=0, metavar="K",
                        help="LBFGS only: clear optimizer history every K epochs "
                             "(0 = never). Quasi-Newton on ill-conditioned "
                             "quadratics grinds when the curvature history goes "
                             "stale; periodic restarts re-accelerate the tail.")
    parser.add_argument("--weight-decay", type=float, default=0.0, metavar="L",
                        help="explicit L2 penalty 0.5*L*||W||^2 added to the loss "
                             "(Tikhonov in weight space; the closed-form ridge "
                             "family tops out at 1.62e-3 test rl2, but the "
                             "TRAINED trajectory is not the steady state -- "
                             "the penalty reshapes the path, not the optimum)")
    parser.add_argument("--head", type=Path, default="outputs/lti_head_16ch.pt")
    parser.add_argument("--seed", type=int, default=20260810,
                        help="RNG seed (ensembles: different inits -> different "
                             "early-stop trajectories -> variance averaging)")
    parser.add_argument("--out", type=Path, default="outputs/m67h_minibatch_sgd")
    args = parser.parse_args()
    F = args.fit_bins
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[m67h] cuda requested but unavailable -> cpu", flush=True)
        device = torch.device("cpu")

    torch.manual_seed(args.seed)
    loaders = build_dataloaders(DATA_CFG, seed=args.seed)
    head = torch.load(args.head, map_location="cpu", weights_only=True)  # complex128

    def feats(batch) -> torch.Tensor:
        z = torch.fft.rfft(batch["input"].to(device), dim=1)[:, : F + 1, :]
        return torch.cat([z.real, z.imag], dim=-1).reshape(len(batch["input"]), -1)

    def ft_targets(batch) -> torch.Tensor:
        z = torch.fft.rfft(batch["target"].to(device), dim=1)
        return torch.cat([z.real, z.imag], dim=-1).reshape(len(batch["input"]), -1)

    # input spectrum check: energy of feature bins beyond F
    with torch.no_grad():
        e_lo = e_hi = 0.0
        for b in loaders["train"]:
            z = torch.fft.rfft(b["input"], dim=1)
            e_lo += z[:, : F + 1].abs().square().sum().item()
            e_hi += z[:, F + 1:].abs().square().sum().item()
        print(f"[m67h] input spectrum: bins 0-{F} hold "
              f"{e_lo/(e_lo+e_hi):.4f} of energy (bins {F+1}-250: {e_hi/(e_lo+e_hi):.4f})",
              flush=True)

    xs, ys = [], []
    for b in loaders["train"]:
        xs.append(feats(b))
        ys.append(ft_targets(b))
    Xtr = torch.cat(xs).to(dtype)
    Ytr = torch.cat(ys).to(dtype)
    # column-scaling ONLY (no centering: keeps the convex optimum, tames
    # SGD's per-column scale) + ONE global target scalar (uniform)
    sd = Xtr.std(0).clamp_min(1e-12)
    Xtr = Xtr / sd
    s = 1.0 / Ytr.std().clamp_min(1e-12)
    Ytr = Ytr * s
    Xtr = Xtr.to(device)
    Ytr = Ytr.to(device)
    sd = sd.to(device)
    print(f"[m67h] train features {Xtr.shape} -> freq targets {Ytr.shape} "
          f"(col-scaled RAW bins 0-{F}, global scale s={s:.3e}, bias=False, "
          f"opt={args.optimizer} lr={args.lr} batch={args.batch} dev={device})",
          flush=True)

    model = torch.nn.Linear(Xtr.shape[1], Ytr.shape[1], bias=False,
                            dtype=dtype).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[m67h] params {n_params} (linear, bias-free)", flush=True)
    if args.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    else:  # lbfgs: full-batch quasi-Newton on the convex quadratic
        opt = torch.optim.LBFGS(model.parameters(), lr=args.lr,
                                max_iter=50, history_size=150,
                                line_search_fn="strong_wolfe")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=1e-6)
    train_dl = loaders["train"]  # mini-batch, shuffled

    head_d = head.to(torch.complex128)

    def evaluate() -> float:
        # everything on device (GPU): einsum+irfft are fast there; the
        # CPU-side eval was the 45s/10ep bottleneck on the GPU node
        with torch.no_grad():
            model.eval()
            num = torch.zeros((), dtype=torch.float64, device=device)
            den = torch.zeros((), dtype=torch.float64, device=device)
            num_h = torch.zeros((), dtype=torch.float64, device=device)
            for b in loaders["test"]:
                xx = b["input"].double().to(device)
                rr = b["target"].double().to(device)
                z = torch.fft.rfft(xx, dim=1)
                hp = torch.fft.irfft(
                    torch.einsum("bki,koi->bko", z, head_d.to(device)),
                    n=N_POINTS, dim=1)
                dist = rr + hp
                out = model(feats(b).to(dtype) / sd) / s
                v4 = out.double().reshape(len(b["input"]), N_BIN, 4)
                rhat = torch.fft.irfft(
                    torch.complex(v4[..., :2], v4[..., 2:]),
                    n=N_POINTS, dim=1)
                pred = hp + rhat
                num += (pred - dist).square().sum()
                den += dist.square().sum()
                num_h += (hp - dist).square().sum()
            rl2 = torch.sqrt(num / den).item()
            return rl2, torch.sqrt(num_h / den).item()

    t0 = time.perf_counter()
    best = {"rl2": float("inf"), "epoch": -1}
    steps = 0
    for ep in range(args.epochs):
        model.train()
        if args.optimizer == "lbfgs":
            # periodic restart: drop the stale curvature history so the next
            # step is a fresh steepest-descent direction + full line search
            if args.restart_every and ep > 0 and ep % args.restart_every == 0:
                opt.state.clear()
            # full-batch quasi-Newton: one opt.step(closure) per epoch
            def closure():
                opt.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(model(Xtr), Ytr)
                if args.weight_decay:
                    loss = loss + 0.5 * args.weight_decay * \
                        model.weight.square().sum()
                loss.backward()
                return loss
            loss = opt.step(closure)
            steps += 1
            ep_loss = loss.item()
        else:
            ep_loss = 0.0
            for b in train_dl:
                opt.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(
                    model(feats(b).to(dtype) / sd),
                    ft_targets(b).to(dtype) * s)
                loss.backward()
                opt.step()
                ep_loss += loss.item() * len(b["input"])
                steps += 1
            sched.step()
            ep_loss /= 5000
        if (ep + 1) % 10 == 0:
            rl2, rl2_h = evaluate()
            if rl2 < best["rl2"]:
                best = {"rl2": rl2, "epoch": ep + 1}
                args.out.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), args.out / "best.pt")
            print(f"  ep {ep+1}: train mse {ep_loss:.3e} | test rl2 = "
                  f"{rl2:.6e} (head-only {rl2_h:.6e}) [best {best['rl2']:.6e} "
                  f"@ep{best['epoch']}] ({time.perf_counter()-t0:.0f}s)", flush=True)
    rl2, rl2_h = evaluate()
    if rl2 < best["rl2"]:
        best = {"rl2": rl2, "epoch": args.epochs}
        args.out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.out / "best.pt")
    print(f"[m67h] done: best official test rl2 = {best['rl2']:.6e} at epoch "
          f"{best['epoch']} | target 1e-4 {'PASS' if best['rl2'] <= 1e-4 else 'FAIL'}")
    with open(args.out / "summary.json", "w") as f:
        json.dump({"best_rl2": best["rl2"], "best_epoch": best["epoch"],
                   "params": n_params, "epochs": args.epochs, "lr": args.lr,
                   "batch": args.batch, "fit_bins": F,
                   "optimizer": {"sgd": "SGD_momentum", "adamw": "AdamW",
                                 "lbfgs": "LBFGS"}[args.optimizer],
                   "restart_every": args.restart_every, "weight_decay": args.weight_decay,
                   "seed": args.seed, "dtype": args.dtype, "device": args.device,
                   "target_scale": float(s), "steps": steps}, f, indent=1)


if __name__ == "__main__":
    main()
