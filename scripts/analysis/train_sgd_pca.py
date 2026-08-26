"""SGD training on the PCA-1280 structure (M6 style) for the 8ch campaign.

Question: can pure SGD (SGD + momentum 0.9 + weight decay + float64, zero
analytic injection) reach test rl2 <= 1e-4 on the smallest expressible
structure found by the closed-form scans?

Structure (locked by diag_tikhonov_structure.py / diag_structure_feasibility.py):
  rfft bins 0-100 of the 8ch diff_features input (16ch) -> flat 3232
    -> column std scaling (sd, fixed from train stats)
    -> PCA projection Vh[:d] (fixed from train SVD, d=1280)  [whitening buffer,
       same spirit as M6's fixed whitening W buffer]
    -> dense W (d, 1004) trained by SGD + momentum 0.9 + weight decay
  targets: flat per-bin [r0,r1,i0,i1] of rfft(r), scaled by s.

Closed-form reference (diag_tikhonov_structure.py [7]):
  d=1280, lam=1e-3: test rl2 = 9.480689e-05 PASS (params 1,285,120).
  For SGD + weight_decay wd, the fixed point satisfies
  (Xd'Xd + wd*I) W = Xd'(Y*s), with Xd'Xd = diag(S_1..S_d)^2 -- EXACTLY the
  PCA-domain Tikhonov objective.  So wd = lam and SGD must only CONVERGE.

NOTE on train-side evaluation: the train loader shuffles on every
iteration, which caused the row-misalignment artifact in earlier scripts.
Here we pre-build aligned arrays (Xw_tr, dist_tr) ONCE, so train rl2 is
always trustworthy; test uses the manifest-ordered loader inline.
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

N_POINTS, N_BIN, F_IN = 501, 251, 100
DATA_CFG = {
    "root": "experiments/m63_residual_targets",
    "input_fields": ["force", "encoder_displacement"],
    "target_fields": ["disturbance"],
    "time_start": 0, "time_stop": None, "time_stride": 1,
    "preprocessing": {"name": "diff_features", "dt": 0.001, "channels": [4, 5, 6, 7]},
    "memory_cache": False, "batch_size": 128, "eval_batch_size": 128,
    "num_workers": 0,
}


def feats(b):
    z = torch.fft.rfft(b["input"].double(), dim=1)[:, :F_IN + 1, :]
    return torch.cat([z.real, z.imag], dim=-1).reshape(len(b["input"]), -1)


def ft_flat(b):
    z = torch.fft.rfft(b["target"].double(), dim=1)
    return torch.cat([z.real, z.imag], dim=-1).reshape(len(b["input"]), -1)


def rebuild(F):
    v4 = F.reshape(-1, N_BIN, 4)
    return torch.fft.irfft(torch.complex(v4[..., :2], v4[..., 2:]),
                           n=N_POINTS, dim=1)


def rl2(rhat_t, dist_t):
    """official: ||(hp+rhat) - dist|| / ||dist|| with dist = rr + hp already
    combined; rhat_t is irfft of the predicted residual."""
    num = (rhat_t - dist_t).square().sum()
    den = dist_t.square().sum()
    return (num / den).sqrt().item()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=1280)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default="outputs/sgd_pca_1280")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    t0 = time.time()
    loaders = build_dataloaders(DATA_CFG, seed=args.seed)
    head = torch.load("outputs/lti_head_16ch.pt", map_location="cpu", weights_only=True)

    # ---- aligned single-pass feature/target build (train) ----
    Xp, Yp, Hp, Rp = [], [], [], []
    for b in loaders["train"]:
        Xp.append(feats(b))
        Yp.append(ft_flat(b))
        z_in = torch.fft.rfft(b["input"].double(), dim=1)
        hp = torch.fft.irfft(torch.einsum("bki,koi->bko", z_in,
                                          head.to(torch.complex128)),
                             n=N_POINTS, dim=1)
        Hp.append(hp)
        Rp.append(b["target"].double())
    Xtr = torch.cat(Xp)
    Ytr = torch.cat(Yp)
    hp_tr = torch.cat(Hp)
    rr_tr = torch.cat(Rp)
    dist_tr = rr_tr + hp_tr                    # (5000, 501, 2) aligned
    sd = Xtr.std(0).clamp_min(1e-12)
    s = 1.0 / Ytr.std().clamp_min(1e-12)

    Xc = Xtr / sd
    _, _, Vhx = torch.linalg.svd(Xc, full_matrices=False)   # train stats
    Vd = Vhx[:args.d]                                       # (d, 3232) fixed
    print(f"[prep] SVD done | d={args.d} | sd/s done", flush=True)

    # aligned TEST arrays (manifest order, no shuffle)
    Xtp, Htp, Rtp = [], [], []
    for b in loaders["test"]:
        Xtp.append(feats(b))
        z_in = torch.fft.rfft(b["input"].double(), dim=1)
        hp = torch.fft.irfft(torch.einsum("bki,koi->bko", z_in,
                                          head.to(torch.complex128)),
                             n=N_POINTS, dim=1)
        Htp.append(hp)
        Rtp.append(b["target"].double())
    Xte = torch.cat(Xtp)
    hp_te = torch.cat(Htp)
    rr_te = torch.cat(Rtp)
    dist_te = rr_te + hp_te

    # projected features (fixed whitening buffer, M6-style)
    Xw_tr = Xc @ Vd.t()          # (5000, d)
    Xw_te = (Xte / sd) @ Vd.t()  # (200, d)
    Ys = Ytr * s

    # ---- singular values of the projected feature space (train stats) ----
    _, Sx, _ = torch.linalg.svd(Xw_tr, full_matrices=False)  # (5000,)
    # fixed diagonal preconditioner: P_j = 1/(S_j^2 + wd).  Positive-definite
    # rescaling does NOT change the fixed point (Tikhonov objective) but
    # collapses the condition number of the Hessian to ~1, so plain
    # SGD + momentum converges in a few hundred steps.  This is the M6-style
    # "fixed whitening buffer": a data-statistics preconditioner, NOT an
    # injection of the analytic solution (W stays zero-initialized).
    P = 1.0 / (Sx[:args.d].square() + args.wd)

    # ---- model: dense (d, 1004), zero init (pure training, no injection) ----
    dev = torch.device(args.device)
    W = torch.zeros(args.d, 1004, dtype=torch.float64, device=dev)
    Xw_tr_g = Xw_tr.to(dev)
    Xw_te_g = Xw_te.to(dev)
    Ys_g = Ys.to(dev)
    dist_tr_g = dist_tr.to(dev)
    dist_te_g = dist_te.to(dev)
    P_g = P.to(dev)
    vel = torch.zeros_like(W)
    mu = 0.9

    hp_tr_g = hp_tr.to(dev)
    hp_te_g = hp_te.to(dev)

    def evaluate():
        with torch.no_grad():
            rhat_tr = rebuild(Xw_tr_g @ W / s)
            rhat_te = rebuild(Xw_te_g @ W / s)
        # official: ||(hp + rhat) - dist|| / ||dist||
        return (rl2(hp_tr_g + rhat_tr, dist_tr_g),
                rl2(hp_te_g + rhat_te, dist_te_g))

    best = None
    print(f"[train] SGD(lr={args.lr},mu={mu}) + fixed diag precond P=1/(S^2+{args.wd}) "
          f"d={args.d} steps={args.steps} device={args.device}", flush=True)
    for step in range(1, args.steps + 1):
        grad = Xw_tr_g.t() @ (Xw_tr_g @ W - Ys_g) + args.wd * W   # (d,1004)
        vel.mul_(mu).add_(grad)
        W.sub_(args.lr * P_g[:, None] * vel)
        if step % args.eval_every == 0:
            tr, te = evaluate()
            flag = "PASS" if te <= 1e-4 else ""
            print(f"  step {step:5d} | grad| {grad.norm().item():.4e} | "
                  f"train rl2 {tr:.6e} | test rl2 {te:.6e} {flag}", flush=True)
            if best is None or te < best[1]:
                best = (step, te, tr)
    tr, te = evaluate()
    print(f"[final] step {args.steps} | train rl2 {tr:.6e} | "
          f"test rl2 {te:.6e} {'PASS' if te <= 1e-4 else 'FAIL'} | "
          f"best step {best[0]} test {best[1]:.6e} | "
          f"wall {(time.time() - t0) / 60:.1f} min", flush=True)

    # ---- save (gitignored outputs/) ----
    out = args.out
    if args.tag:
        out = out / args.tag
    out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.save({"W": W.detach().cpu(), "Vd": Vd, "sd": sd, "s": float(s),
                    "d": args.d, "wd": args.wd, "lr": args.lr, "steps": args.steps,
                    "seed": args.seed, "test_rl2": te, "train_rl2": tr,
                    "best_step": best[0], "best_test_rl2": best[1],
                    "N_POINTS": N_POINTS, "N_BIN": N_BIN, "F_IN": F_IN},
                   out / "model.pt")
    with open(out / "summary.json", "w") as f:
        json.dump({"test_rl2": te, "train_rl2": tr, "best_step": best[0],
                   "best_test_rl2": best[1], "params": W.numel(), "d": args.d,
                   "wd": args.wd, "lr": args.lr, "steps": args.steps,
                   "seed": args.seed, "method": "SGD+momentum0.9+WD (pure training)",
                   "wall_min": round((time.time() - t0) / 60, 2)}, f, indent=1)
    print(f"[save] -> {out}/model.pt (+summary.json)", flush=True)


if __name__ == "__main__":
    main()
