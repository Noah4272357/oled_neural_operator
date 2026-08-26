"""Tikhonov-regularized closed-form solution of the 8ch full-spectrum
linear residual head (m67h feature map), lambda = 1e-3.

Status (2026-08-26): this closed-form linear solution reaches the
official test rl2 = 9.42e-5 < 1e-4 target with ZERO training.  The
m67h SGD/LBFGS training family tops out at 1.065e-4 (LBFGS trajectory)
and the unregularized LS solution overfits (test 6.1e-2).

Pipeline (matches train_m67h_minibatch_sgd.py exactly):
  features: RAW rfft bins 0-100 real+imag of the 8ch diff_features input
            (16ch), column-scaled X / col-std (NO centering)
  targets : flat per-bin [r0,r1,i0,i1] of rfft(r), ONE global scalar s
  solve   : W = V g S^-1 U^H (Y*s), g = S^2/(S^2+lam)  (Tikhonov)
            float64 SVD of (X/sd): true numerical rank 2339/3232
            (the f32 rank-3216 'full rank' was a float-precision artifact)
  predict : rhat_time = irfft(rebuild((X/sd) @ W / s))

Artifacts: outputs/tikhonov_linear_1e-3/model.pt  ({"weight": (1004,3232),
           "sd": (3232,), "s": scalar, "lambda": 1e-3, ...})
           outputs/tikhonov_linear_1e-3/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import build_dataloaders

N_POINTS = 501
N_BIN = 251
F_IN = 100
DATA_CFG = {
    "root": "experiments/m63_residual_targets",
    "input_fields": ["force", "encoder_displacement"],
    "target_fields": ["disturbance"],
    "time_start": 0, "time_stop": None, "time_stride": 1,
    "preprocessing": {"name": "diff_features", "dt": 0.001, "channels": [4, 5, 6, 7]},
    "memory_cache": False, "batch_size": 128, "eval_batch_size": 128,
    "num_workers": 0,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda", dest="lam", type=float, default=1e-3,
                        metavar="L", help="Tikhonov ridge (default 1e-3)")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--head", type=Path, default="outputs/lti_head_16ch.pt")
    parser.add_argument("--out", type=Path, default="outputs/tikhonov_linear_1e-3")
    args = parser.parse_args()

    loaders = build_dataloaders(DATA_CFG, seed=args.seed)
    head = torch.load(args.head, map_location="cpu", weights_only=True)

    def feats(b):
        z = torch.fft.rfft(b["input"].double(), dim=1)[:, :F_IN + 1, :]
        return torch.cat([z.real, z.imag], dim=-1).reshape(len(b["input"]), -1)

    def ft_flat(b):
        z = torch.fft.rfft(b["target"].double(), dim=1)
        return torch.cat([z.real, z.imag], dim=-1).reshape(len(b["input"]), -1)

    # single pass per split; test/val keep manifest order so later
    # evaluation re-iteration is row-aligned, but we do NOT rely on it:
    # evaluation below is inline per batch.
    Xtr_parts, Ytr_parts = [], []
    for b in loaders["train"]:
        Xtr_parts.append(feats(b))
        Ytr_parts.append(ft_flat(b))
    Xtr = torch.cat(Xtr_parts)
    Ytr = torch.cat(Ytr_parts)
    sd = Xtr.std(0).clamp_min(1e-12)
    s = 1.0 / Ytr.std().clamp_min(1e-12)

    Xt = Xtr / sd
    U, S, Vh = torch.linalg.svd(Xt, full_matrices=False)
    rank = int((S > 1e-14 * S[0]).sum())
    g = S / (S * S + args.lam)
    W = Vh.t() @ (g.unsqueeze(1) * (U.t() @ (Ytr * s)))
    print(f"[tik] SVD {tuple(Xt.shape)} true rank {rank}/{Xt.shape[1]} | "
          f"lambda {args.lam:.1e} | W {tuple(W.shape)}", flush=True)

    def rebuild(F):
        v4 = F.reshape(-1, N_BIN, 4)
        return torch.fft.irfft(torch.complex(v4[..., :2], v4[..., 2:]),
                               n=N_POINTS, dim=1)

    def rl2(split):
        num = den = 0.0
        for b in loaders[split]:
            rr = b["target"].double()
            z_in = torch.fft.rfft(b["input"].double(), dim=1)
            hp = torch.fft.irfft(torch.einsum("bki,koi->bko", z_in,
                                              head.to(torch.complex128)),
                                 n=N_POINTS, dim=1)
            dist = rr + hp
            out = (feats(b) / sd) @ W / s
            rhat = rebuild(out)
            num += (hp + rhat - dist).square().sum().item()
            den += dist.square().sum().item()
        return (num / den) ** 0.5

    rl2_tr = rl2("train")
    rl2_te = rl2("test")
    rl2_va = rl2("val")
    print(f"[tik] train rl2 = {rl2_tr:.6e} | val rl2 = {rl2_va:.6e} | "
          f"test rl2 = {rl2_te:.6e} | target 1e-4 "
          f"{'PASS' if rl2_te <= 1e-4 else 'FAIL'}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": W, "sd": sd, "s": float(s), "lambda": args.lam,
                "fit_bins": F_IN, "rank": rank, "seed": args.seed,
                "N_POINTS": N_POINTS, "N_BIN": N_BIN}, args.out / "model.pt")
    with open(args.out / "summary.json", "w") as f:
        json.dump({"test_rl2": rl2_te, "val_rl2": rl2_va, "train_rl2": rl2_tr,
                   "lambda": args.lam, "rank": rank, "seed": args.seed,
                   "method": "Tikhonov closed-form (zero training)"}, f, indent=1)
    print(f"[tik] saved -> {args.out}/model.pt")


if __name__ == "__main__":
    main()
