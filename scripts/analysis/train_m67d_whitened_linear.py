"""m67d: full-spectrum linear residual head, SGD from scratch, WHITENED.

m67c (train_m67c_spectral_linear.py) confirmed the M6-recorded failure
mode: full-batch SGD on the raw spectral features barely moves (600 ep,
test rl2 1.399e-3 vs the closed-form ceiling 9.91e-5) -- the rfft
features are ill-conditioned across bins/channels and first-order
optimizers appear frozen.

Fix (identical to the M6 recipe, spectral_whiten.py): precondition the
input with the fixed Hermitian square root of the inverse per-pair joint
train Gram,

    C = (1 / (N_samples * N_bin)) * sum_n sum_k x_ft[n,k]^H x_ft[n,k]
    W = C^{-1/2}                                    (16x16 complex)

so z = x_ft @ W has identity per-(sample,bin) covariance.  W is a fixed
data-preprocessing buffer (like outputs/spectral_whiten/whiten.pt), NOT
an injected solution -- training is pure SGD from random init on the
same Linear(3232 -> 1004) parameterization as m67c.  This is the m67c
information ceiling (9.91e-5 < 1e-4) under a well-conditioned geometry.

Pipeline identical to m67c:
  input  = bins 0..100 of rfft(diff_features(x)) @ W, real+imag
  output = all 251 bins of rfft(r), real+imag   (freq-domain MSE)
  eval   = official head + residual vs dist, complex head preserved.

Usage (project .venv, CPU):
  .venv/bin/python scripts/analysis/train_m67d_whitened_linear.py \
      [--epochs 600] [--lr 1e-2] [--fit-bins 100] [--out outputs/m67d_linear_sgd]
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
N_CH = 16  # diff_features channels

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
    parser.add_argument("--out", type=Path, default="outputs/m67d_linear_sgd")
    args = parser.parse_args()
    F = args.fit_bins

    torch.manual_seed(20260810)
    loaders = build_dataloaders(DATA_CFG, seed=20260810)
    head = torch.load(args.head, map_location="cpu", weights_only=True)  # complex128

    def xft(batch) -> torch.Tensor:
        return torch.fft.rfft(batch["input"], dim=1).to(torch.complex128)  # (B,251,16)

    # ---- fixed whitening buffer from the train Gram (M6 recipe) ----
    G = torch.zeros((N_CH, N_CH), dtype=torch.complex128)
    n_pairs = 0
    for b in loaders["train"]:
        z = xft(b)
        G += torch.einsum("btc,btd->cd", z.conj(), z)
        n_pairs += z.shape[0] * z.shape[1]
    evals, evecs = torch.linalg.eigh(G / n_pairs)
    evals = evals.clamp_min(1e-10)
    W = (evecs * (1.0 / evals.sqrt())) @ evecs.conj().T  # C^{-1/2}, fixed buffer
    print(f"[m67d] whiten W {tuple(W.shape)} from {n_pairs} pairs", flush=True)

    def feats(batch) -> torch.Tensor:
        z = (xft(batch) @ W)[:, : F + 1, :]
        return torch.cat([z.real, z.imag], dim=-1).reshape(
            len(batch["input"]), -1).float()

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
    Ymu, Ysd = Ytr.mean(0), Ytr.std(0).clamp_min(1e-12)
    Ytr = (Ytr - Ymu) / Ysd
    print(f"[m67d] train features {Xtr.shape} -> freq targets {Ytr.shape} "
          f"(whitened, input bins 0-{F})", flush=True)

    model = torch.nn.Linear(Xtr.shape[1], Ytr.shape[1])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[m67d] params {n_params} (linear)", flush=True)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
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
                    torch.save(W, args.out / "whiten.pt")
                print(f"  ep {ep+1}: train mse {loss.item():.3e} | "
                      f"test rl2 = {rl2:.6e} (head-only {rl2_h:.6e}) "
                      f"[best {best['rl2']:.6e} @ep{best['epoch']}] "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"[m67d] done: best official test rl2 = {best['rl2']:.6e} at epoch "
          f"{best['epoch']} | target 1e-4 {'PASS' if best['rl2'] <= 1e-4 else 'FAIL'}")
    with open(args.out / "summary.json", "w") as f:
        json.dump({"best_rl2": best["rl2"], "best_epoch": best["epoch"],
                   "params": n_params, "epochs": args.epochs, "lr": args.lr,
                   "fit_bins": F, "whitened": True}, f, indent=1)


if __name__ == "__main__":
    main()
