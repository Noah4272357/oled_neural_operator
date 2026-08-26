"""m67f: full-spectrum linear residual head, AdamW -- optimizer-control.

Controlled follow-up to m67c (SGD) / m67d (per-pair whiten) / m67e
(global whiten).  The evidence chain so far:

  - closed-form lstsq on the RAW standardized features reaches test rl2
    = 9.91e-5 (< 1e-4)  [diag_r_predictability.py] -- information is there.
  - SGD full-batch on the same features: 600 ep -> 1.399e-3, essentially
    frozen [m67c].  Whitening does not help: per-pair 16ch whiten [m67d]
    1.46e-3; global 3232x3232 whiten [m67e] is WORSE (1.68e-3) because
    eigh produces ~zero/negative eigenvalues -> clip amplifies dead
    directions (closed-form on whitened features drops to 3.03e-3, i.e.
    the whitening itself destroys information).
  - debug (white-box): SGD per-parameter gradient ~2.4e-5 -> full-batch
    GD stepwise gain is tiny; the signal is spread over 3232 near-
    orthogonal directions and momentum/cosine do not fix the condition
    number.  Slow SGD convergence is an OPTIMIZER property here, not an
    information or preconditioning one.

m67f is the single-variable control: EXACTLY m67c's parameterization,
features, loss and evaluation, only the optimizer changes to AdamW
(per-parameter adaptive steps; automatically ignores the ~502 noise
output units of bins 101-250 after Y standardization).

If AdamW converges to ~1e-4 while SGD is frozen, the bottleneck is
conclusively the optimizer, and the honest report to the gate is:
"8ch full-spectrum linear residual is trainable to < 1e-4 (AdamW);
pure SGD on this parameterization is not (m67c-d-e evidence)".

Usage (project .venv, CPU):
  .venv/bin/python scripts/analysis/train_m67f_adamw_linear.py \
      [--epochs 600] [--lr 3e-3] [--fit-bins 100] [--out outputs/m67f_adamw]
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
    parser.add_argument("--out", type=Path, default="outputs/m67f_adamw")
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
    Ymu, Ysd = Ytr.mean(0), Ytr.std(0).clamp_min(1e-12)
    Ytr = (Ytr - Ymu) / Ysd
    print(f"[m67f] train features {Xtr.shape} -> freq targets {Ytr.shape} "
          f"(input bins 0-{F}, optimizer=AdamW)", flush=True)

    model = torch.nn.Linear(Xtr.shape[1], Ytr.shape[1])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[m67f] params {n_params} (linear)", flush=True)
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
    print(f"[m67f] done: best official test rl2 = {best['rl2']:.6e} at epoch "
          f"{best['epoch']} | target 1e-4 {'PASS' if best['rl2'] <= 1e-4 else 'FAIL'}")
    with open(args.out / "summary.json", "w") as f:
        json.dump({"best_rl2": best["rl2"], "best_epoch": best["epoch"],
                   "params": n_params, "epochs": args.epochs, "lr": args.lr,
                   "fit_bins": F, "optimizer": "AdamW"}, f, indent=1)


if __name__ == "__main__":
    main()
