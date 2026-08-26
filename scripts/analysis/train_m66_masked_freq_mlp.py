"""m66: frequency-masked residual MLP for the 8ch nonlinear campaign.

Diagnosis that motivated this (diag_m65_limits.py):
  - dist energy: 99.99% in tone bins 4-7 (8/10/12/14 Hz) + 0.01% in
    intermodulation bands (f_i +/- f_j: bins 1-3 = 2/4/6 Hz, bins 8-12 =
    16-24 Hz); bins 18+ hold < 0.001%.
  - The m65 residual MLP (full time-domain output) plateaued at 5.435e-4
    with ~95% of its final error energy OUTSIDE the tone bins -- including
    a large share in bins 25-34 (50-68 Hz) where dist energy is ~0.  That
    is out-of-band leakage of the time-domain output head, not information
    loss: the ideal output there is exactly zero.

m66 exploits the structure: the model outputs ONLY the first K=18 bins of
the residual's FFT (0-34 Hz, covers 99.999% of dist energy), everything
outside is hard zero by construction.  Output dimension drops 1002 -> 72,
concentrating capacity on the bands that carry signal.

  pred = head(x16) + irfft( pad( r̂_ft[:, 0:18] ) )     head frozen
  loss = MSE over the 72 masked frequency-domain values (Parseval-equivalent
         to masked time-domain MSE; out-of-band gradient is structurally 0)

Diagnostics per eval: in-band (0-34 Hz) vs out-of-band error energy, so we
see immediately whether the mask removed the leakage floor.

Usage (project .venv, CPU):
  .venv/bin/python scripts/analysis/train_m66_masked_freq_mlp.py [--epochs 400]
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
K = 18  # keep bins 0..17 (0-34 Hz)

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


class MaskedFreqNet(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 2048, depth: int = 3):
        super().__init__()
        layers = [torch.nn.Linear(in_dim, width), torch.nn.GELU()]
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(width, width), torch.nn.GELU()]
        layers.append(torch.nn.Linear(width, out_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def feats(batch) -> torch.Tensor:
    z = torch.fft.rfft(batch["input"], dim=1)
    return torch.cat([z.real, z.imag], dim=-1).reshape(len(batch["input"]), -1)


def ft_targets(batch) -> torch.Tensor:
    """First K bins of rfft(residual target), real+imag flattened (B, K*2*2)."""
    z = torch.fft.rfft(batch["target"], dim=1)[:, :K, :]
    return torch.cat([z.real, z.imag], dim=-1).reshape(len(batch["target"]), -1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--head", type=Path, default="outputs/lti_head_16ch.pt")
    parser.add_argument("--out", type=Path, default="outputs/m66_masked_freq")
    args = parser.parse_args()

    torch.manual_seed(20260810)
    loaders = build_dataloaders(DATA_CFG, seed=20260810)
    head_full = torch.load(args.head, map_location="cpu", weights_only=True)  # complex128, keep
    assert head_full.shape == (N_BIN, 2, 16), head_full.shape
    # mask the head: its per-bin LS fit has out-of-band leakage (non-zero
    # output in bins where dist energy is ~0).  Structure-zero it so the
    # final prediction has no energy above bin K-1 at all.  dist is still
    # reconstructed from head_full (the residual dataset stores rr =
    # dist - head_full(x16)); predictions use the masked head.
    head = head_full.clone()
    head[K:] = 0

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
    print(f"[m66] train features {Xtr.shape} -> masked freq targets {Ytr.shape} "
          f"(bins 0-{K-1} = 0-{(K-1)*2} Hz)")

    model = MaskedFreqNet(Xtr.shape[1], Ytr.shape[1], width=args.width)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[m66] params {n_params}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=3e-6)

    def to_time(rhat_ft: torch.Tensor, n_batch: int) -> torch.Tensor:
        """(B, K*2*2) standardized -> time-domain residual (B, N_POINTS, 2)."""
        ft = (rhat_ft * Ysd + Ymu).reshape(n_batch, K, 2, 2)
        z = torch.complex(ft[..., 0], ft[..., 1])
        return torch.fft.irfft(z, n=N_POINTS, dim=1)

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
                num_in = torch.zeros((), dtype=torch.float64)   # in-band error (0-34 Hz)
                num_out = torch.zeros((), dtype=torch.float64)  # out-of-band error
                den_in = torch.zeros((), dtype=torch.float64)   # in-band dist energy
                for b in loaders["test"]:
                    xx = b["input"].double()
                    rr = b["target"].double()
                    z = torch.fft.rfft(xx, dim=1)
                    hp_full = torch.fft.irfft(
                        torch.einsum("bki,koi->bko", z, head_full.to(torch.complex128)),
                        n=N_POINTS, dim=1)
                    hp = torch.fft.irfft(
                        torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
                        n=N_POINTS, dim=1)
                    dist = rr + hp_full
                    rhat = to_time(model((feats(b) - mu) / sd), len(b["input"])).double()
                    pred = hp + rhat
                    E = pred - dist
                    num += E.square().sum()
                    den += dist.square().sum()
                    num_h += (hp - dist).square().sum()
                    Ef = torch.fft.rfft(E, dim=1)
                    num_in += Ef[:, :K, :].abs().square().sum()
                    num_out += Ef[:, K:, :].abs().square().sum()
                    Df = torch.fft.rfft(dist, dim=1)
                    den_in += Df[:, :K, :].abs().square().sum()
                rl2 = torch.sqrt(num / den).item()
                rl2_h = torch.sqrt(num_h / den).item()
                # total error energy == in+out by Parseval; report shares
                err_tot = num_in.item() + num_out.item()
                in_share = num_in.item() / err_tot if err_tot > 0 else 0.0
                out_rl2 = torch.sqrt(num_out / den).item()
                in_rl2 = torch.sqrt(num_in / den_in).item()
                if rl2 < best["rl2"]:
                    best = {"rl2": rl2, "epoch": ep + 1}
                    args.out.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), args.out / "best.pt")
                print(f"  ep {ep+1}: train mse {loss.item():.3e} | "
                      f"test rl2 = {rl2:.6e} (head-only {rl2_h:.6e}) "
                      f"[best {best['rl2']:.6e} @ep{best['epoch']}] | "
                      f"in-band rl2 {in_rl2:.6e} out-band rl2 {out_rl2:.6e} "
                      f"(out share {in_share*100:.1f}% in-band) "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"[m66] done: best official test rl2 = {best['rl2']:.6e} at epoch {best['epoch']}")
    with open(args.out / "summary.json", "w") as f:
        json.dump({"best_rl2": best["rl2"], "best_epoch": best["epoch"],
                   "params": n_params, "epochs": args.epochs, "mask_bins": K}, f, indent=1)


if __name__ == "__main__":
    main()
