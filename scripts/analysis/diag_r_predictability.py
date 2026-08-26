"""Per-bin linear predictability of the residual r, and the model error spectrum.

r = dist - head(x16) has 88.2% of its energy in bins 8-250 (the broadband
tail + intermodulation bands); tone bins 4-7 hold only ~4.9%.  The m65/m67a
MLP plateau at ~5.4e-4 (= 12.5% of r unrecovered).  This script answers:
is that plateau the *information limit* (r's tail is only partly determined
by the current window's spectrum) or a *learning failure* (the tail IS
linearly predictable and the MLP just didn't recover it)?

  1. per-bin linear R^2 of r_b ~ z_bins(0-100)  (fitted on 4000 train
     samples, evaluated on test): energy share of each bin recoverable by
     the best linear map.
  2. cumulative unrecoverable energy -> theoretical best global rl2 for
     ANY per-bin-linear model (head + linear residual head).
  3. m67a error spectrum: per-bin |E|^2 vs per-bin |r|^2, to see which
     bins the trained MLP actually failed on.

Usage:
  .venv/bin/python scripts/analysis/diag_r_predictability.py \
      [--checkpoint outputs/m67a_full_2048/best.pt]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import build_dataloaders

N_POINTS = 501
N_BIN = 251
F_IN = 100  # use input bins 0..100 (0-200 Hz) as features

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
    parser.add_argument("--checkpoint", type=Path,
                        default="outputs/m67a_full_2048/best.pt")
    args = parser.parse_args()

    loaders = build_dataloaders(DATA_CFG, seed=20260810)

    # ---- build train design matrix (features: bins 0..F_IN of z, real+imag) ----
    def feats_ft(batch) -> torch.Tensor:
        z = torch.fft.rfft(batch["input"], dim=1)[:, : F_IN + 1, :]
        return torch.cat([z.real, z.imag], dim=-1).reshape(len(batch["input"]), -1)

    Xtr, Rtr = [], []
    ntr = 0
    for b in loaders["train"]:
        Xtr.append(feats_ft(b).double())
        Rtr.append(torch.fft.rfft(b["target"].double(), dim=1))
        ntr += len(b["input"])
        if ntr >= 4000:
            break
    X = torch.cat(Xtr)          # (4000, (F_IN+1)*16*2)
    R = torch.cat(Rtr)          # (4000, 251, 2) complex128
    print(f"[train] X {X.shape}  (4000 samples, input bins 0-{F_IN} = 0-{F_IN*2} Hz)")
    # W solves X @ W ~ R per bin: W = pinv(X) @ R_b
    W = torch.linalg.pinv(X.to(torch.complex128)) @ R.permute(1, 0, 2)  # (251, nfeat, 2) complex

    # ---- test evaluation ----
    E_r = torch.zeros(N_BIN, dtype=torch.float64)   # per-bin r energy
    E_lin = torch.zeros(N_BIN, dtype=torch.float64)  # per-bin unrecovered energy
    E_m = torch.zeros(N_BIN, dtype=torch.float64)   # per-bin model error energy
    E_d = torch.zeros(N_BIN, dtype=torch.float64)   # per-bin dist energy

    model = None
    if args.checkpoint.exists():
        import train_m65_residual_mlp as m65
        model = m65.ResidualNet((N_BIN * 16 * 2), N_POINTS * 2, width=2048)
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu",
                                         weights_only=True))
        model.eval()

    mu = sd = Ymu = Ysd = None
    if model is not None:
        # rebuild training standardization (cheap: reuse first 4000 samples)
        xs, ys = [], []
        for b in loaders["train"]:
            z = torch.fft.rfft(b["input"], dim=1)
            xs.append(torch.cat([z.real, z.imag], dim=-1).reshape(len(b["input"]), -1))
            ys.append(b["target"].reshape(-1, N_POINTS * 2))
            if len(xs) * 128 >= 4000:
                break
        Xtr_full = torch.cat(xs)[:4000]
        Ytr = torch.cat(ys)[:4000]
        mu, sd = Xtr_full.mean(0), Xtr_full.std(0).clamp_min(1e-12)
        Ymu, Ysd = Ytr.mean(0), Ytr.std(0).clamp_min(1e-12)

    head = torch.load("outputs/lti_head_16ch.pt", map_location="cpu", weights_only=True)
    for b in loaders["test"]:
        xx = b["input"].double()
        rr = b["target"].double()
        z = torch.fft.rfft(xx, dim=1)
        hp = torch.fft.irfft(
            torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
            n=N_POINTS, dim=1)
        dist = rr + hp
        Rf = torch.fft.rfft(rr, dim=1)
        Df = torch.fft.rfft(dist, dim=1)
        E_r += Rf.abs().square().sum((0, 2))
        E_d += Df.abs().square().sum((0, 2))
        # linear prediction of r per bin
        Xb = feats_ft(b)
        rhat_lin = torch.einsum("bi,kio->bko", Xb.to(torch.complex128), W)
        E_lin += (Rf - rhat_lin).abs().square().sum((0, 2))
        if model is not None:
            zfull = torch.fft.rfft(b["input"], dim=1)
            ffull = torch.cat([zfull.real, zfull.imag], dim=-1).reshape(
                len(b["input"]), -1)
            rhat = (model((ffull.float() - mu) / sd) * Ysd + Ymu).reshape(
                len(b["input"]), N_POINTS, 2)
            Eh = torch.fft.rfft(rhat.double() - rr, dim=1)
            E_m += Eh.abs().square().sum((0, 2))

    r2 = 1.0 - E_lin / E_r.clamp_min(1e-300)
    tot_r = E_r.sum().item()
    tot_d = E_d.sum().item()
    print("[per-bin] k : R2_lin | r_share(r) | m67a-recovered-share")
    for k in list(range(0, 21)) + [25, 30, 40, 50, 64, 80, 100, 128, 160, 200, 250]:
        rec = 1.0 - (E_m[k].item() / E_r[k].item()) if model is not None and E_r[k].item() > 0 else float("nan")
        print(f"  {k:3d}: {r2[k].item():+.4f} | {E_r[k].item()/tot_r:.4e} | "
              f"{rec:.4f}")
    # cumulative linear limit
    unr = (E_lin * (E_r > 1e-30)).sum().item()
    rl2_lin = (unr / tot_d) ** 0.5
    print(f"[linear limit] unrecovered r energy {unr:.3e} / dist {tot_d:.3e} -> "
          f"best-possible global rl2 = {rl2_lin:.6e}")
    if model is not None:
        em = E_m.sum().item()
        print(f"[m67a] error energy {em:.3e} -> global rl2 = "
          f"{(em/tot_d)**0.5:.6e} | recovery of r = {1 - em/tot_r:.4f} | "
          f"gap to linear limit = {em/unr:.2f}x")
    print(f"[tail note] bins 41-250 hold {E_r[41:].sum().item()/tot_r:.3f} of r energy "
          f"(fitted with input bins 0-{F_IN} only; high-freq coupling may be missing)")


if __name__ == "__main__":
    main()
