"""Two decisive diagnostics for the m65 5.44e-4 plateau.

1. TRAIN rl2: if train rl2 is far below test, the model memorizes (capacity
   exhausted, regularization/generalization issue); if train ~= test ~ 5e-4,
   the architecture cannot even fit the training set (structure problem).

2. RESIDUAL spectrum: the residual r = dist - head(x16) energy per bin.
   If r is concentrated in bins {4,5,6,7} (the tone bins), a sparse-frequency
   model (regress few complex bins) replaces the 1002-dim time-domain output.

Usage (project .venv, CPU):
  .venv/bin/python scripts/analysis/diag_m65_limits.py \
      --checkpoint outputs/m65_residual_mlp/best.pt
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
from train_m65_residual_mlp import ResidualNet

N_POINTS = 501
N_BIN = 251
TONE_BINS = [4, 5, 6, 7]  # 8/10/12/14 Hz at 2 Hz/bin

DATA_CFG = {
    "root": "experiments/m63_residual_targets",
    "input_fields": ["force", "encoder_displacement"],
    "target_fields": ["disturbance"],
    "time_start": 0, "time_stop": None, "time_stride": 1,
    "preprocessing": {"name": "diff_features", "dt": 0.001, "channels": [4, 5, 6, 7]},
    "memory_cache": False, "batch_size": 128, "eval_batch_size": 128,
    "num_workers": 0,
    "max_train_samples": None, "max_val_samples": None, "max_test_samples": None,
}


def feats(batch) -> torch.Tensor:
    z = torch.fft.rfft(batch["input"], dim=1)
    return torch.cat([z.real, z.imag], dim=-1).reshape(len(batch["input"]), -1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--head", type=Path, default="outputs/lti_head_16ch.pt")
    args = parser.parse_args()

    loaders = build_dataloaders(DATA_CFG, seed=20260810)
    head = torch.load(args.head, map_location="cpu", weights_only=True)  # complex128, keep

    xs, ys = [], []
    for b in loaders["train"]:
        xs.append(feats(b))
        ys.append(b["target"].reshape(-1, N_POINTS * 2))
    Xtr = torch.cat(xs)
    Ytr = torch.cat(ys)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-12)
    Ymu, Ysd = Ytr.mean(0), Ytr.std(0).clamp_min(1e-12)

    model = ResidualNet(Xtr.shape[1], N_POINTS * 2, width=1536)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    def eval_split(split: str) -> tuple:
        num = torch.zeros((), dtype=torch.float64)
        den = torch.zeros((), dtype=torch.float64)
        num_h = torch.zeros((), dtype=torch.float64)
        res_bin = torch.zeros((N_BIN, 2), dtype=torch.float64)   # residual r energy per bin
        dist_bin = torch.zeros((N_BIN, 2), dtype=torch.float64)  # dist energy per bin
        err_bin = torch.zeros((N_BIN, 2), dtype=torch.float64)   # (head+MLP) err energy per bin
        with torch.no_grad():
            for b in loaders[split]:
                xx = b["input"].double()
                rr = b["target"].double()
                z = torch.fft.rfft(xx, dim=1)
                hp = torch.fft.irfft(
                    torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
                    n=N_POINTS, dim=1)
                dist = rr + hp
                rhat = (model((feats(b) - mu) / sd) * Ysd + Ymu).reshape(
                    len(b["input"]), N_POINTS, 2).double()
                pred = hp + rhat
                num += (pred - dist).square().sum()
                den += dist.square().sum()
                num_h += (hp - dist).square().sum()
                # per-bin energy of residual r and of the final error
                Rf = torch.fft.rfft(rr, dim=1).abs().square().mean(0)
                Df = torch.fft.rfft(dist, dim=1).abs().square().mean(0)
                Ef = torch.fft.rfft(pred - dist, dim=1).abs().square().mean(0)
                res_bin += Rf
                dist_bin += Df
                err_bin += Ef
        return (torch.sqrt(num / den).item(), torch.sqrt(num_h / den).item(),
                res_bin, dist_bin, err_bin)

    for split in ("train", "test"):
        rl2, rl2_h, res_bin, dist_bin, err_bin = eval_split(split)
        print(f"[{split}] head+MLP rl2 = {rl2:.6e} | head-only rl2 = {rl2_h:.6e}")

    # ---- residual spectrum: where does r live? ----
    for split, label in (("train", "train"), ("test", "test")):
        rl2, rl2_h, res_bin, dist_bin, err_bin = eval_split(split)
        res_tot = res_bin.sum().item()
        dist_tot = dist_bin.sum().item()
        print(f"[{split}] residual r energy share: tone bins {4,5,6,7} = "
              f"{res_bin[TONE_BINS].sum().item()/res_tot*100:.1f}% | "
              f"all bins except 4-7 = {(1-res_bin[TONE_BINS].sum().item()/res_tot)*100:.1f}%")
        # top-8 bins of residual energy
        order = res_bin.sum(1).argsort(descending=True)[:8]
        for k in order.tolist():
            print(f"    res bin {k} ({k*2} Hz): {res_bin[k].sum().item()/res_tot*100:.1f}% of r "
                  f"| err/dist ratio {torch.sqrt(err_bin[k].sum()/dist_bin[k].sum().clamp_min(1e-30)).item():.4f}")
        # error share of tone bins in the FINAL error
        err_tot = err_bin.sum().item()
        print(f"[{split}] final error energy share: tone bins = "
              f"{err_bin[TONE_BINS].sum().item()/err_tot*100:.1f}% | rest = "
              f"{(1-err_bin[TONE_BINS].sum().item()/err_tot)*100:.1f}%")


if __name__ == "__main__":
    main()
