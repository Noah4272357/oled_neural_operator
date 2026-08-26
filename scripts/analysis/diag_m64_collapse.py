"""Diag the m64 residual-FNO collapse (pred ~ 0 plateau).

Questions answered:
  1. Is the trained checkpoint's prediction actually ~ 0?
  2. Where does the gradient flow at that point (per-parameter-group norms)?
  3. Is the boundary residual learnable *in principle*?  (linear probe:
     residual at the 8 boundary points regressed on the boundary-neighbourhood
     input, fit on train, measured on test -- if a linear probe already
     explains most of the 79% boundary energy, the residual is a
     deterministic function of the window, and the failure is architectural;
     if not, the boundary value depends on out-of-window info and no 16ch
     model can reach 1e-4 on this task definition.)

Usage (project .venv, CPU):
  .venv/bin/python scripts/analysis/diag_m64_collapse.py \
      --checkpoint experiments/m64_fno16ch_residual_s20260810_01/best_model.pt \
      --data-root experiments/m63_residual_targets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import build_dataloaders
from src.models.factory import build_model
from src.utils.checkpoint import load_checkpoint

BOUNDARY = [0, 1, 2, 3, 497, 498, 499, 500]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default="experiments/m63_residual_targets")
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    cfg = ckpt["config"]
    model_cfg = dict(cfg["model"])
    in_ch = int(model_cfg.pop("input_channels", 16))
    model = build_model(model_cfg, in_ch, int(cfg.get("target_channels", 2)))
    model.load_state_dict(ckpt["model_state_dict"])

    data_cfg = dict(cfg["data"])
    data_cfg["root"] = str(args.data_root)
    data_cfg["num_workers"] = 0
    data_cfg["max_train_samples"] = 256
    data_cfg["max_val_samples"] = 128
    data_cfg["max_test_samples"] = 128
    loaders = build_dataloaders(data_cfg, seed=20260810)

    # ---- 1. prediction stats on a train batch ----
    model.eval()
    b = next(iter(loaders["train"]))
    x, t, g = b["input"], b["target"], b["grid"]
    with torch.no_grad():
        p = model(x, g)
    n = p.numel()
    se = (p - t).square().sum().item()
    st = t.square().sum().item()
    stb = t[:, BOUNDARY, :].square().sum().item()
    print(f"[1] pred  shape={tuple(p.shape)} mean={p.mean():.3e} std={p.std():.3e} "
          f"nan={torch.isnan(p).any().item()}")
    print(f"    target mean={t.mean():.3e} std={t.std():.3e} nan={torch.isnan(t).any().item()}")
    print(f"    rl2(pred)={np.sqrt(se / st):.6f}  (zero-output would be 1.0)")
    print(f"    boundary energy / total target = {stb / st:.4f}")
    # prediction spectrum vs target spectrum
    pf = torch.fft.rfft(p, dim=1).abs().square().mean(dim=(0, 2))
    tf = torch.fft.rfft(t, dim=1).abs().square().mean(dim=(0, 2))
    print(f"    pred spectrum: top-3 bins energy {pf.topk(3).values.tolist()}")
    print(f"    tgt spectrum: top-3 bins energy {tf.topk(3).values.tolist()}")
    print(f"    pred per-bin energy / tgt per-bin energy (mean) = "
          f"{(pf.sum() / tf.sum()).item():.4f}")

    # ---- 2. gradient flow at the collapsed point ----
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    loss = torch.linalg.vector_norm(p - t) / torch.linalg.vector_norm(t).clamp_min(1e-30)
    # re-forward inside autograd (p above is under no_grad)
    p2 = model(x, g)
    loss = torch.linalg.vector_norm(p2 - t) / torch.linalg.vector_norm(t).clamp_min(1e-30)
    loss.backward()
    print(f"[2] loss at collapsed point = {loss.item():.6f}")
    total_g = 0.0
    zero_g = []
    for name, prm in model.named_parameters():
        if prm.grad is None:
            zero_g.append(name)
            continue
        gn = prm.grad.detach().float().norm().item()
        total_g += gn * gn
        if prm.dim() and prm.grad.abs().max().item() == 0:
            zero_g.append(name)
    print(f"    global grad norm = {np.sqrt(total_g):.3e}")
    print(f"    params with NO gradient: {zero_g if zero_g else 'none'}")

    # ---- 3. learnability probe: boundary residual vs boundary input ----
    # Features: the 16ch input on a small neighbourhood of each boundary edge,
    # plus the same neighbourhood of the target-edge itself is NOT allowed
    # (that would leak); we regress r[edge] on x[edge±k] only.
    def collect(split: str, max_n: int) -> tuple:
        Xs, Ys = [], []
        for i, b in enumerate(loaders[split]):
            if i * 64 >= max_n:
                break
            xx = b["input"].double()  # (B, 501, 16)
            rr = b["target"].double()  # (B, 501, 2)
            # left edge: window 0..7, probe point 0 ; right edge: window 493..500, probe point 500
            for name, sl, pi in (("L", slice(0, 8), 0), ("R", slice(493, 501), 500)):
                Xs.append(xx[:, sl, :].reshape(len(xx), -1))
                Ys.append(rr[:, pi, :])
        return torch.cat(Xs), torch.cat(Ys)

    Xtr, Ytr = collect("train", 256)
    Xte, Yte = collect("test", 128)
    # per-edge-channel closed form (skip trivial constant? include bias via ones)
    Xa = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1)], dim=1)  # (N, 8*16+1)
    W = torch.linalg.lstsq(Xa, Ytr).solution
    pred_te = torch.cat([Xte, torch.ones(Xte.shape[0], 1)], dim=1) @ W
    se_probe = (pred_te - Yte).square().sum().item()
    st_probe = Yte.square().sum().item()
    print(f"[3] linear probe: boundary-point residual r[edge] ~ x16[edge±4]")
    print(f"    test rl2 on probed points = {np.sqrt(se_probe / st_probe):.4f} "
          f"(0.0 => fully learnable from window; ~1.0 => out-of-window info)")
    # also probe WITHOUT the differential channels (pure 8ch) to isolate d1/d2 value
    Xa8 = torch.cat([Xtr[:, : 8 * 8], torch.ones(Xtr.shape[0], 1)], dim=1)
    W8 = torch.linalg.lstsq(Xa8, Ytr).solution
    pred8 = torch.cat([Xte[:, : 8 * 8], torch.ones(Xte.shape[0], 1)], dim=1) @ W8
    se8 = (pred8 - Yte).square().sum().item()
    print(f"    (same probe, 8ch only) test rl2 = {np.sqrt(se8 / st_probe):.4f}")


if __name__ == "__main__":
    main()
