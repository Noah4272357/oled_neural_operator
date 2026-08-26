"""Verify the time-domain closed-form disturbance map.

Physics:  M q̈ = B u + f_dist  (K=C=0),  s = H q (enc, ideal).
With q̈ = H⁺ s̈ the disturbance is a *time-domain* linear map of the
measured signals:

    f_dist(t) = A s̈(t) + C u(t)        (A: 2x4, C: 2x4, 16 real params)

The frequency-domain per-bin head (fit_lti_head_16ch.py) fits the same map
under a periodic (rfft) hypothesis, which contaminates every bin with the
window-boundary jump -- that is the 79% boundary residual.  A time-domain
fit on interior points has no such leakage and, if the physics is exact,
should reach ~1e-6 on interior points.

Boundary points (t=0,1,497,498,499,500) need s̈[-1] (out of window); we
report them separately (head-template diff vs one-sided extrapolation).

Usage (project .venv, CPU):
  .venv/bin/python scripts/analysis/verify_closed_form.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import build_dataloaders

DATA_ROOT = Path("~/data/neural_operator_3").expanduser()
BOUNDARY = [0, 1, 2, 3, 497, 498, 499, 500]


def main() -> None:
    manifest = json.loads((DATA_ROOT / "dataset_manifest.json").read_text())
    H = torch.tensor(manifest["model"]["encoder_measurement_matrix_H"], dtype=torch.float64)
    B = torch.tensor(manifest["model"]["actuator_mapping_B"], dtype=torch.float64)
    Hp = torch.linalg.pinv(H)  # (3,4)
    print(f"H {tuple(H.shape)} rank {torch.linalg.matrix_rank(H).item()}, "
          f"H⁺H error {torch.linalg.matrix_norm(Hp @ H - torch.eye(3, dtype=torch.float64)).item():.2e}")

    data_cfg = {
        "root": str(DATA_ROOT),
        "input_fields": ["force", "encoder_displacement"],
        "target_fields": ["disturbance"],
        "time_start": 0,
        "time_stop": None,
        "time_stride": 1,
        "preprocessing": {"name": "none"},  # raw fields; we differentiate ourselves
        "memory_cache": False,
        "batch_size": 256,
        "eval_batch_size": 256,
        "num_workers": 0,
        "max_train_samples": None,
        "max_val_samples": None,
        "max_test_samples": None,
    }
    loaders = build_dataloaders(data_cfg, seed=20260810)

    def sdotdot(s):
        # central 2nd difference, template y[0]=y[2], y[-1]=y[-3] (matches
        # diff_features exactly); interior t=2..498 is exact O(dt^2).
        d = torch.zeros_like(s)
        d[1:-1] = (s[2:] - 2.0 * s[1:-1] + s[:-2]) / 1e-6
        d[0] = d[2]
        d[-1] = d[-3]
        return d

    # Least squares on interior points (t=4..496) across the train split:
    #   [s̈, u] -> f_dist  (A 2x4, C 2x4)
    Aa = []  # design matrix rows
    bb = []
    n_used = 0
    for batch in loaders["train"]:
        s = batch["input"][:, :, 4:8].double()          # encoder_displacement (B,501,4)
        u = batch["input"][:, :, 0:4].double()          # force (B,501,4)
        f = batch["target"].double()                    # (B,501,2)
        sdd = sdotdot(s)
        feats = torch.cat([sdd, u], dim=-1)             # (B,501,8)
        # interior indices 4..496
        X = feats[:, 4:497, :].reshape(-1, 8)
        Y = f[:, 4:497, :].reshape(-1, 2)
        Aa.append(X)
        bb.append(Y)
        n_used += X.shape[0]
    Aa = torch.cat(Aa)
    bb = torch.cat(bb)
    print(f"LS design: {n_used} interior point-rows")
    W = torch.linalg.lstsq(Aa, bb).solution  # (8,2)
    A, C = W[:4].T.contiguous(), W[4:].T.contiguous()  # (2,4) each
    print(f"A (s̈ -> f) row0 = {A[0].tolist()}")
    print(f"C (u -> f) row0 = {C[0].tolist()}")

    # Evaluate per split
    for split in ("train", "val", "test"):
        num = torch.zeros((), dtype=torch.float64)
        den = torch.zeros((), dtype=torch.float64)
        num_in = torch.zeros((), dtype=torch.float64)
        den_in = torch.zeros((), dtype=torch.float64)
        num_b = torch.zeros((), dtype=torch.float64)
        den_b = torch.zeros((), dtype=torch.float64)
        for batch in loaders[split]:
            s = batch["input"][:, :, 4:8].double()
            u = batch["input"][:, :, 0:4].double()
            f = batch["target"].double()
            sdd = sdotdot(s)
            pred = sdd @ A.T + u @ C.T  # (B,501,2)
            e = pred - f
            num += e.square().sum()
            den += f.square().sum()
            num_in += e[:, 4:497, :].square().sum()
            den_in += f[:, 4:497, :].square().sum()
            num_b += e[:, BOUNDARY, :].square().sum()
            den_b += f[:, BOUNDARY, :].square().sum()
        rl2 = torch.sqrt(num / den).item()
        rl2_in = torch.sqrt(num_in / den_in).item()
        rl2_b = torch.sqrt(num_b / den_b).item()
        print(f"[{split}] full-window rl2 = {rl2:.3e} | interior (4..496) rl2 = "
              f"{rl2_in:.3e} | boundary-8 rl2 = {rl2_b:.3e}")


if __name__ == "__main__":
    main()
