"""LS f_dist ~ [u, s, ṡ, s̈] (the 16ch diff_features set) in the TIME DOMAIN.

fit_lti_head_16ch.py fits the same features per-FT-bin (periodic hypothesis
-- boundary contamination).  Here the same 16 features are regressed with
SHARED time-domain weights (32 real params), interior rows only.  If the
dynamics are a time-domain linear function of (u, s, ṡ, s̈) -- as the
physics f_dist = M q̈ + C q̇ + K q - B u would predict -- this should reach
well below 1e-3, unlike the 1.5e-3 periodic head.

Usage (project .venv, CPU):
  .venv/bin/python scripts/analysis/verify_full_state_ls.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import build_dataloaders

DATA_ROOT = "~/data/neural_operator_3"


def main() -> None:
    data_cfg = {
        "root": DATA_ROOT,
        "input_fields": ["force", "encoder_displacement"],
        "target_fields": ["disturbance"],
        "time_start": 0, "time_stop": None, "time_stride": 1,
        "preprocessing": {"name": "diff_features", "dt": 0.001, "channels": [4, 5, 6, 7]},
        "memory_cache": False,
        "batch_size": 256, "eval_batch_size": 256, "num_workers": 0,
        "max_train_samples": None, "max_val_samples": None, "max_test_samples": None,
    }
    loaders = build_dataloaders(data_cfg, seed=20260810)

    def ls(loader, max_fit, name):
        X, Y = [], []
        for bi, b in enumerate(loader):
            if bi >= max_fit:
                break
            X.append(b["input"].double()[:, 4:497, :].reshape(-1, 16))
            Y.append(b["target"].double()[:, 4:497, :].reshape(-1, 2))
        X = torch.cat(X)
        Y = torch.cat(Y)
        W = torch.linalg.lstsq(X, Y).solution  # (16,2)
        num = den = 0.0
        numb = denb = 0.0
        for bi, b in enumerate(loader):
            if bi < max_fit or bi >= max_fit + 5:
                continue
            Xt = b["input"].double()
            Yt = b["target"].double()
            p = Xt @ W
            e = p - Yt
            num += e[:, 4:497, :].square().sum().item()
            den += Yt[:, 4:497, :].square().sum().item()
            numb += e[:, [0, 1, 2, 3, 497, 498, 499, 500], :].square().sum().item()
            denb += Yt[:, [0, 1, 2, 3, 497, 498, 499, 500], :].square().sum().item()
        rl2 = np.sqrt(num / den)
        rl2b = np.sqrt(numb / denb)
        print(f"[{name}] time-domain LS(16ch) interior rl2 = {rl2:.6e} | "
              f"boundary-8 rl2 = {rl2b:.6e}")
        print(f"    W x-row = {W[:, 0].tolist()}")
        return W

    W = ls(loaders["train"], 15, "train")
    # proper test evaluation with the train-fitted W:
    num = den = numi = deni = 0.0
    for b in loaders["test"]:
        Xt = b["input"].double()
        Yt = b["target"].double()
        p = Xt @ W
        e = p - Yt
        num += e.square().sum().item()
        den += Yt.square().sum().item()
        numi += e[:, 4:497, :].square().sum().item()
        deni += Yt[:, 4:497, :].square().sum().item()
    print(f"[test] full-window rl2 = {np.sqrt(num/den):.6e} | "
          f"interior rl2 = {np.sqrt(numi/deni):.6e}")


if __name__ == "__main__":
    main()
