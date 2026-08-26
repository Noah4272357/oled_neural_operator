"""Official test evaluation for the m64 residual-trained 8ch network.

The m64 model is a plain FNO1d trained on the residual target
(derived dataset, ``build_residual_targets.py``): its prediction is the
residual ``r``, so the full 8ch prediction is

    pred = FNO(x16) + irfft(M @ rfft(x16))          (M: lti_head_16ch.pt)

and the official metrics (relative_l2 / mse / rmse, global aggregation,
matching scripts/evaluate.py semantics) are computed against the ORIGINAL
test split disturbance.  The linear head is the closed-form per-bin map
fitted on the train split (fit_lti_head_16ch.py), never touched by the
network training.

Usage (project .venv, CPU):
  .venv/bin/python scripts/analysis/evaluate_residual_net.py \
      --checkpoint experiments/m64_*/best_model.pt \
      --head outputs/lti_head_16ch.pt \
      --data-root ~/data/neural_operator_3
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

N_POINTS = 501
N_BIN = N_POINTS // 2 + 1
DATA_CONFIG = {
    "root": None,  # filled from --data-root
    "input_fields": ["force", "encoder_displacement"],
    "target_fields": ["disturbance"],
    "time_start": 0,
    "time_stop": None,
    "time_stride": 1,
    "preprocessing": {"name": "diff_features", "dt": 0.001, "channels": [4, 5, 6, 7]},
    "memory_cache": False,
    "batch_size": 64,
    "eval_batch_size": 64,
    "num_workers": 0,
    "max_train_samples": None,
    "max_val_samples": None,
    "max_test_samples": None,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--head", default="outputs/lti_head_16ch.pt", type=Path)
    parser.add_argument("--data-root", default="~/data/neural_operator_3", type=Path)
    args = parser.parse_args()

    ckpt = load_checkpoint(Path(args.checkpoint).expanduser())
    cfg = ckpt["config"]
    model_cfg = dict(cfg["model"])
    input_channels = int(model_cfg.pop("input_channels", 16))
    model = build_model(
        model_cfg, input_channels, int(cfg.get("target_channels", 2))
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mk = torch.load(args.head, map_location="cpu", weights_only=True)  # complex128, keep
    assert mk.shape == (N_BIN, 2, 16), mk.shape

    data_cfg = dict(DATA_CONFIG)
    data_cfg["root"] = str(Path(args.data_root).expanduser())
    loaders = build_dataloaders(data_cfg, seed=20260810)

    n_total = 200 * N_POINTS * 2
    num = torch.zeros((), dtype=torch.float64)
    den = torch.zeros((), dtype=torch.float64)
    with torch.no_grad():
        for b in loaders["test"]:
            x = b["input"].double()          # (B, 501, 16)
            grid = b["grid"].double()        # (B, 501)
            out = model(x.float(), grid.float())  # FNO runs float32
            zft = torch.fft.rfft(x, dim=1).to(torch.complex128)
            head_ft = torch.einsum("bki,koi->bko", zft, mk)
            head_pred = torch.fft.irfft(head_ft, n=N_POINTS, dim=1)
            pred = out.double() + head_pred
            t = b["target"].double()
            num += (pred - t).square().sum()
            den += t.square().sum()

    rl2 = (num / den).sqrt().item()
    mse = (num / n_total).item()
    rmse = (num / n_total).sqrt().item()
    result = {
        "checkpoint": str(args.checkpoint),
        "split": "test",
        "samples": 200,
        "metrics": {"mse": mse, "relative_l2": rl2, "rmse": rmse},
        "head": str(args.head),
    }
    print(result)
    with open(Path(args.checkpoint).parent / "official_test_residual.json", "w") as f:
        import json

        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
