"""How much of the reported test error is the metric's own float32 floor?

`RegressionMetrics.update` (src/training/metrics.py:29-30) forces
`prediction.detach().float() - target.float()`, so the official test
relative_l2 is *always* computed in float32.  For a model whose true error is
near 1e-8 that measurement precision becomes the reported number.

This script recomputes the delivered checkpoint's test error three ways and
prints them side by side:

  float64            the true error (double()-ed prediction and target)
  float32 / metrics  the official path -- float32 accumulator, per-batch +=
  float32 / sums64   same float32 difference, but per-batch sums accumulated
                     in float64 (an independent recomputation)

The last two agree to ~7 significant digits and differ in the 8th; neither is
the true error.  Run with the project venv from the repo root:

    PYTHONPATH=. .venv/bin/python scripts/analysis/f64_metric_floor.py

Read-only: loads the checkpoint and the test split, writes nothing.

Background: experiments/analysis/2026-09-18/ld-dense-campaign.md §9.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import build_dataloaders  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.training.metrics import RegressionMetrics  # noqa: E402
from src.utils.checkpoint import load_checkpoint  # noqa: E402

DEFAULT_RUN = Path("experiments/ld_dense_final")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN,
                        help="run directory holding best_model.pt / official_test.json")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    checkpoint = args.run / "best_model.pt"
    # Project loader, not a bare torch.load: v2 checkpoints carry the resolved
    # config alongside the tensors, so weights_only=True cannot read them.
    blob = load_checkpoint(checkpoint)
    config = blob["config"]

    model = build_model(config["model"], 16, 2)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()

    loaders = build_dataloaders(config["data"], seed=int(config.get("seed", 0)))
    loader = loaders[args.split]

    official_metrics = RegressionMetrics()
    squared_error_f64 = 0.0
    squared_target_f64 = 0.0
    squared_error_sums64 = 0.0
    squared_target_sums64 = 0.0

    with torch.no_grad():
        for batch in loader:
            inputs, target, grid = batch["input"], batch["target"], batch["grid"]
            prediction = model(inputs, grid)

            # The official path, verbatim (float32 accumulator inside).
            official_metrics.update(prediction, target)

            # Same float32 difference, but summed per batch in float64.
            difference = prediction.detach().float() - target.float()
            squared_error_sums64 += float(difference.square().sum())
            squared_target_sums64 += float(target.float().square().sum())

            # The true error, in float64 throughout.
            difference64 = prediction.detach().double() - target.double()
            squared_error_f64 += float(difference64.square().sum())
            squared_target_f64 += float(target.double().square().sum())

    official = official_metrics.compute(0.0)["relative_l2"]
    sums64 = (squared_error_sums64 / squared_target_sums64) ** 0.5
    true64 = (squared_error_f64 / squared_target_f64) ** 0.5

    print(f"checkpoint      : {checkpoint}")
    print(f"split / samples : {args.split} / {len(loader.dataset)}")
    print()
    print(f"float64 (true)          : {true64:.16e}")
    print(f"float32 (metrics.py)    : {official:.16e}")
    print(f"float32 (float64 sums)  : {sums64:.16e}")
    print()
    print(f"floor / true            : {official / true64:.4f}x")
    digits = -math.log10(abs(official - sums64) / official)
    print(f"metrics vs sums64       : agree to {digits:.1f} significant digits")
    print(f"margin vs 1e-4 (true)   : {1e-4 / true64:.0f}x")

    official_json = args.run / "official_test.json"
    if official_json.exists() and args.split == "test":
        recorded = json.loads(official_json.read_text())["metrics"]["relative_l2"]
        print()
        print(f"official_test.json      : {recorded:.16e}")
        print(f"bit-identical to replay : {recorded == official}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
