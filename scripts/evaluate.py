"""Acceptance STEP 5: evaluate the best checkpoint on the full test split.

Runs every test sample, at every retained time point, on both target channels,
and reports the **global** relative L2

    sqrt( sum_all (prediction - target)^2 ) / sqrt( sum_all target^2 )

accumulated over the whole split at once -- not a per-sample relative L2
averaged afterwards.

Two values are recorded, because they differ and the difference is documented:

* ``relative_l2`` -- computed exactly as ``src/training/metrics.py`` does,
  which promotes to float32 before accumulating.  This is the official number.
* ``relative_l2_float64`` -- the same prediction accumulated in float64.  This
  is the true model error; the float32 accumulation is slightly conservative
  (it over-reports by ~1.27x on the recorded run).

``pass`` requires **both** to be below the threshold.

Usage:
    python scripts/evaluate.py --run-dir runs/acceptance_20260918-120000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.dataset import OLEDNeuralOperatorDataset  # noqa: E402
from src.data.preprocessing import build_preprocessor  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.training.metrics import RegressionMetrics  # noqa: E402
from src.utils.checkpoint import load_checkpoint, load_model_state  # noqa: E402
from src.utils.device import select_device  # noqa: E402

DTYPES = {"float32": torch.float32, "float64": torch.float64}
DEFAULT_THRESHOLD = 1.0e-4


def build_split(config: Dict[str, Any], split: str) -> OLEDNeuralOperatorDataset:
    data = config["data"]
    return OLEDNeuralOperatorDataset(
        Path(data["root"]),
        split,
        input_fields=data["input_fields"],
        target_fields=data["target_fields"],
        time_start=data.get("time_start", 0),
        time_stop=data.get("time_stop"),
        time_stride=data["time_stride"],
        dtype=DTYPES[str(data.get("dtype", "float32")).lower()],
        include_metadata=False,
        transform=build_preprocessor(data.get("preprocessing", {"name": "none"})),
        transform_dtype=(
            DTYPES[str(data["transform_dtype"]).lower()]
            if data.get("transform_dtype")
            else None
        ),
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="default: <run-dir>/best_model.pt")
    parser.add_argument("--config", type=Path, default=None,
                        help="default: <run-dir>/config.yaml, else the checkpoint's own config")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="cpu")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None,
                        help="default: <run-dir>/test_metrics.json")
    parser.add_argument("--predictions", type=Path, default=None,
                        help="default: <run-dir>/predictions.npz")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    run_dir = args.run_dir
    checkpoint_path = args.checkpoint or (
        Path(run_dir) / "best_model.pt" if run_dir else None
    )
    if checkpoint_path is None:
        raise SystemExit("Provide --run-dir or --checkpoint.")
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_file():
        raise SystemExit(f"No checkpoint at {checkpoint_path}.")

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    if args.config is not None:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    elif run_dir is not None and (Path(run_dir) / "config.yaml").is_file():
        config = json.loads((Path(run_dir) / "config.yaml").read_text(encoding="utf-8"))
    if args.data_root is not None:
        config["data"]["root"] = str(args.data_root)

    threshold = (
        args.threshold
        if args.threshold is not None
        else float(config.get("acceptance", {}).get("threshold", DEFAULT_THRESHOLD))
    )

    device = select_device(args.device)
    dataset = build_split(config, args.split)
    batch_size = int(config["data"].get("eval_batch_size") or config["data"]["batch_size"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(
        config["model"], dataset.input_channels, dataset.target_channels
    ).to(device)
    model.load_state_dict(load_model_state(checkpoint))
    model.eval()

    official = RegressionMetrics()
    squared_error = 0.0
    squared_target = 0.0
    predictions, targets = [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["input"].to(device)
            grid = batch["grid"].to(device)
            target = batch["target"].to(device)
            prediction = model(inputs, grid)
            official.update(prediction, target)
            difference = prediction.double() - target.double()
            squared_error += float(difference.square().sum())
            squared_target += float(target.double().square().sum())
            predictions.append(prediction.double().cpu().numpy())
            targets.append(target.double().cpu().numpy())
    inference_seconds = time.perf_counter() - started

    metrics = official.compute(inference_seconds)
    relative_l2_float64 = float(np.sqrt(squared_error / squared_target))
    time_points = int(dataset[0]["target"].shape[0])
    result: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "split": args.split,
        "samples": len(dataset),
        "time_points": time_points,
        "target_channels": int(dataset.target_channels),
        "elements": int(len(dataset) * time_points * dataset.target_channels),
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "relative_l2": metrics["relative_l2"],
        "relative_l2_float64": relative_l2_float64,
        "threshold": threshold,
        "pass": bool(
            metrics["relative_l2"] < threshold and relative_l2_float64 < threshold
        ),
        "inference_seconds": inference_seconds,
    }

    if run_dir is not None:
        run_dir = Path(run_dir)
    output = args.output or (run_dir / "test_metrics.json" if run_dir else None)
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    prediction_path = args.predictions or (
        run_dir / "predictions.npz" if run_dir else None
    )
    if prediction_path is not None:
        Path(prediction_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            prediction_path,
            predictions=np.concatenate(predictions, axis=0),
            targets=np.concatenate(targets, axis=0),
            split=np.array(args.split),
        )

    print(
        "\n".join(
            [
                "",
                "Test evaluation",
                f"  Splits              : {args.split}",
                f"  Samples             : {result['samples']}",
                f"  Time points         : {result['time_points']}",
                f"  Target channels     : {result['target_channels']}",
                f"  Elements            : {result['elements']}",
                f"  MSE                 : {result['mse']:.7e}",
                f"  RMSE                : {result['rmse']:.7e}",
                "",
                f"  GLOBAL RELATIVE L2  : {result['relative_l2']:.7e}"
                "   (official, float32 accumulation)",
                f"  GLOBAL RELATIVE L2  : {result['relative_l2_float64']:.7e}"
                "   (float64 direct recomputation)",
                f"  ACCEPTANCE LIMIT    : {threshold:.6e}",
                f"  RESULT              : {'PASS' if result['pass'] else 'FAIL'}",
            ]
        ),
        flush=True,
    )
    if output is not None:
        print(f"\nwrote {output}")
    if prediction_path is not None:
        print(f"wrote {prediction_path}")
    return result


if __name__ == "__main__":
    main()
