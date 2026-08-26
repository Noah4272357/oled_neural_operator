"""Evaluate a checkpoint without constructing a trainer or retraining."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import build_dataloaders, unwrap_dataset
from src.models.factory import build_model
from src.training.factories import build_loss
from src.training.validate import validate
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.device import select_device
from src.utils.paths import apply_data_root


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _old_checkpoint_config(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    old = checkpoint["args"]
    overrides = {
        "seed": old.get("seed", 20260810),
        "data": {
            "root": old["data_root"],
            "input_fields": old["input_fields"],
            "target_fields": old["target_fields"],
            "time_stride": old["time_stride"],
            "batch_size": old["batch_size"],
            "eval_batch_size": old.get("eval_batch_size"),
            "num_workers": old["num_workers"],
        },
        "model": {
            "embed_dim": old["embed_dim"],
            "modes": old["modes"],
            "width": old["width"],
            "lift_dim": old["lift_dim"],
            "num_blocks": old["num_blocks"],
        },
        "loss": {"name": old.get("loss", "mse")},
    }
    return load_config(DEFAULT_CONFIG, overrides)


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    config = copy.deepcopy(checkpoint.get("config") or _old_checkpoint_config(checkpoint))
    if args.data_root is not None:
        config["data"]["root"] = str(args.data_root)
    if args.batch_size is not None:
        config["data"]["eval_batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    config["device"] = args.device
    apply_data_root(config)  # resolve --data-root / $DATA_ROOT / default

    device = select_device(args.device)
    loaders = build_dataloaders(config["data"], int(config["seed"]))
    dataset = unwrap_dataset(loaders[args.split].dataset)
    model = build_model(
        config["model"], dataset.input_channels, dataset.target_channels
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = validate(model, loaders[args.split], build_loss(config["loss"]), device)
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "split": args.split,
        "samples": len(loaders[args.split].dataset),
        "metrics": metrics,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
