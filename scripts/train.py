"""Parse experiment options, assemble components, and start OLED training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.dataloader import build_dataloaders, unwrap_dataset
from src.data.dataset import FIELD_PATHS
from src.models.factory import build_model
from src.training.factories import build_loss, build_optimizer, build_scheduler
from src.training.trainer import Trainer
from src.utils.checkpoint import load_checkpoint, restore_training_state
from src.utils.config import (
    create_experiment_dir,
    legacy_args,
    load_config,
    save_resolved_config,
)
from src.utils.device import select_device
from src.utils.logging import ExperimentLogger
from src.utils.paths import apply_data_root
from src.utils.seed import set_seed


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, validate, and test FNO1d on OLED data."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--keep-resume-learning-rate", action="store_true", default=None
    )
    parser.add_argument(
        "--freeze",
        type=str,
        default="",
        help="Comma-separated parameter-name prefixes to freeze "
        "(requires_grad=False); e.g. 'fc0,blocks.0' keeps the feature "
        "extractor fixed while new blocks train.",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--validate-every", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--optimizer", choices=("adam", "adamw"))
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--eta-min", type=float)
    parser.add_argument("--scheduler", choices=("cosine", "plateau"))
    parser.add_argument("--plateau-factor", type=float)
    parser.add_argument("--plateau-patience", type=int)
    parser.add_argument("--plateau-threshold", type=float)
    parser.add_argument("--plateau-threshold-mode", choices=("rel", "abs"))
    parser.add_argument("--plateau-cooldown", type=int)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"))
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-memory-cache", action="store_true")
    parser.add_argument("--loss", choices=("mse", "relative_l2"))
    parser.add_argument("--time-stride", type=int)
    time_fields = tuple(
        field for field in FIELD_PATHS if field not in {"time", "measurement_matrix_H"}
    )
    parser.add_argument("--input-fields", nargs="+", choices=time_fields)
    parser.add_argument("--target-fields", nargs="+", choices=time_fields)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--modes", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--embed-dim", type=int)
    parser.add_argument("--lift-dim", type=int)
    parser.add_argument("--num-blocks", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.keep_resume_learning_rate and args.resume is None:
        parser.error("--keep-resume-learning-rate requires --resume")
    return args


def config_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    sections: Dict[str, Dict[str, Any]] = {
        "data": {},
        "model": {},
        "loss": {},
        "optimizer": {},
        "scheduler": {},
        "training": {},
        "experiment": {},
    }
    mapping: Tuple[Tuple[str, str, str], ...] = (
        ("data_root", "data", "root"),
        ("batch_size", "data", "batch_size"),
        ("eval_batch_size", "data", "eval_batch_size"),
        ("time_stride", "data", "time_stride"),
        ("input_fields", "data", "input_fields"),
        ("target_fields", "data", "target_fields"),
        ("num_workers", "data", "num_workers"),
        ("max_train_samples", "data", "max_train_samples"),
        ("max_val_samples", "data", "max_val_samples"),
        ("max_test_samples", "data", "max_test_samples"),
        ("modes", "model", "modes"),
        ("width", "model", "width"),
        ("embed_dim", "model", "embed_dim"),
        ("lift_dim", "model", "lift_dim"),
        ("num_blocks", "model", "num_blocks"),
        ("loss", "loss", "name"),
        ("optimizer", "optimizer", "name"),
        ("learning_rate", "optimizer", "lr"),
        ("weight_decay", "optimizer", "weight_decay"),
        ("scheduler", "scheduler", "name"),
        ("eta_min", "scheduler", "eta_min"),
        ("plateau_factor", "scheduler", "factor"),
        ("plateau_patience", "scheduler", "patience"),
        ("plateau_threshold", "scheduler", "threshold"),
        ("plateau_threshold_mode", "scheduler", "threshold_mode"),
        ("plateau_cooldown", "scheduler", "cooldown"),
        ("epochs", "training", "epochs"),
        ("validate_every", "training", "validate_every"),
        ("grad_clip", "training", "grad_clip"),
        ("keep_resume_learning_rate", "training", "keep_resume_learning_rate"),
        ("amp", "training", "amp"),
        ("compile", "training", "compile"),
        ("output_dir", "experiment", "output_dir"),
        ("run_name", "experiment", "run_name"),
    )
    for attribute, section, key in mapping:
        value = getattr(args, attribute)
        if value is not None:
            sections[section][key] = str(value) if isinstance(value, Path) else value
    overrides: Dict[str, Any] = {
        key: value for key, value in sections.items() if value
    }
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.device is not None:
        overrides["device"] = args.device
    if args.no_save:
        overrides.setdefault("experiment", {})["save"] = False
    if args.overwrite:
        overrides.setdefault("experiment", {})["overwrite"] = True
    if args.no_memory_cache:
        overrides.setdefault("data", {})["memory_cache"] = False
    return overrides


def run_training(argv: Any = None) -> Dict[str, Any]:
    args = parse_args(argv)
    config = load_config(args.config, config_overrides(args))
    apply_data_root(config)  # resolve --data-root / $DATA_ROOT / default
    config["runtime"] = {
        "resume_checkpoint": str(args.resume.resolve()) if args.resume else None,
    }
    set_seed(int(config["seed"]))
    device = select_device(config["device"])
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    save_run = bool(config["experiment"].get("save", True))
    run_dir: Optional[Path] = create_experiment_dir(config) if save_run else None
    if run_dir is not None:
        config["experiment"]["resolved_dir"] = str(run_dir.resolve())
        save_resolved_config(config, run_dir / "config.yaml")

    loaders = build_dataloaders(config["data"], int(config["seed"]))
    train_dataset = unwrap_dataset(loaders["train"].dataset)
    model = build_model(
        config["model"], train_dataset.input_channels, train_dataset.target_channels
    ).to(device)
    if args.freeze:
        prefixes = [p for p in args.freeze.split(",") if p]
        for name, parameter in model.named_parameters():
            if any(name.startswith(prefix) for prefix in prefixes):
                parameter.requires_grad = False
        print(
            "[train] frozen params matching: "
            + ", ".join(f"*{p}*" for p in prefixes)
        )
    optimizer = build_optimizer(config["optimizer"], model)
    start_epoch = 0
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, map_location=device)
        start_epoch = restore_training_state(checkpoint, model, optimizer)
        if not config["training"].get("keep_resume_learning_rate", False):
            requested_lr = float(config["optimizer"]["lr"])
            for group in optimizer.param_groups:
                group["lr"] = requested_lr
                group["initial_lr"] = requested_lr

    scheduler = build_scheduler(
        config["scheduler"], optimizer, config["training"]["epochs"]
    )
    initial_learning_rate = float(optimizer.param_groups[0]["lr"])
    criterion = build_loss(config["loss"])
    legacy_arguments = legacy_args(config, run_dir)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with ExperimentLogger(run_dir) as logger:
        logger.log(
            {
                "device": str(device),
                "run_dir": str(run_dir) if run_dir else None,
                "additional_epochs": config["training"]["epochs"],
                "end_epoch": start_epoch + config["training"]["epochs"],
                "parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "resume_checkpoint": str(args.resume) if args.resume else None,
                "scheduler_restarted_on_resume": args.resume is not None,
                "scheduler": config["scheduler"]["name"],
                "optimizer": config["optimizer"]["name"],
                "initial_learning_rate": optimizer.param_groups[0]["lr"],
                "split_sizes": {
                    key: len(value.dataset) for key, value in loaders.items()
                },
                "start_epoch": start_epoch,
                "input_channels": train_dataset.input_channels,
                "input_fields": config["data"]["input_fields"],
                "target_channels": train_dataset.target_channels,
                "target_fields": config["data"]["target_fields"],
                "retained_time_points": len(train_dataset[0]["grid"]),
                "training_loss": config["loss"]["name"],
            }
        )
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            loaders=loaders,
            config=config,
            logger=logger,
            run_dir=run_dir,
            legacy_arguments=legacy_arguments,
            start_epoch=start_epoch,
            initial_learning_rate=initial_learning_rate,
        )
        return trainer.fit()


if __name__ == "__main__":
    run_training()
