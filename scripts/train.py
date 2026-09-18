"""Test STEP 4: train the surrogate from a fresh initialization.

Loads the train-only basis produced by ``scripts/fit_basis.py`` from the run
directory, initializes a new model, and runs gradient descent (SGD) with
periodic validation.  The best checkpoint is selected by validation.

There is no resume path: the test run always starts from scratch.

Usage:
    python scripts/train.py --config configs/test.yaml \
        --run-dir runs/test_20260918-120000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from src.data.dataloader import build_dataloaders, unwrap_dataset  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.training.factories import (  # noqa: E402
    build_loss,
    build_optimizer,
    build_scheduler,
)
from src.training.trainer import Trainer  # noqa: E402
from src.utils.config import (  # noqa: E402
    load_config,
    resolve_run_dir,
    save_resolved_config,
)
from src.utils.device import select_device  # noqa: E402
from src.utils.logging import RunLogger  # noqa: E402
from src.utils.paths import apply_data_root  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

DEFAULT_CONFIG = Path("configs/test.yaml")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--basis", type=Path, default=None,
                        help="override the basis path (default <run-dir>/spectral_basis.pt)")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--validate-every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="smoke runs only: cap the training split")
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--no-save", action="store_true",
                        help="run without writing any run artifact")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    config = load_config(args.config)

    if args.data_root is not None:
        config["data"]["root"] = str(args.data_root)
    if args.device is not None:
        config["device"] = args.device
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.validate_every is not None:
        config["training"]["validate_every"] = args.validate_every
    if args.max_train_samples is not None:
        config["data"]["max_train_samples"] = args.max_train_samples
    if args.max_val_samples is not None:
        config["data"]["max_val_samples"] = args.max_val_samples
    if args.no_save:
        config["experiment"]["save"] = False
    if args.seed is not None:
        config["seed"] = args.seed
    apply_data_root(config)

    run_dir = None
    if config["experiment"].get("save", True):
        run_dir = resolve_run_dir(config, args.run_dir)
    elif args.run_dir is not None:
        run_dir = Path(args.run_dir)

    basis_path = args.basis
    if basis_path is None:
        if run_dir is None:
            raise SystemExit("--basis is required when --no-save is used without --run-dir")
        basis_path = run_dir / "spectral_basis.pt"
    basis_path = Path(basis_path).expanduser()
    if not basis_path.is_file():
        raise SystemExit(
            f"No spectral basis at {basis_path}. Run scripts/fit_basis.py first."
        )
    config["model"]["basis_path"] = str(basis_path)

    seed = int(config["seed"])
    set_seed(seed)
    device = select_device(config["device"])
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if run_dir is not None:
        save_resolved_config(config, run_dir / "config.yaml")

    loaders = build_dataloaders(config["data"], seed)
    train_dataset = unwrap_dataset(loaders["train"].dataset)
    model = build_model(
        config["model"], train_dataset.input_channels, train_dataset.target_channels
    ).to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(
        "\n".join(
            [
                "Test training run",
                f"  Run directory       : {run_dir}",
                f"  Dataset             : {config['data']['root']}",
                f"  Device              : {device}",
                f"  Basis               : {basis_path}",
                f"  Model               : {config['model']['name']}",
                f"  Trainable parameters: {parameters}",
                f"  Input channels      : {train_dataset.input_channels}"
                f" (raw {len(config['data']['input_fields'])} field group(s),"
                f" preprocessing {config['data']['preprocessing']['name']})",
                f"  Target channels     : {train_dataset.target_channels}",
                f"  Splits (train/val)  : "
                f"{len(loaders['train'].dataset)}/{len(loaders['val'].dataset)}",
                f"  Optimizer           : {config['optimizer']['name']}"
                f" lr={config['optimizer']['lr']}"
                f" momentum={config['optimizer'].get('momentum', 0.0)}",
                f"  Loss                : {config['loss']['name']}",
                "",
            ]
        ),
        flush=True,
    )

    if config["data"].get("max_train_samples") is not None:
        print(
            "\n".join(
                [
                    "",
                    "WARNING: --max-train-samples caps the training split.",
                    "  The canonical learning rate is calibrated for the full",
                    f"  {config['data']['batch_size']}-sample batch: MSE averages over the",
                    "  batch, so the loss Hessian scales as 1/numel. On a small",
                    "  subset the canonical lr will diverge. This mode is for",
                    "  wiring checks only, not for a test result.",
                    "",
                ]
            ),
            flush=True,
        )

    optimizer = build_optimizer(config["optimizer"], model)
    scheduler = build_scheduler(config["scheduler"], optimizer, config["training"]["epochs"])
    criterion = build_loss(config["loss"])

    with RunLogger(run_dir) as logger:
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
        )
        summary = trainer.fit()

    if run_dir is not None:
        print(
            "\n".join(
                [
                    "",
                    "Training artifacts",
                    f"  {run_dir / 'config.yaml'}",
                    f"  {run_dir / 'history.json'}",
                    f"  {run_dir / 'metrics.csv'}",
                    f"  {run_dir / 'train.log'}",
                    f"  {run_dir / 'best_model.pt'}",
                    f"  {run_dir / 'last_model.pt'}",
                ]
            ),
            flush=True,
        )
    return summary


if __name__ == "__main__":
    main()
