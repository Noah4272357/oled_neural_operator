"""Gradient training, validation, scheduling, and best-checkpoint selection.

The acceptance pipeline trains once, from a fresh initialization, and never
resumes: there is no resume path here by design.  Terminal output is one
compact block per epoch so the acceptance session can read it on screen.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.utils.checkpoint import save_checkpoint
from src.utils.logging import RunLogger

from .factories import LossFunction
from .train_epoch import train_one_epoch
from .validate import validate


def _format_block(
    epoch: int, epochs: int, train: Mapping[str, float],
    validation: Optional[Mapping[str, float]], best_epoch: int,
    best_score: float, learning_rate: float,
) -> str:
    lines = [
        f"Epoch {epoch:03d}/{epochs}",
        f"  Train MSE         : {train['mse']:.7e}",
        f"  Train relative L2 : {train['relative_l2']:.7e}",
        f"  Learning rate     : {learning_rate:.6e}",
    ]
    if validation is not None:
        lines += [
            f"  Val   MSE         : {validation['mse']:.7e}",
            f"  Val   relative L2 : {validation['relative_l2']:.7e}",
            f"  Best epoch        : {best_epoch} (val MSE {best_score:.7e})",
        ]
    return "\n".join(lines)


class Trainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: Any,
        criterion: LossFunction,
        device: torch.device,
        loaders: Mapping[str, DataLoader],
        config: Mapping[str, Any],
        logger: RunLogger,
        run_dir: Optional[Path],
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.loaders = loaders
        self.config = config
        self.logger = logger
        self.run_dir = run_dir

    def _save(
        self,
        filename: str,
        epoch: int,
        validation: Mapping[str, float],
        best: float,
    ) -> None:
        if self.run_dir is None:
            return
        save_checkpoint(
            self.run_dir / filename,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            validation=validation,
            best_metric=best,
            config=self.config,
        )

    def fit(self) -> Dict[str, Any]:
        training = self.config["training"]
        loss_name = self.config["loss"]["name"]
        epochs = int(training["epochs"])
        validate_every = int(training["validate_every"])

        best_score = float("inf")
        best_epoch = 0
        best_validation: Dict[str, float] = {}
        best_state: Optional[Dict[str, torch.Tensor]] = None
        last_validation: Dict[str, float] = {}
        history = []
        started = time.perf_counter()

        print(
            f"Training {self.model.__class__.__name__} on {self.device} "
            f"for {epochs} epochs (validate every {validate_every})",
            flush=True,
        )
        for epoch in range(1, epochs + 1):
            train_metrics = train_one_epoch(
                self.model,
                self.loaders["train"],
                self.optimizer,
                self.criterion,
                self.device,
                float(training.get("grad_clip", 0.0)),
            )
            should_validate = epoch % validate_every == 0 or epoch == epochs
            validation_metrics = None
            if should_validate:
                validation_metrics = validate(
                    self.model, self.loaders["val"], self.criterion, self.device
                )
                last_validation = dict(validation_metrics)
                # The acceptance recipe trains on MSE and selects the best
                # checkpoint by validation MSE; ``relative_l2`` is reported
                # alongside it and is the acceptance metric itself.
                selection_metric = loss_name if loss_name in validation_metrics else "relative_l2"
                score = float(validation_metrics[selection_metric])
                if score < best_score:
                    best_score = score
                    best_epoch = epoch
                    best_validation = dict(validation_metrics)
                    best_state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in self.model.state_dict().items()
                    }
                    if self.config["checkpoint"].get("save_best", True):
                        self._save("best_model.pt", epoch, validation_metrics, best_score)

            self.scheduler.step()

            record: Dict[str, Any] = {
                "epoch": epoch,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "train": train_metrics,
            }
            if validation_metrics is not None:
                record["validation"] = validation_metrics
            history.append(record)
            self.logger.log_metrics(record)

            print(
                _format_block(
                    epoch,
                    epochs,
                    train_metrics,
                    validation_metrics,
                    best_epoch,
                    best_score,
                    self.optimizer.param_groups[0]["lr"],
                ),
                flush=True,
            )

        if self.config["checkpoint"].get("save_last", True):
            self._save("last_model.pt", epochs, last_validation, best_score)

        # Leave the model holding the best weights so that any subsequent
        # in-process use matches best_model.pt exactly.
        if best_state is not None:
            self.model.load_state_dict(best_state)

        total_seconds = time.perf_counter() - started
        summary: Dict[str, Any] = {
            "training_loss": loss_name,
            "epochs": epochs,
            "validate_every": validate_every,
            "optimizer": self.config["optimizer"]["name"],
            "learning_rate": self.config["optimizer"]["lr"],
            "momentum": self.config["optimizer"].get("momentum", 0.0),
            "weight_decay": self.config["optimizer"].get("weight_decay", 0.0),
            "scheduler": self.config["scheduler"]["name"],
            "device": str(self.device),
            "best_epoch": best_epoch,
            "best_validation_mse": best_validation.get("mse"),
            "best_validation_relative_l2": best_validation.get("relative_l2"),
            "total_seconds": total_seconds,
        }
        if self.run_dir is not None:
            (self.run_dir / "history.json").write_text(
                json.dumps({"history": history, "summary": summary}, indent=2) + "\n",
                encoding="utf-8",
            )
        self.logger.log({"summary": summary})

        print(
            "\n".join(
                [
                    "",
                    "Training complete",
                    f"  Epochs            : {epochs}",
                    f"  Best epoch        : {best_epoch}",
                    f"  Best val MSE      : {best_score:.7e}",
                    f"  Best val relative L2: "
                    f"{best_validation.get('relative_l2', float('nan')):.7e}",
                    f"  Wall clock        : {total_seconds:.1f} s",
                ]
            ),
            flush=True,
        )
        return summary
