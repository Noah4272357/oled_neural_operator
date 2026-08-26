"""Central training, validation, scheduling, and checkpoint orchestration."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.utils.checkpoint import save_checkpoint
from src.utils.logging import ExperimentLogger

from .factories import LossFunction
from .train_epoch import train_one_epoch
from .validate import validate


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
        logger: ExperimentLogger,
        run_dir: Optional[Path],
        legacy_arguments: Mapping[str, Any],
        start_epoch: int = 0,
        initial_learning_rate: Optional[float] = None,
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
        self.legacy_arguments = legacy_arguments
        self.start_epoch = start_epoch
        self.initial_learning_rate = (
            float(initial_learning_rate)
            if initial_learning_rate is not None
            else float(optimizer.param_groups[0]["lr"])
        )
        if self.config["training"].get("compile", False):
            # Dynamo raises at first forward when Triton cannot build kernels
            # (e.g. missing Python.h). Fall back to eager instead of failing.
            try:
                import torch._dynamo

                torch._dynamo.config.suppress_errors = True
            except ImportError:  # pragma: no cover - older torch
                pass
            try:
                self.model = torch.compile(model)
            except Exception as error:  # pragma: no cover - environment specific
                print(f"[trainer] torch.compile failed; continuing un-compiled: {error}")

    def _save(
        self, filename: str, epoch: int, validation_metrics: Mapping[str, float], best: float
    ) -> None:
        if self.run_dir is None:
            return
        save_checkpoint(
            self.run_dir / filename,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            validation=validation_metrics,
            best_metric=best,
            config=self.config,
            legacy_arguments=self.legacy_arguments,
        )

    def fit(self) -> Dict[str, Any]:
        training = self.config["training"]
        scheduler_name = self.config["scheduler"]["name"]
        loss_name = self.config["loss"]["name"]
        epochs = int(training["epochs"])
        validate_every = int(training["validate_every"])
        best_score = float("inf")
        best_validation: Dict[str, float] = {}
        last_validation: Dict[str, float] = {}
        best_state = None
        history = []
        training_start = time.perf_counter()

        for run_index in range(1, epochs + 1):
            epoch = self.start_epoch + run_index
            train_metrics = train_one_epoch(
                self.model,
                self.loaders["train"],
                self.optimizer,
                self.criterion,
                self.device,
                float(training["grad_clip"]),
                training.get("amp"),
            )
            should_validate = run_index % validate_every == 0 or run_index == epochs
            validation_metrics = None
            if should_validate:
                validation_metrics = validate(
                    self.model, self.loaders["val"], self.criterion, self.device
                )
                last_validation = dict(validation_metrics)
                # Custom losses (e.g. spectral_rl2) have no metrics key of
                # their own; select the best checkpoint by relative_l2 then.
                selection_metric = (
                    loss_name if loss_name in validation_metrics else "relative_l2"
                )
                score = validation_metrics[selection_metric]
                if score < best_score:
                    best_score = score
                    best_validation = dict(validation_metrics)
                    best_state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in self.model.state_dict().items()
                    }
                    if self.config["checkpoint"].get("save_best", True):
                        self._save("best_model.pt", epoch, validation_metrics, best_score)

            if scheduler_name in ("cosine", "multistep"):
                # Multistep (M6 pure-spectral recipe, Ruling M6-P.2) decays
                # by epoch like cosine; plateau keeps its metric-driven step.
                self.scheduler.step()
            elif validation_metrics is not None:
                self.scheduler.step(validation_metrics[loss_name])

            record: Dict[str, Any] = {
                "epoch": epoch,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "train": train_metrics,
            }
            if validation_metrics is not None:
                record["validation"] = validation_metrics
            history.append(record)
            self.logger.log_metrics(record)

        end_epoch = self.start_epoch + epochs
        if self.config["checkpoint"].get("save_last", True):
            self._save("last_model.pt", end_epoch, last_validation, best_score)

        if best_state is not None:
            self.model.load_state_dict(best_state)
        test_metrics = validate(
            self.model, self.loaders["test"], self.criterion, self.device
        )
        summary: Dict[str, Any] = {
            "training_loss": loss_name,
            "start_epoch": self.start_epoch,
            "end_epoch": end_epoch,
            "additional_epochs": epochs,
            "resume_checkpoint": self.config.get("runtime", {}).get("resume_checkpoint"),
            "scheduler_restarted_on_resume": self.start_epoch > 0,
            "scheduler": scheduler_name,
            "optimizer": self.config["optimizer"]["name"],
            "weight_decay": self.config["optimizer"]["weight_decay"],
            "initial_learning_rate": self.initial_learning_rate,
            "minimum_learning_rate": self.config["scheduler"]["eta_min"],
            "plateau_factor": self.config["scheduler"].get("factor"),
            "plateau_patience": self.config["scheduler"].get("patience"),
            "plateau_threshold": self.config["scheduler"].get("threshold"),
            "plateau_threshold_mode": self.config["scheduler"].get("threshold_mode"),
            "plateau_cooldown": self.config["scheduler"].get("cooldown"),
            "kept_resume_learning_rate": training.get(
                "keep_resume_learning_rate", False
            ),
            "validation_selection_metric": selection_metric,
            "best_validation_score": best_score,
            "best_validation_mse": best_validation.get("mse"),
            "best_validation_relative_l2": best_validation.get("relative_l2"),
            "test": test_metrics,
            "total_seconds": time.perf_counter() - training_start,
        }
        if self.device.type == "cuda":
            summary["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(self.device)
        if self.run_dir is not None:
            (self.run_dir / "history.json").write_text(
                json.dumps({"history": history, "summary": summary}, indent=2) + "\n",
                encoding="utf-8",
            )
        self.logger.log({"summary": summary})
        return summary
