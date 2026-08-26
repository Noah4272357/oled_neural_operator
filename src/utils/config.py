"""Configuration loading, validation, overrides, and experiment directories."""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional


Config = Dict[str, Any]


def _deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def load_config(path: Path, overrides: Optional[Mapping[str, Any]] = None) -> Config:
    """Load JSON-compatible YAML (or ordinary YAML when PyYAML is installed)."""
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError(
                "The configuration is not JSON-compatible YAML; install PyYAML to read it."
            ) from error
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError("Configuration root must be a mapping.")
    config = copy.deepcopy(payload)
    if overrides:
        _deep_merge(config, overrides)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "data", "model", "loss", "optimizer", "scheduler", "training",
        "checkpoint", "experiment",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Configuration is missing sections: {sorted(missing)}")

    positive = {
        "data.batch_size": config["data"]["batch_size"],
        "data.time_stride": config["data"]["time_stride"],
        "model.embed_dim": config["model"]["embed_dim"],
        "model.modes": config["model"]["modes"],
        "model.width": config["model"]["width"],
        "model.lift_dim": config["model"]["lift_dim"],
        "model.num_blocks": config["model"]["num_blocks"],
        "optimizer.lr": config["optimizer"]["lr"],
        "training.epochs": config["training"]["epochs"],
        "training.validate_every": config["training"]["validate_every"],
    }
    eval_batch_size = config["data"].get("eval_batch_size")
    if eval_batch_size is not None:
        positive["data.eval_batch_size"] = eval_batch_size
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if int(config["data"]["num_workers"]) < 0:
        raise ValueError("data.num_workers must be non-negative.")
    if int(config["data"].get("time_start", 0)) < 0:
        raise ValueError("data.time_start must be non-negative.")
    time_stop = config["data"].get("time_stop")
    if time_stop is not None and time_stop <= config["data"].get("time_start", 0):
        raise ValueError("data.time_stop must be greater than data.time_start.")
    for name in ("max_train_samples", "max_val_samples", "max_test_samples"):
        value = config["data"].get(name)
        if value is not None and value <= 0:
            raise ValueError(f"data.{name} must be positive when supplied.")
    if not config["data"].get("input_fields") or not config["data"].get("target_fields"):
        raise ValueError("data.input_fields and data.target_fields cannot be empty.")
    if config["device"] not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported device: {config['device']!r}")
    if config["loss"]["name"] not in {"mse", "relative_l2", "spectral_rl2", "band_rl2"}:
        raise ValueError(f"Unsupported loss: {config['loss']['name']!r}")
    if str(config["optimizer"]["name"]).lower() not in {"adam", "adamw", "sgd"}:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']['name']!r}")
    if config["scheduler"]["name"] not in {"cosine", "plateau", "multistep"}:
        raise ValueError(f"Unsupported scheduler: {config['scheduler']['name']!r}")
    eta_min = float(config["scheduler"]["eta_min"])
    learning_rate = float(config["optimizer"]["lr"])
    if eta_min < 0 or eta_min > learning_rate:
        raise ValueError("scheduler.eta_min must be between zero and optimizer.lr.")
    if float(config["optimizer"]["weight_decay"]) < 0:
        raise ValueError("optimizer.weight_decay must be non-negative.")
    if float(config["training"]["grad_clip"]) < 0:
        raise ValueError("training.grad_clip must be non-negative.")
    if config["scheduler"]["name"] == "plateau":
        if not 0 < float(config["scheduler"]["factor"]) < 1:
            raise ValueError("scheduler.factor must be strictly between zero and one.")
        for name in ("patience", "threshold", "cooldown"):
            if float(config["scheduler"][name]) < 0:
                raise ValueError(f"scheduler.{name} must be non-negative.")


def save_resolved_config(config: Mapping[str, Any], path: Path) -> None:
    """Save the exact resolved configuration as JSON-compatible YAML."""
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def create_experiment_dir(config: Mapping[str, Any]) -> Path:
    """Create a collision-safe run directory, honoring an explicit output path."""
    experiment = config["experiment"]
    explicit = experiment.get("output_dir")
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists() and any(path.iterdir()) and not experiment.get("overwrite", False):
            raise FileExistsError(
                f"Experiment directory is not empty: {path}. "
                "Choose another --output-dir or pass --overwrite."
            )
        path.mkdir(parents=True, exist_ok=True)
        return path

    root = Path(experiment.get("root_dir", "experiments")).expanduser()
    run_name = experiment.get("run_name")
    if not run_name:
        model = config["model"]
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = (
            f"{model['name']}_w{model['width']}_b{model['num_blocks']}_"
            f"seed{config['seed']}_{timestamp}"
        )
    candidate = root / str(run_name)
    suffix = 1
    while candidate.exists():
        candidate = root / f"{run_name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def legacy_args(config: Mapping[str, Any], output_dir: Optional[Path]) -> Dict[str, Any]:
    """Flatten resolved config for readers of the historical checkpoint schema."""
    data = config["data"]
    model = config["model"]
    optimizer = config["optimizer"]
    scheduler = config["scheduler"]
    training = config["training"]
    return {
        "data_root": str(data["root"]),
        "output_dir": str(output_dir) if output_dir else None,
        "epochs": training["epochs"],
        "validate_every": training["validate_every"],
        "batch_size": data["batch_size"],
        "eval_batch_size": data.get("eval_batch_size"),
        "learning_rate": optimizer["lr"],
        "weight_decay": optimizer["weight_decay"],
        "eta_min": scheduler["eta_min"],
        "scheduler": scheduler["name"],
        "plateau_factor": scheduler.get("factor"),
        "plateau_patience": scheduler.get("patience"),
        "plateau_threshold": scheduler.get("threshold"),
        "plateau_threshold_mode": scheduler.get("threshold_mode"),
        "plateau_cooldown": scheduler.get("cooldown"),
        "grad_clip": training["grad_clip"],
        "loss": config["loss"]["name"],
        "time_stride": data["time_stride"],
        "input_fields": list(data["input_fields"]),
        "target_fields": list(data["target_fields"]),
        "num_workers": data["num_workers"],
        "modes": model["modes"],
        "width": model["width"],
        "embed_dim": model["embed_dim"],
        "lift_dim": model["lift_dim"],
        "num_blocks": model["num_blocks"],
        "seed": config["seed"],
        "device": config["device"],
        "max_train_samples": data.get("max_train_samples"),
        "max_val_samples": data.get("max_val_samples"),
        "max_test_samples": data.get("max_test_samples"),
        "keep_resume_learning_rate": training.get("keep_resume_learning_rate", False),
    }
