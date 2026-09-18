"""Configuration loading, validation, and test run-directory resolution."""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

Config = Dict[str, Any]

RUN_ROOT = "runs"
RUN_PREFIX = "test_"

_REQUIRED_SECTIONS = (
    "data",
    "model",
    "loss",
    "optimizer",
    "scheduler",
    "training",
    "checkpoint",
    "experiment",
)


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
    """Validate structure and numeric ranges.

    Component *names* (model / loss / optimizer / scheduler) are validated by
    the factories that own them, so the accepted set has exactly one source of
    truth; this function checks that the numbers are sane before any data is
    read.
    """
    missing = set(_REQUIRED_SECTIONS).difference(config)
    if missing:
        raise ValueError(f"Configuration is missing sections: {sorted(missing)}")

    data = config["data"]
    model = config["model"]
    optimizer = config["optimizer"]
    scheduler = config["scheduler"]
    training = config["training"]

    positive = {
        "data.batch_size": data["batch_size"],
        "data.time_stride": data["time_stride"],
        "model.init_scale": model.get("init_scale", 1e-3),
        "optimizer.lr": optimizer["lr"],
        "training.epochs": training["epochs"],
        "training.validate_every": training["validate_every"],
    }
    if data.get("eval_batch_size") is not None:
        positive["data.eval_batch_size"] = data["eval_batch_size"]
    for name, value in positive.items():
        if float(value) <= 0.0:
            raise ValueError(f"{name} must be positive.")

    if int(data["num_workers"]) < 0:
        raise ValueError("data.num_workers must be non-negative.")
    if int(data.get("time_start", 0)) < 0:
        raise ValueError("data.time_start must be non-negative.")
    time_stop = data.get("time_stop")
    if time_stop is not None and time_stop <= data.get("time_start", 0):
        raise ValueError("data.time_stop must be greater than data.time_start.")
    for name in ("max_train_samples", "max_val_samples", "max_test_samples"):
        value = data.get(name)
        if value is not None and value <= 0:
            raise ValueError(f"data.{name} must be positive when supplied.")
    if not data.get("input_fields") or not data.get("target_fields"):
        raise ValueError("data.input_fields and data.target_fields cannot be empty.")
    for name in ("dtype", "transform_dtype"):
        value = data.get(name)
        if value is not None and str(value).lower() not in {"float32", "float64"}:
            raise ValueError(f"data.{name} must be float32 or float64, got {value!r}.")
    preprocessing = data.get("preprocessing", {"name": "none"})
    if str(preprocessing.get("name", "none")).lower() == "diff_features" and data.get(
        "transform_dtype"
    ) is None:
        raise ValueError(
            "data.transform_dtype must be set when preprocessing appends "
            "difference features: float32 quantisation injects a floor that no "
            "later upcast can undo."
        )

    if config["device"] not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported device: {config['device']!r}")

    eta_min = float(scheduler.get("eta_min", 0.0))
    learning_rate = float(optimizer["lr"])
    if not 0.0 <= eta_min <= learning_rate:
        raise ValueError("scheduler.eta_min must be between zero and optimizer.lr.")
    if float(optimizer.get("weight_decay", 0.0)) < 0.0:
        raise ValueError("optimizer.weight_decay must be non-negative.")
    if float(optimizer.get("momentum", 0.0)) < 0.0:
        raise ValueError("optimizer.momentum must be non-negative.")
    if float(training.get("grad_clip", 0.0)) < 0.0:
        raise ValueError("training.grad_clip must be non-negative.")


def save_resolved_config(config: Mapping[str, Any], path: Path) -> None:
    """Save the exact resolved configuration as JSON-compatible YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_run_dir(
    config: Mapping[str, Any], run_dir: Optional[Path] = None
) -> Path:
    """Return the test run directory, creating it when necessary.

    An explicit ``run_dir`` (or ``experiment.run_dir`` in the config) is used
    verbatim, so the runbook can create the directory once and have every step
    fill the same place.  Otherwise a fresh, collision-safe
    ``runs/test_<timestamp>`` is created.
    """
    explicit = run_dir if run_dir is not None else config["experiment"].get("run_dir")
    if explicit:
        path = Path(explicit).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    root = Path(config["experiment"].get("root", RUN_ROOT)).expanduser()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = root / f"{RUN_PREFIX}{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"{RUN_PREFIX}{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate
