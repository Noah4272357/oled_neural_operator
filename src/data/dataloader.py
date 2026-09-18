"""Dataset selection and deterministic DataLoader assembly."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .cache import MemoryCachedDataset
from .dataset import create_datasets

from .preprocessing import build_preprocessor
from src.utils.paths import resolve_data_root
from src.utils.seed import seed_worker


_SPECIALISED_DTYPES = {"float32": torch.float32, "float64": torch.float64}


def _resolve_dtype(value: Any, key: str) -> Optional[torch.dtype]:
    """Resolve a config dtype (name or ``torch.dtype``) to a ``torch.dtype``.

    Configs are JSON-compatible, so dtypes arrive as strings such as
    ``"float64"``; tests and analysis scripts may pass the object directly.
    """
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    try:
        return _SPECIALISED_DTYPES[str(value).lower()]
    except KeyError:
        raise ValueError(
            f"Unsupported {key}={value!r}; expected one of "
            f"{sorted(_SPECIALISED_DTYPES)}."
        ) from None


def _limited(dataset: Dataset, maximum: Optional[int]) -> Dataset:
    if maximum is None:
        return dataset
    if maximum <= 0:
        raise ValueError("Sample limits must be positive when supplied.")
    return Subset(dataset, range(min(maximum, len(dataset))))


def unwrap_dataset(dataset: Dataset) -> Dataset:
    """Return the domain dataset beneath a possible sample-limit subset."""
    return dataset.dataset if isinstance(dataset, Subset) else dataset


def build_dataloaders(config: Mapping[str, Any], seed: int) -> Dict[str, DataLoader]:
    """Build train/validation/test loaders without leaking assembly into CLIs."""
    transform = build_preprocessor(config.get("preprocessing", {"name": "none"}))
    datasets = create_datasets(
        resolve_data_root(config.get("root")),
        input_fields=config["input_fields"],
        target_fields=config["target_fields"],
        time_start=config.get("time_start", 0),
        time_stop=config.get("time_stop"),
        time_stride=config["time_stride"],
        include_metadata=False,
        transform=transform,
        # Absent key must fall back to the dataset default rather than None.
        dtype=_resolve_dtype(config.get("dtype"), "data.dtype") or torch.float32,
        transform_dtype=_resolve_dtype(
            config.get("transform_dtype"), "data.transform_dtype"
        ),
    )
    required = {"train", "val", "test"}
    missing = required.difference(datasets)
    if missing:
        raise RuntimeError(f"Dataset is missing required splits: {sorted(missing)}")

    limits = {
        "train": config.get("max_train_samples"),
        "val": config.get("max_val_samples"),
        "test": config.get("max_test_samples"),
    }
    if config.get("memory_cache", False):
        # Warm in the main process (only the sampled prefix when limits are
        # set); forked workers share the tensors via copy-on-write.
        datasets = {
            split: MemoryCachedDataset(dataset).warm(limit=limits[split])
            for split, dataset in datasets.items()
        }
    batch_size = int(config["batch_size"])
    eval_batch_size = int(config.get("eval_batch_size") or batch_size)
    num_workers = int(config["num_workers"])
    pin_memory = torch.cuda.is_available()
    loaders: Dict[str, DataLoader] = {}
    for split, dataset in datasets.items():
        selected = _limited(dataset, limits[split])
        generator = torch.Generator().manual_seed(seed)
        loaders[split] = DataLoader(
            selected,
            batch_size=batch_size if split == "train" else eval_batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            generator=generator,
            worker_init_fn=seed_worker if num_workers > 0 else None,
        )
    return loaders


__all__ = ["build_dataloaders", "unwrap_dataset"]
