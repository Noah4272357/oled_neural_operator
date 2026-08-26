"""Whole-split in-memory caching for the lazy HDF5 dataset.

The OLED dataset holds ~110 MB of float32 tensors per split (5000 samples x
501 time points x 11 channels), so materializing a split in RAM removes every
per-sample HDF5 open/read from the training loop. On Linux the DataLoader
workers are forked, so a cache warmed in the main process is shared by the
workers copy-on-write (zero copy, single materialization).

Enable via ``config.data.memory_cache``; disabling restores the original
lazy per-sample HDF5 read behavior exactly.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset


class MemoryCachedDataset(Dataset):
    """Index-addressed cache over another dataset.

    ``warm()`` materializes the whole split in the calling (main) process;
    forked DataLoader workers then share the tensors copy-on-write. Without
    ``warm()`` the cache fills lazily per worker on first access.
    """

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset
        self._cache: Optional[Dict[int, Dict[str, Any]]] = None

    def warm(self, limit: Optional[int] = None) -> "MemoryCachedDataset":
        """Materialize the split (or its first ``limit`` samples); returns self."""
        self._cache = {}
        for index in range(len(self._dataset) if limit is None else min(limit, len(self._dataset))):
            self._cache[index] = dict(self._dataset[index])
        return self

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        cache = self._cache
        if cache is None:
            cache = {}
            self._cache = cache
        if index not in cache:
            cache[index] = dict(self._dataset[index])
        # Return detached tensor copies so callers can never mutate the
        # shared cache entries (microsecond cost vs. millisecond HDF5 reads).
        return {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in cache[index].items()
        }

    def __getattr__(self, name: str) -> Any:
        # Transparently forward dataset-level attributes (e.g. input_channels,
        # target_channels used by the model factory after unwrap_dataset).
        return getattr(self._dataset, name)


__all__ = ["MemoryCachedDataset"]
