"""PyTorch input pipeline for the OLED microstage neural-operator dataset.

The dataset uses one HDF5 file per simulation and provides ``train``, ``val``,
and ``test`` splits. Each item is a dictionary with channel-last time series::

    {
        "input": Tensor[T, C_in],
        "target": Tensor[T, C_out],
        "grid": Tensor[T],
        "metadata": {...},
    }

Files are opened inside ``__getitem__`` so HDF5 handles are never shared by
DataLoader worker processes and memory use is independent of dataset size.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


PathLike = Union[str, Path]

# Names used by the manifest mapped to their HDF5 locations.
FIELD_PATHS: Mapping[str, str] = {
    "time": "/time",
    "force": "/actuators/force",
    "disturbance": "/disturbance/force",
    "displacement": "/states/q",
    "velocity": "/states/q_dot",
    "acceleration": "/states/q_ddot",
    "encoder_displacement": "/sensors/encoder_displacement",
    "measurement_matrix_H": "/sensors/measurement_matrix_H",
}

DEFAULT_INPUT_FIELDS: Tuple[str, ...] = ("force", "disturbance")
DEFAULT_TARGET_FIELDS: Tuple[str, ...] = (
    "displacement",
    "velocity",
    "acceleration",
)


def _cast_tensors(value: Any, dtype: torch.dtype) -> Any:
    """Cast every tensor reachable from ``value`` to ``dtype``.

    Walks dictionaries, lists, and tuples, which is what a sample dictionary
    and its metadata contain.  Dicts are updated in place; lists, tuples, and
    tensors are replaced, so callers must use the returned value.
    """
    if isinstance(value, torch.Tensor):
        return value.to(dtype)
    if isinstance(value, dict):
        for key, item in value.items():
            value[key] = _cast_tensors(item, dtype)
    elif isinstance(value, (list, tuple)):
        return type(value)(_cast_tensors(item, dtype) for item in value)
    return value


class OLEDNeuralOperatorDataset(Dataset):
    """Lazy, worker-safe reader for one OLED dataset split.

    Args:
        root: Directory containing ``dataset_manifest.json`` and split folders.
        split: One of ``"train"``, ``"val"``, or ``"test"``.
        input_fields: Time-dependent fields concatenated on the last axis.
        target_fields: Time-dependent fields concatenated on the last axis.
        time_start: First retained time index.
        time_stop: Exclusive final time index; ``None`` retains the whole trace.
        time_stride: Retain every Nth time sample.
        dtype: Floating-point dtype returned to PyTorch.
        include_metadata: Include manifest metadata in each returned item.
        transform: Optional callable applied to the completed sample dictionary.
        transform_dtype: Optional dtype used while the ``transform`` runs, with
            every returned tensor cast back to ``dtype`` afterwards.  Set this
            to ``torch.float64`` for transforms that amplify rounding, such as a
            second difference divided by ``dt**2``: the HDF5 values are float64,
            so downgrading before differentiating injects a quantisation floor
            (``eps32 * |x| / dt**2``) that no later upcast can undo.  ``None``
            (default) leaves the historical behaviour untouched.

    .. note::
        Reads happen at ``transform_dtype`` only when a ``transform`` is
        present; otherwise ``dtype`` is used throughout.
    """

    VALID_SPLITS = ("train", "val", "test")

    def __init__(
        self,
        root: PathLike,
        split: str,
        *,
        input_fields: Sequence[str] = DEFAULT_INPUT_FIELDS,
        target_fields: Sequence[str] = DEFAULT_TARGET_FIELDS,
        time_start: int = 0,
        time_stop: Optional[int] = None,
        time_stride: int = 1,
        dtype: torch.dtype = torch.float32,
        include_metadata: bool = True,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        transform_dtype: Optional[torch.dtype] = None,
    ) -> None:
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Unknown split {split!r}; expected one of {self.VALID_SPLITS}."
            )
        if time_start < 0:
            raise ValueError("time_start must be non-negative.")
        if time_stop is not None and time_stop <= time_start:
            raise ValueError("time_stop must be greater than time_start.")
        if time_stride <= 0:
            raise ValueError("time_stride must be positive.")

        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.split_dir = self.root / split
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Split directory does not exist: {self.split_dir}")

        self.input_fields = self._validate_fields(input_fields, "input_fields")
        self.target_fields = self._validate_fields(target_fields, "target_fields")
        self.time_slice = slice(time_start, time_stop, time_stride)
        self.dtype = dtype
        self.include_metadata = include_metadata
        self.transform = transform
        self.transform_dtype = transform_dtype

        self.manifest = self._read_manifest()
        self.records = self._records_for_split()
        if not self.records:
            raise RuntimeError(f"No successful HDF5 samples found for split {split!r}.")

        self.input_channel_slices = self._channel_slices(self.input_fields)
        self.target_channel_slices = self._channel_slices(self.target_fields)

    @staticmethod
    def _validate_fields(fields: Sequence[str], argument: str) -> Tuple[str, ...]:
        fields = tuple(fields)
        if not fields:
            raise ValueError(f"{argument} cannot be empty.")
        unknown = set(fields).difference(FIELD_PATHS)
        if unknown:
            raise ValueError(f"Unknown fields in {argument}: {sorted(unknown)}")
        if "time" in fields or "measurement_matrix_H" in fields:
            raise ValueError(
                f"{argument} may contain only time-dependent, channel-valued "
                "fields; time is returned as 'grid' and measurement_matrix_H "
                "is not a time series."
            )
        return fields

    def _read_manifest(self) -> Dict[str, Any]:
        manifest_path = self.root / "dataset_manifest.json"
        if not manifest_path.is_file():
            return {}
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("complete") is False:
            raise RuntimeError(
                f"Dataset manifest reports incomplete generation: {manifest_path}"
            )
        return manifest

    def _records_for_split(self) -> Tuple[Dict[str, Any], ...]:
        manifest_splits = self.manifest.get("splits", {})
        if self.split in manifest_splits:
            records = []
            for record in manifest_splits[self.split]:
                if record.get("status", "success") != "success":
                    continue
                path = self.split_dir / record["path"]
                if not path.is_file():
                    raise FileNotFoundError(f"Manifest sample does not exist: {path}")
                records.append({**record, "_path": path})
            return tuple(records)

        # Manifest-free fallback for copied subsets or ad-hoc local testing.
        return tuple(
            {"path": path.name, "status": "success", "_path": path}
            for path in sorted(self.split_dir.glob("*.h5"))
        )

    def _field_width(self, field: str) -> int:
        fields = self.manifest.get("fields", {})
        shape = fields.get(field, {}).get("shape")
        if shape and len(shape) >= 2:
            return int(shape[-1])

        # The fallback reads one small slice, not an entire trace.
        with h5py.File(self.records[0]["_path"], "r") as handle:
            field_dataset = handle[FIELD_PATHS[field]]
            return 1 if field_dataset.ndim == 1 else int(field_dataset.shape[-1])

    def _channel_slices(self, fields: Sequence[str]) -> Mapping[str, slice]:
        result: Dict[str, slice] = {}
        start = 0
        for field in fields:
            stop = start + self._field_width(field)
            result[field] = slice(start, stop)
            start = stop
        return result

    @property
    def input_channels(self) -> int:
        """Number of concatenated input channels (post-transform).

        Samples one transformed item so preprocessing that appends derived
        channels (e.g. ``diff_features``) is reflected in the width seen by
        the model factory; for identity preprocessing this equals the sum of
        the declared field widths.
        """
        return int(self[0]["input"].shape[-1])

    @property
    def target_channels(self) -> int:
        """Number of concatenated target channels."""
        return sum(
            item.stop - item.start for item in self.target_channel_slices.values()
        )

    def __len__(self) -> int:
        return len(self.records)

    @property
    def _pre_transform_dtype(self) -> torch.dtype:
        """Dtype used while a sample is built and transformed.

        Equals ``transform_dtype`` when a transform is configured, otherwise
        ``dtype`` — so the default configuration is unchanged.
        """
        if self.transform is not None and self.transform_dtype is not None:
            return self.transform_dtype
        return self.dtype

    def _read_time_fields(self, handle: h5py.File, fields: Sequence[str]) -> Tensor:
        arrays = []
        expected_length: Optional[int] = None
        for field in fields:
            array = np.asarray(handle[FIELD_PATHS[field]][self.time_slice])
            if array.ndim == 1:
                array = array[:, None]
            if array.ndim != 2:
                raise ValueError(
                    f"Field {field!r} must have shape [time, channels], got {array.shape}."
                )
            if expected_length is None:
                expected_length = len(array)
            elif len(array) != expected_length:
                raise ValueError("Selected fields do not share the same time dimension.")
            arrays.append(array)

        # HDF5 values are float64. Reading at ``_pre_transform_dtype`` keeps the
        # transform's arithmetic exact; the memory-saving cast back to ``dtype``
        # happens in ``__getitem__`` once the transform has run.
        merged = np.ascontiguousarray(np.concatenate(arrays, axis=-1))
        return torch.as_tensor(merged, dtype=self._pre_transform_dtype)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        path = record["_path"]
        with h5py.File(path, "r") as handle:
            inputs = self._read_time_fields(handle, self.input_fields)
            targets = self._read_time_fields(handle, self.target_fields)
            grid_array = np.ascontiguousarray(
                handle[FIELD_PATHS["time"]][self.time_slice]
            )
            grid = torch.as_tensor(grid_array, dtype=self._pre_transform_dtype)

        if inputs.shape[0] != grid.shape[0] or targets.shape[0] != grid.shape[0]:
            raise ValueError(f"Inconsistent time dimensions in {path}")

        sample: Dict[str, Any] = {
            "input": inputs,
            "target": targets,
            "grid": grid,
        }
        if self.include_metadata:
            sample["metadata"] = {
                key: value for key, value in record.items() if key != "_path"
            }
            sample["metadata"]["split"] = self.split

        if self.transform is not None:
            sample = self.transform(sample)

        # Grid included: ``FNO1d.forward`` concatenates it onto the inputs, and
        # mixing dtypes there would silently promote the whole graph.
        if self._pre_transform_dtype is not self.dtype:
            sample = _cast_tensors(sample, self.dtype)
        return sample


def create_datasets(
    root: PathLike,
    *,
    input_fields: Sequence[str] = DEFAULT_INPUT_FIELDS,
    target_fields: Sequence[str] = DEFAULT_TARGET_FIELDS,
    time_start: int = 0,
    time_stop: Optional[int] = None,
    time_stride: int = 1,
    dtype: torch.dtype = torch.float32,
    include_metadata: bool = True,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    transform_dtype: Optional[torch.dtype] = None,
) -> Dict[str, OLEDNeuralOperatorDataset]:
    """Create the available train/validation/test Dataset objects."""
    common = dict(
        input_fields=input_fields,
        target_fields=target_fields,
        time_start=time_start,
        time_stop=time_stop,
        time_stride=time_stride,
        dtype=dtype,
        include_metadata=include_metadata,
        transform=transform,
        transform_dtype=transform_dtype,
    )
    return {
        split: OLEDNeuralOperatorDataset(root, split, **common)
        for split in OLEDNeuralOperatorDataset.VALID_SPLITS
        if (Path(root).expanduser() / split).is_dir()
    }


def create_dataloaders(
    root: PathLike,
    *,
    batch_size: int = 8,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    seed: int = 20260810,
    drop_last: bool = False,
    input_fields: Sequence[str] = DEFAULT_INPUT_FIELDS,
    target_fields: Sequence[str] = DEFAULT_TARGET_FIELDS,
    time_start: int = 0,
    time_stop: Optional[int] = None,
    time_stride: int = 1,
    dtype: torch.dtype = torch.float32,
    include_metadata: bool = True,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    transform_dtype: Optional[torch.dtype] = None,
) -> Dict[str, DataLoader]:
    """Build deterministic train/validation/test DataLoaders.

    Training is shuffled. Validation and test retain manifest order.
    ``eval_batch_size`` defaults to ``batch_size``.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if eval_batch_size is not None and eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    datasets = create_datasets(
        root,
        input_fields=input_fields,
        target_fields=target_fields,
        time_start=time_start,
        time_stop=time_stop,
        time_stride=time_stride,
        dtype=dtype,
        include_metadata=include_metadata,
        transform=transform,
        transform_dtype=transform_dtype,
    )
    eval_batch_size = eval_batch_size or batch_size
    pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
    persistent_workers = (
        num_workers > 0 if persistent_workers is None else persistent_workers
    )
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers=True requires num_workers > 0.")

    loaders: Dict[str, DataLoader] = {}
    for split, dataset in datasets.items():
        generator = torch.Generator()
        generator.manual_seed(seed)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size if split == "train" else eval_batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=drop_last if split == "train" else False,
            generator=generator,
        )
    return loaders


def create_train_test_dataloaders(
    root: PathLike, **kwargs: Any
) -> Tuple[DataLoader, DataLoader]:
    """Convenience wrapper returning exactly ``(train_loader, test_loader)``."""
    loaders = create_dataloaders(root, **kwargs)
    return loaders["train"], loaders["test"]


__all__ = [
    "FIELD_PATHS",
    "DEFAULT_INPUT_FIELDS",
    "DEFAULT_TARGET_FIELDS",
    "OLEDNeuralOperatorDataset",
    "create_datasets",
    "create_dataloaders",
    "create_train_test_dataloaders",
]
