"""`transform_dtype` guards the differencing stage against float32 quantisation.

The HDF5 splits store float64.  ``_read_time_fields`` used to cast to the
sample dtype *before* ``transform`` ran, so ``diff_features``' second
difference -- which divides by ``dt**2 = 1e-6`` -- amplified the float32
rounding of the *input* by a factor of ~1e6.  Reading at ``transform_dtype``
and casting back afterwards removes that amplification; the default
(``transform_dtype=None``) must remain bit-identical to the old behaviour.

No real HDF5 is involved: ``h5py.File`` is stubbed with float64 arrays.

Requires torch (project venv).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np
    import torch

    from src.data.dataloader import build_dataloaders
    from src.data.dataset import FIELD_PATHS, OLEDNeuralOperatorDataset
    from src.data.preprocessing import build_preprocessor
except ImportError:  # pragma: no cover - stdlib-only test runners
    torch = None
    OLEDNeuralOperatorDataset = None


N_POINTS = 501
DT = 1e-3
FORCE_PATH = FIELD_PATHS["force"]
DIST_PATH = FIELD_PATHS["disturbance"]


class _StubH5:
    """Minimal stand-in for the ``h5py`` module: paths map to float64 arrays."""

    def __init__(self, arrays):
        self._arrays = arrays

    def File(self, *_args, **_kwargs):  # noqa: N802 - mirrors the h5py API
        return _StubHandle(self._arrays)


class _StubHandle:
    def __init__(self, arrays):
        self._arrays = arrays

    def __getitem__(self, path):
        return self._arrays[path]

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _make_root(root: Path, force: "np.ndarray", dist: "np.ndarray") -> None:
    """Lay out a one-sample train split with a minimal manifest."""
    (root / "train").mkdir(parents=True)
    (root / "train" / "sample.h5").touch()
    manifest = {
        "complete": True,
        "fields": {
            "force": {"shape": [N_POINTS, force.shape[1]]},
            "disturbance": {"shape": [N_POINTS, dist.shape[1]]},
        },
        "splits": {"train": [{"path": "sample.h5", "status": "success"}]},
    }
    (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _arrays(force: "np.ndarray", dist: "np.ndarray"):
    return {
        FORCE_PATH: force,
        DIST_PATH: dist,
        FIELD_PATHS["time"]: np.linspace(0.0, (N_POINTS - 1) * DT, N_POINTS),
    }


def _build(root: Path, arrays, **kwargs):
    with mock.patch("src.data.dataset.h5py", _StubH5(arrays)):
        dataset = OLEDNeuralOperatorDataset(
            root,
            "train",
            input_fields=("force",),
            target_fields=("disturbance",),
            include_metadata=False,
            **kwargs,
        )
        return dataset, dataset[0]


def _diff_features(channels):
    return build_preprocessor(
        {"name": "diff_features", "dt": DT, "channels": list(channels)}
    )


@unittest.skipIf(torch is None, "torch not available")
class TransformDtypeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        rng = np.random.default_rng(20260810)
        self.force = rng.standard_normal((N_POINTS, 4))
        self.dist = rng.standard_normal((N_POINTS, 2))
        _make_root(self.root, self.force, self.dist)
        self.arrays = _arrays(self.force, self.dist)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_default_is_bit_identical_to_pre_fix_behaviour(self) -> None:
        """transform_dtype=None must reproduce read-float32-then-transform."""
        transform = _diff_features([0, 1, 2, 3])
        _, item = _build(self.root, self.arrays, transform=transform,
                         dtype=torch.float32)
        legacy = torch.as_tensor(self.force, dtype=torch.float32)
        expected = transform({"input": legacy})["input"]
        self.assertEqual(item["input"].dtype, torch.float32)
        self.assertTrue(torch.equal(item["input"], expected))

    def test_opt_in_reads_float64_and_returns_float32(self) -> None:
        seen = []

        def probe(sample):
            seen.append(sample["input"].dtype)
            return sample

        _, item = _build(
            self.root, self.arrays, transform=probe,
            dtype=torch.float32, transform_dtype=torch.float64,
        )
        self.assertEqual(seen, [torch.float64])
        for key in ("input", "target", "grid"):
            self.assertEqual(item[key].dtype, torch.float32, key)

    def test_rounding_is_amplified_on_default_path_only(self) -> None:
        """y = a t^2 has exact second difference 2a everywhere.

        The default path rounds y to float32 first, so d2 carries an error of
        order eps32 * |y| / dt**2; reading in float64 keeps it at float32
        storage precision.
        """
        a = 0.5
        t = np.arange(N_POINTS) * DT
        force = np.repeat((a * t**2)[:, None], 4, axis=1)
        dist = np.zeros((N_POINTS, 2))
        root = Path(self._tmp.name) / "quadratic"
        _make_root(root, force, dist)
        arrays = _arrays(force, dist)
        transform = _diff_features([0, 1, 2, 3])

        def d2_error(transform_dtype):
            _, item = _build(root, arrays, transform=transform,
                             dtype=torch.float32, transform_dtype=transform_dtype)
            # diff_features appends d1 then d2: d2 occupies columns 8..11.
            return float((item["input"][:, 8:] - 2.0 * a).abs().max())

        fixed = d2_error(torch.float64)
        broken = d2_error(None)
        self.assertLess(fixed, 1e-5, f"float64 path error {fixed:.3e}")
        self.assertGreater(broken, 1e-3, f"default path error {broken:.3e}")
        self.assertGreater(broken / max(fixed, 1e-30), 100.0)

    def test_build_dataloaders_forwards_transform_dtype(self) -> None:
        captured = {}

        def fake_create_datasets(root, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop after capture")

        config = {
            "root": self.root,
            "input_fields": ["force"],
            "target_fields": ["disturbance"],
            "time_stride": 1,
            "batch_size": 2,
            "num_workers": 0,
            "transform_dtype": "float64",
        }
        with mock.patch("src.data.dataloader.create_datasets", fake_create_datasets):
            with self.assertRaises(RuntimeError):
                build_dataloaders(config, seed=0)
        self.assertIs(captured.get("transform_dtype"), torch.float64)

        captured.clear()
        config.pop("transform_dtype")
        with mock.patch("src.data.dataloader.create_datasets", fake_create_datasets):
            with self.assertRaises(RuntimeError):
                build_dataloaders(config, seed=0)
        self.assertIsNone(captured.get("transform_dtype"))

    def test_build_dataloaders_forwards_output_dtype(self) -> None:
        """`data.dtype` must default to float32, never to None.

        ``_resolve_dtype(None)`` returns None, so an absent key would override
        ``create_datasets``'s own default and hand the dataset a None dtype.
        """
        captured = {}

        def fake_create_datasets(root, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop after capture")

        config = {
            "root": self.root,
            "input_fields": ["force"],
            "target_fields": ["disturbance"],
            "time_stride": 1,
            "batch_size": 2,
            "num_workers": 0,
        }
        with mock.patch("src.data.dataloader.create_datasets", fake_create_datasets):
            with self.assertRaises(RuntimeError):
                build_dataloaders(config, seed=0)
        self.assertIs(captured.get("dtype"), torch.float32)

        captured.clear()
        config["dtype"] = "float64"
        with mock.patch("src.data.dataloader.create_datasets", fake_create_datasets):
            with self.assertRaises(RuntimeError):
                build_dataloaders(config, seed=0)
        self.assertIs(captured.get("dtype"), torch.float64)


if __name__ == "__main__":
    unittest.main()
