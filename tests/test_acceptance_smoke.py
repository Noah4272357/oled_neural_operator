"""End-to-end acceptance smoke test on a synthetic miniature dataset.

Exercises the whole formal flow at toy scale -- dataset -> train-only basis ->
model -> backward -> SGD step -> validation -> test evaluation -> run
artifacts -> figures -- without the real 5000x100 training run.

The dataset is synthesized here rather than read from disk so the test is
self-contained: it still has to satisfy the HDF5/manifest contract the real
loader enforces, so a contract change breaks this test.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_CONFIG = PROJECT_ROOT / "configs" / "acceptance.yaml"
SCRIPTS = PROJECT_ROOT / "scripts"

# Long enough that the default --f-in 100 stays inside the available spectrum
# (a 256-point trace has 129 bins), so the smoke test exercises the same code
# path as the real 501-point dataset rather than a special-cased short trace.
N_POINTS = 256
SPLITS = {"train": 6, "val": 3, "test": 3}

MANIFEST_TEMPLATE: Dict[str, Any] = {
    "schema": "neural_operator_dataset_manifest/3.0",
    "hdf5_schema_version": "5.0",
    "complete": True,
    "failure_count": 0,
}


def load_script(name: str) -> ModuleType:
    """Import a scripts/*.py module by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_sample(path: Path, rng: np.random.Generator) -> None:
    time = np.arange(N_POINTS, dtype=np.float64) / 1000.0
    force = rng.normal(scale=1.0, size=(N_POINTS, 4))
    encoder = rng.normal(scale=1e-3, size=(N_POINTS, 4))
    disturbance = rng.normal(scale=0.5, size=(N_POINTS, 2))
    with h5py.File(path, "w") as handle:
        handle.create_dataset("time", data=time)
        handle.create_group("actuators").create_dataset("force", data=force)
        handle.create_group("sensors").create_dataset(
            "encoder_displacement", data=encoder
        )
        handle.create_group("disturbance").create_dataset("force", data=disturbance)


def build_dataset(root: Path) -> Path:
    rng = np.random.default_rng(20260810)
    root.mkdir(parents=True, exist_ok=True)
    splits: Dict[str, list] = {}
    for split, count in SPLITS.items():
        directory = root / split
        directory.mkdir(exist_ok=True)
        records = []
        for index in range(count):
            name = f"sample_{index:06d}.h5"
            write_sample(directory / name, rng)
            records.append({"path": name, "status": "success"})
        splits[split] = records

    manifest = dict(MANIFEST_TEMPLATE)
    manifest.update(
        {
            "fields": {
                "time": {"path": "/time", "shape": [N_POINTS]},
                "force": {"path": "/actuators/force", "shape": [N_POINTS, 4]},
                "encoder_displacement": {
                    "path": "/sensors/encoder_displacement",
                    "shape": [N_POINTS, 4],
                },
                "disturbance": {"path": "/disturbance/force", "shape": [N_POINTS, 2]},
            },
            "time_grid": {"count": N_POINTS, "dt_s": 0.001, "start_s": 0.0,
                          "end_s": 0.064, "preroll_s": 0.5},
            "simulation": {"preroll_s": 0.5, "integration_substeps": 5},
            "generation": {"duration": 0.064},
            "splits": splits,
            "recommended_problem": {
                "name": "inverse_disturbance",
                "input": ["/actuators/force", "/sensors/encoder_displacement"],
                "input_channels": 8,
                "target": "/disturbance/force",
                "target_channels": 2,
            },
        }
    )
    (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


class AcceptanceSmokeTest(unittest.TestCase):
    """dataset -> basis -> train -> evaluate -> report, at toy scale."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        base = Path(cls._tmp.name)
        cls.dataset_root = build_dataset(base / "dataset")
        cls.run_dir = base / "runs" / "acceptance_smoke"

        config = json.loads(ACCEPTANCE_CONFIG.read_text(encoding="utf-8"))
        config["data"]["root"] = str(cls.dataset_root)
        config["data"]["max_train_samples"] = SPLITS["train"]
        config["data"]["max_val_samples"] = SPLITS["val"]
        config["data"]["max_test_samples"] = SPLITS["test"]
        config["data"]["batch_size"] = SPLITS["train"]
        config["data"]["eval_batch_size"] = SPLITS["test"]
        config["training"]["epochs"] = 3
        config["training"]["validate_every"] = 1
        # The canonical lr (2.684e5) is calibrated for the full 5000-sample
        # batch: MSE averages over the batch, so the loss Hessian scales as
        # 1/numel and a tiny subset needs a correspondingly smaller step.  This
        # is the same coupling scripts/train.py warns about in smoke mode.
        config["optimizer"]["lr"] = 100.0
        config["scheduler"]["eta_min"] = 0.0
        config["experiment"]["run_dir"] = str(cls.run_dir)
        cls.config_path = base / "acceptance_tiny.yaml"
        cls.config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        cls.basis_result = load_script("fit_basis").main(
            ["--config", str(cls.config_path), "--run-dir", str(cls.run_dir), "--d", "4"]
        )
        cls.train_summary = load_script("train").main(
            ["--config", str(cls.config_path), "--run-dir", str(cls.run_dir), "--device", "cpu"]
        )
        cls.eval_result = load_script("evaluate").main(
            ["--run-dir", str(cls.run_dir), "--device", "cpu"]
        )
        cls.figures = load_script("report").main(["--run-dir", str(cls.run_dir)])

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_basis_used_the_train_split_only(self) -> None:
        self.assertEqual(self.basis_result["in_channels"], 16)
        self.assertEqual(self.basis_result["out_channels"], 2)
        self.assertEqual(self.basis_result["n_points"], N_POINTS)

    def test_training_ran_for_the_configured_epochs(self) -> None:
        self.assertEqual(self.train_summary["epochs"], 3)
        self.assertEqual(self.train_summary["device"], "cpu")
        self.assertGreaterEqual(self.train_summary["best_epoch"], 1)
        self.assertIsNotNone(self.train_summary["best_validation_relative_l2"])

    def test_training_loss_decreased(self) -> None:
        history = json.loads((self.run_dir / "history.json").read_text(encoding="utf-8"))
        first = history["history"][0]["train"]["mse"]
        last = history["history"][-1]["train"]["mse"]
        self.assertLess(last, first)

    def test_evaluation_covered_the_whole_test_split(self) -> None:
        self.assertEqual(self.eval_result["split"], "test")
        self.assertEqual(self.eval_result["samples"], SPLITS["test"])
        self.assertEqual(self.eval_result["time_points"], N_POINTS)
        self.assertEqual(self.eval_result["target_channels"], 2)
        self.assertEqual(self.eval_result["elements"], SPLITS["test"] * N_POINTS * 2)
        self.assertIn("pass", self.eval_result)

    def test_run_directory_contract(self) -> None:
        for name in (
            "config.yaml",
            "spectral_basis.pt",
            "best_model.pt",
            "last_model.pt",
            "history.json",
            "metrics.csv",
            "train.log",
            "test_metrics.json",
            "predictions.npz",
        ):
            self.assertTrue((self.run_dir / name).is_file(), f"missing {name}")

    def test_test_metrics_artifact_fields(self) -> None:
        metrics = json.loads((self.run_dir / "test_metrics.json").read_text(encoding="utf-8"))
        for key in ("samples", "time_points", "target_channels", "relative_l2",
                    "relative_l2_float64", "threshold", "pass", "mse", "rmse",
                    "inference_seconds"):
            self.assertIn(key, metrics)
        self.assertEqual(metrics["threshold"], 1.0e-4)
        self.assertEqual(metrics["samples"], SPLITS["test"])

    def test_predictions_archive_shapes(self) -> None:
        archive = np.load(self.run_dir / "predictions.npz")
        self.assertEqual(archive["predictions"].shape, (SPLITS["test"], N_POINTS, 2))
        self.assertEqual(archive["targets"].shape, (SPLITS["test"], N_POINTS, 2))

    def test_figures_exist_and_are_not_empty(self) -> None:
        names = [
            "training_curve.png",
            "validation_curve.png",
            "prediction_sample_001.png",
            "prediction_sample_002.png",
            "prediction_sample_003.png",
            "error_sample_001.png",
            "error_sample_002.png",
            "error_sample_003.png",
        ]
        figures_dir = self.run_dir / "figures"
        for name in names:
            path = figures_dir / name
            self.assertTrue(path.is_file(), f"missing {name}")
            self.assertGreater(path.stat().st_size, 1000, f"{name} looks empty")
        self.assertEqual(len(self.figures), len(names))

    def test_dataset_inspection_prints_no_physical_parameters(self) -> None:
        import contextlib
        import io

        module = load_script("inspect_dataset")
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            summary = module.main(["--config", str(self.config_path)])
        text = stream.getvalue()
        for needle in ("mass_kg", "Izz_kg_m2", "actuator_mapping_B", "measurement_matrix_H"):
            self.assertNotIn(needle, text)
        self.assertEqual(summary["raw_input_channels"]["total"], 8)
        self.assertEqual(summary["target_channels"]["total"], 2)
        self.assertEqual(summary["splits"], SPLITS)


if __name__ == "__main__":
    unittest.main()
