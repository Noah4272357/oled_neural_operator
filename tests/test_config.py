"""Dependency-light tests for configuration resolution and run directories."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.utils.config import load_config, resolve_run_dir, save_resolved_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_CONFIG = PROJECT_ROOT / "configs" / "acceptance.yaml"


class ConfigTests(unittest.TestCase):
    def test_nested_override_applies(self) -> None:
        config = load_config(
            ACCEPTANCE_CONFIG,
            {"data": {"batch_size": 3}, "training": {"epochs": 7}},
        )
        self.assertEqual(config["data"]["batch_size"], 3)
        self.assertEqual(config["training"]["epochs"], 7)

    def test_acceptance_config_is_a_complete_final_recipe(self) -> None:
        config = load_config(ACCEPTANCE_CONFIG)
        self.assertEqual(config["device"], "cpu")
        self.assertEqual(config["model"]["name"], "spectral_dense")
        self.assertEqual(config["optimizer"]["name"], "sgd")
        self.assertEqual(config["scheduler"]["name"], "cosine")
        self.assertEqual(config["training"]["epochs"], 100)
        self.assertEqual(config["training"]["validate_every"], 5)
        self.assertEqual(config["data"]["input_fields"], ["force", "encoder_displacement"])
        self.assertEqual(config["data"]["target_fields"], ["disturbance"])
        self.assertIn("oled_microstage_inverse_disturbance", config["data"]["root"])

    def test_non_positive_batch_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size"):
            load_config(ACCEPTANCE_CONFIG, {"data": {"batch_size": 0}})

    def test_empty_field_list_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "input_fields"):
            load_config(ACCEPTANCE_CONFIG, {"data": {"input_fields": []}})

    def test_difference_features_require_float64_transform(self) -> None:
        """A float32 transform injects a quantisation floor the basis cannot undo."""
        with self.assertRaisesRegex(ValueError, "transform_dtype"):
            load_config(ACCEPTANCE_CONFIG, {"data": {"transform_dtype": None}})

    def test_unknown_device_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported device"):
            load_config(ACCEPTANCE_CONFIG, {"device": "tpu"})

    def test_eta_min_above_learning_rate_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "eta_min"):
            load_config(
                ACCEPTANCE_CONFIG,
                {"optimizer": {"lr": 1.0}, "scheduler": {"eta_min": 2.0}},
            )

    def test_explicit_run_dir_is_used_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "acceptance_fixed"
            config = load_config(ACCEPTANCE_CONFIG)
            self.assertEqual(resolve_run_dir(config, target), target)
            self.assertTrue(target.is_dir())

    def test_generated_run_directories_do_not_collide(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(
                ACCEPTANCE_CONFIG, {"experiment": {"root": directory}}
            )
            first = resolve_run_dir(config)
            second = resolve_run_dir(config)
            self.assertTrue(first.name.startswith("acceptance_"))
            self.assertNotEqual(first, second)

    def test_resolved_config_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            config = load_config(ACCEPTANCE_CONFIG, {"training": {"epochs": 3}})
            save_resolved_config(config, path)
            self.assertEqual(load_config(path)["training"]["epochs"], 3)


if __name__ == "__main__":
    unittest.main()
