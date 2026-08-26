"""Dependency-light tests for configuration resolution and run isolation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.utils.config import create_experiment_dir, legacy_args, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_nested_override_and_legacy_projection(self) -> None:
        config = load_config(
            PROJECT_ROOT / "configs" / "config.yaml",
            {"data": {"batch_size": 3}, "training": {"epochs": 7}},
        )
        self.assertEqual(config["data"]["batch_size"], 3)
        self.assertEqual(config["training"]["epochs"], 7)
        self.assertEqual(legacy_args(config, None)["batch_size"], 3)

    def test_generated_run_directories_do_not_collide(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(
                PROJECT_ROOT / "configs" / "config.yaml",
                {"experiment": {"root_dir": directory, "run_name": "smoke"}},
            )
            first = create_experiment_dir(config)
            second = create_experiment_dir(config)
            self.assertEqual(first.name, "smoke")
            self.assertEqual(second.name, "smoke_01")

    def test_invalid_experiment_choice_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size"):
            load_config(
                PROJECT_ROOT / "configs" / "config.yaml",
                {"data": {"batch_size": 0}},
            )

    def test_adam_optimizer_is_accepted(self) -> None:
        config = load_config(
            PROJECT_ROOT / "configs" / "config.yaml",
            {"optimizer": {"name": "adam"}},
        )
        self.assertEqual(config["optimizer"]["name"], "adam")

    def test_unknown_optimizer_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported optimizer"):
            load_config(
                PROJECT_ROOT / "configs" / "config.yaml",
                {"optimizer": {"name": "frobnicate"}},
            )


if __name__ == "__main__":
    unittest.main()
