"""Tests for data-root resolution (pure path math; no filesystem access)."""

from __future__ import annotations

import contextlib
import io
import os
import unittest
from pathlib import Path

from src.utils.paths import DATASET_DIR, resolve_data_root
from src.utils import paths as paths_module


class DataRootTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {name: os.environ.get(name) for name in ("DATA_ROOT",)}
        os.environ.pop("DATA_ROOT", None)
        # The fallback warning is once-per-process; reset per test.
        paths_module._warned_fallback = False

    def tearDown(self) -> None:
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_explicit_absolute_wins_over_env(self) -> None:
        os.environ["DATA_ROOT"] = "/somewhere/else"
        result = resolve_data_root("/explicit/path")
        self.assertEqual(result, Path("/explicit/path"))

    def test_explicit_relative_resolved_against_cwd(self) -> None:
        result = resolve_data_root("sub/dir")
        self.assertEqual(result, (Path.cwd() / "sub/dir").resolve())

    def test_explicit_tilde_expanded(self) -> None:
        result = resolve_data_root("~/somewhere")
        self.assertEqual(result, Path.home() / "somewhere")

    def test_empty_explicit_falls_through(self) -> None:
        os.environ["DATA_ROOT"] = "/somewhere/else"
        self.assertEqual(
            resolve_data_root(""), Path("/somewhere/else") / DATASET_DIR
        )
        self.assertEqual(
            resolve_data_root(None), Path("/somewhere/else") / DATASET_DIR
        )

    def test_data_root_env_used(self) -> None:
        os.environ["DATA_ROOT"] = "/tmp/oled-data-parent"
        self.assertEqual(
            resolve_data_root(), Path("/tmp/oled-data-parent") / DATASET_DIR
        )

    def test_default_fallback_warns_on_stderr(self) -> None:
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            result = resolve_data_root()
        self.assertEqual(result, Path.home() / "data" / DATASET_DIR)
        self.assertIn("WARNING", stream.getvalue())
        self.assertIn(DATASET_DIR, stream.getvalue())

    def test_default_warning_emitted_once(self) -> None:
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            resolve_data_root()
            resolve_data_root()
        self.assertEqual(stream.getvalue().count("WARNING"), 1)


if __name__ == "__main__":
    unittest.main()
