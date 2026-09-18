"""Tests for the final model: SpectralDenseMap and its basis contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.models.factory import build_model
from src.models.spectral_dense import GridAdapter, SpectralDenseMap

F_IN = 100
N_POINTS = 501
N_BINS = N_POINTS // 2 + 1
IN_CHANNELS = 16
OUT_CHANNELS = 2
FEATURE_WIDTH = 2 * (F_IN + 1) * IN_CHANNELS
OUT_FLAT = 2 * N_BINS * OUT_CHANNELS
D = 8


def make_basis(path: Path, *, in_channels: int = IN_CHANNELS,
               out_channels: int = OUT_CHANNELS, d: int = D) -> Path:
    generator = torch.Generator().manual_seed(20260810)
    torch.save(
        {
            "sd": torch.rand(FEATURE_WIDTH, generator=generator, dtype=torch.float64) + 0.5,
            # Orthonormal rows: QR of the (width, d) matrix gives Q as
            # (width, d); transposing yields the (d, width) right-singular-vector
            # layout the model expects.
            "Vd": torch.linalg.qr(
                torch.rand(FEATURE_WIDTH, d, generator=generator, dtype=torch.float64)
            )[0].t().contiguous(),
            "Sd": torch.linspace(10.0, 1.0, d, dtype=torch.float64),
            "sy": torch.tensor(1.5, dtype=torch.float64),
            "f_in": F_IN,
            "d": d,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "out_flat": OUT_FLAT,
            "n_points": N_POINTS,
            "n_bins": N_BINS,
            "preprocessing": {"name": "diff_features", "dt": 0.001, "channels": [4, 5, 6, 7]},
            "root": "unused",
            "transform_dtype": "float64",
        },
        path,
    )
    return path


class SpectralDenseMapTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.basis_path = make_basis(Path(self._tmp.name) / "basis.pt")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_forward_shape_and_dtype(self) -> None:
        model = SpectralDenseMap(IN_CHANNELS, OUT_CHANNELS, self.basis_path)
        model.eval()
        x = torch.randn(3, N_POINTS, IN_CHANNELS, dtype=torch.float64)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (3, N_POINTS, OUT_CHANNELS))
        self.assertEqual(y.dtype, torch.float64)

    def test_single_trainable_parameter_is_W(self) -> None:
        model = SpectralDenseMap(IN_CHANNELS, OUT_CHANNELS, self.basis_path)
        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        self.assertEqual(trainable, {"W"})
        self.assertEqual(tuple(model.W.shape), (D, OUT_FLAT))
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),
                         D * OUT_FLAT)

    def test_basis_statistics_are_buffers_not_parameters(self) -> None:
        model = SpectralDenseMap(IN_CHANNELS, OUT_CHANNELS, self.basis_path)
        buffers = set(dict(model.named_buffers()))
        self.assertEqual(buffers, {"sd", "Vd", "Sd", "sy"})
        for name in buffers:
            self.assertFalse(getattr(model, name).requires_grad)

    def test_basis_channel_mismatch_is_rejected(self) -> None:
        wrong_in = make_basis(Path(self._tmp.name) / "wrong_in.pt", in_channels=13)
        with self.assertRaisesRegex(ValueError, "input channels"):
            SpectralDenseMap(IN_CHANNELS, OUT_CHANNELS, wrong_in)

        wrong_out = make_basis(Path(self._tmp.name) / "wrong_out.pt", out_channels=3)
        with self.assertRaisesRegex(ValueError, "output channels"):
            SpectralDenseMap(IN_CHANNELS, OUT_CHANNELS, wrong_out)

    def test_gradients_reach_W(self) -> None:
        model = SpectralDenseMap(IN_CHANNELS, OUT_CHANNELS, self.basis_path)
        x = torch.randn(2, N_POINTS, IN_CHANNELS, dtype=torch.float64)
        target = torch.randn(2, N_POINTS, OUT_CHANNELS, dtype=torch.float64)
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward()
        self.assertIsNotNone(model.W.grad)
        self.assertTrue(torch.isfinite(model.W.grad).all())
        self.assertGreater(float(model.W.grad.abs().sum()), 0.0)


class GridAdapterTests(unittest.TestCase):
    def test_adapter_casts_inputs_and_ignores_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            basis_path = make_basis(Path(directory) / "basis.pt")
            inner = SpectralDenseMap(IN_CHANNELS, OUT_CHANNELS, basis_path)
            adapter = GridAdapter(inner, in_dtype=torch.float64)
            x = torch.randn(2, N_POINTS, IN_CHANNELS, dtype=torch.float32)
            grid = torch.zeros(2, N_POINTS)
            with torch.no_grad():
                direct = inner(x.double())
                through = adapter(x, grid)
            self.assertEqual(through.dtype, torch.float64)
            self.assertTrue(torch.allclose(direct, through))


class FactoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.basis_path = make_basis(Path(self._tmp.name) / "basis.pt")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_builds_the_final_model(self) -> None:
        model = build_model(
            {"name": "spectral_dense", "basis_path": str(self.basis_path), "init_scale": 1e-3},
            IN_CHANNELS,
            OUT_CHANNELS,
        )
        self.assertIsInstance(model, GridAdapter)
        self.assertIsInstance(model.model, SpectralDenseMap)

    def test_missing_basis_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "basis_path"):
            build_model({"name": "spectral_dense"}, IN_CHANNELS, OUT_CHANNELS)

    def test_abandoned_architectures_are_rejected(self) -> None:
        for name in ("fno1d", "fno1d_refined", "spectral", "frobnicate"):
            with self.assertRaisesRegex(ValueError, "Unsupported model"):
                build_model(
                    {"name": name, "basis_path": str(self.basis_path)},
                    IN_CHANNELS,
                    OUT_CHANNELS,
                )


if __name__ == "__main__":
    unittest.main()
