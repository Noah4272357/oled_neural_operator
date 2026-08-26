"""RegressionMetrics equivalence: async GPU-style accumulators vs naive floats."""

from __future__ import annotations

import math
import unittest

try:
    import torch
    from src.training.metrics import RegressionMetrics
except ImportError:  # pragma: no cover - stdlib-only test runners
    torch = None
    RegressionMetrics = None


def _naive_update(state: dict, prediction: torch.Tensor, target: torch.Tensor) -> None:
    difference = prediction.detach() - target
    state["squared_error"] += difference.square().sum().item()
    state["squared_target"] += target.square().sum().item()
    state["element_count"] += target.numel()


@unittest.skipIf(torch is None, "torch not available")
class MetricsEquivalenceTests(unittest.TestCase):
    def test_matches_naive_float_accumulation(self) -> None:
        torch.manual_seed(7)
        accelerated = RegressionMetrics()
        naive = {"squared_error": 0.0, "squared_target": 0.0, "element_count": 0}
        for _ in range(5):
            prediction = torch.randn(2, 501, 8)
            target = torch.randn(2, 501, 8)
            accelerated.update(prediction, target)
            _naive_update(naive, prediction, target)
        report = accelerated.compute(seconds=1.25)
        self.assertAlmostEqual(report["mse"], naive["squared_error"] / naive["element_count"], places=6)
        expected_relative_l2 = math.sqrt(
            naive["squared_error"] / max(naive["squared_target"], 1e-30)
        )
        self.assertAlmostEqual(report["relative_l2"], expected_relative_l2, places=6)
        self.assertEqual(report["seconds"], 1.25)

    def test_empty_metrics_report_zeros(self) -> None:
        report = RegressionMetrics().compute(seconds=0.0)
        self.assertEqual(report["mse"], 0.0)
        self.assertEqual(report["rmse"], 0.0)
        self.assertEqual(report["relative_l2"], 0.0)

    def test_bf16_promoted_to_fp32_equivalence(self) -> None:
        torch.manual_seed(11)
        fp32 = RegressionMetrics()
        bf16 = RegressionMetrics()
        for _ in range(3):
            prediction = torch.randn(2, 501, 8, dtype=torch.bfloat16)
            target = torch.randn(2, 501, 8, dtype=torch.float32)
            fp32.update(prediction.float(), target)
            bf16.update(prediction, target)  # internal .float() promotion
        a = fp32.compute(0.0)
        b = bf16.compute(0.0)
        self.assertAlmostEqual(a["relative_l2"], b["relative_l2"], places=6)


if __name__ == "__main__":
    unittest.main()
