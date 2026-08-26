"""MemoryCachedDataset semantics. Requires torch (project venv)."""

from __future__ import annotations

import unittest

try:
    import torch
    from torch.utils.data import Dataset
    from src.data.cache import MemoryCachedDataset
except ImportError:  # pragma: no cover - stdlib-only test runners
    torch = None
    Dataset = None
    MemoryCachedDataset = None


class FakeDataset(Dataset):
    """Deterministic in-memory dataset with channel metadata."""

    def __init__(self, size: int, calls: list) -> None:
        self.size = size
        self.calls = calls
        self.input_channels = 8
        self.target_channels = 2

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        self.calls.append(index)
        return {
            "input": torch.full((501, 8), float(index)),
            "target": torch.full((501, 2), float(-index)),
            "grid": torch.arange(501, dtype=torch.float32),
        }


@unittest.skipIf(torch is None, "torch not available")
class MemoryCacheTests(unittest.TestCase):
    def test_warm_matches_underlying_and_stops_calls(self) -> None:
        calls: list = []
        wrapped = MemoryCachedDataset(FakeDataset(16, calls)).warm()
        self.assertEqual(len(wrapped), 16)
        for index in range(16):
            item = wrapped[index]
            self.assertEqual(item["input"][0, 0].item(), float(index))
            self.assertEqual(item["target"][0, 0].item(), float(-index))
        # warm() consumed all samples; cached reads must not re-call the source.
        self.assertEqual(calls, list(range(16)))

    def test_lazy_fill_never_double_reads(self) -> None:
        calls: list = []
        wrapped = MemoryCachedDataset(FakeDataset(8, calls))
        for _ in range(3):
            wrapped[5]
            wrapped[2]
        self.assertEqual(calls, [5, 2])  # each index read exactly once

    def test_returned_dicts_are_detached_copies(self) -> None:
        calls: list = []
        wrapped = MemoryCachedDataset(FakeDataset(4, calls)).warm()
        first = wrapped[0]
        first["input"][0, 0] = 123.0
        second = wrapped[0]
        self.assertIsNot(first, second)
        self.assertNotEqual(second["input"][0, 0].item(), 123.0)

    def test_attribute_forwarding(self) -> None:
        calls: list = []
        wrapped = MemoryCachedDataset(FakeDataset(4, calls))
        self.assertEqual(wrapped.input_channels, 8)
        self.assertEqual(wrapped.target_channels, 2)


if __name__ == "__main__":
    unittest.main()
