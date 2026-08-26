"""Build the m64 residual-target derived dataset (8ch nonlinear campaign).

For every sample of every split the closed-form per-bin head
(outputs/lti_head_16ch.pt, fitted by fit_lti_head_16ch.py) predicts
``head(x16) = irfft(M @ rfft(features16))``; the residual target

    r = disturbance - head(x16)

is stored at ``/disturbance/force`` of a derived HDF5 that otherwise
mirrors the original (force / encoder_displacement / time copied
verbatim).  Training a network on ``r`` concentrates all capacity on the
part the 8ch linear information limit cannot see (window-boundary
restoration), instead of re-learning the linear map.

The derived dataset lives under ``experiments/`` (run artifacts), the
original dataset is never written to.  It carries the same manifest
record structure, so train.py consumes it unchanged.

Usage (project .venv, CPU):
  .venv/bin/python scripts/analysis/build_residual_targets.py \
      [--data-root ~/data/neural_operator_3] \
      [--head outputs/lti_head_16ch.pt] \
      [--out experiments/m63_residual_targets]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import build_preprocessor

TRANSFORM_CONFIG = {"name": "diff_features", "dt": 0.001, "channels": [4, 5, 6, 7]}
SRC_FIELDS = {
    "force": "/actuators/force",
    "encoder_displacement": "/sensors/encoder_displacement",
    "disturbance": "/disturbance/force",
    "time": "/time",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default="~/data/neural_operator_3")
    parser.add_argument("--head", type=Path, default="outputs/lti_head_16ch.pt")
    parser.add_argument("--out", type=Path, default="experiments/m63_residual_targets")
    args = parser.parse_args()

    root = Path(args.data_root).expanduser()
    out = Path(args.out).expanduser()
    manifest = json.loads((root / "dataset_manifest.json").read_text())

    mk = torch.load(args.head, map_location="cpu", weights_only=True).numpy()
    assert mk.shape == (251, 2, 16), mk.shape
    transform = build_preprocessor(TRANSFORM_CONFIG)

    t0 = time.perf_counter()
    n_samples = 0
    for split in ("train", "val", "test"):
        split_dir = out / split
        split_dir.mkdir(parents=True, exist_ok=True)
        records = manifest["splits"][split]
        for rec in records:
            src = root / split / rec["path"]
            dst = split_dir / rec["path"]
            if not src.is_file():
                raise FileNotFoundError(src)
            with h5py.File(src, "r") as handle:
                force = handle[SRC_FIELDS["force"]][:]
                enc = handle[SRC_FIELDS["encoder_displacement"]][:]
                dist = handle[SRC_FIELDS["disturbance"]][:]
                time_grid = handle[SRC_FIELDS["time"]][:]
            x = np.concatenate([force, enc], axis=1)
            sample = {
                "input": torch.as_tensor(x, dtype=torch.float32),
                "target": torch.as_tensor(dist, dtype=torch.float32),
                "grid": torch.as_tensor(time_grid, dtype=torch.float32),
            }
            x16 = transform(sample)["input"].numpy().astype(np.float64)
            z = np.fft.rfft(x16, axis=0)  # (251, 16)
            pred_ft = np.einsum("ki,koi->ko", z, mk)  # (251, 2)
            head_pred = np.fft.irfft(pred_ft, n=501, axis=0)
            residual = dist.astype(np.float64) - head_pred
            with h5py.File(dst, "w") as handle:
                handle[SRC_FIELDS["force"]] = force
                handle[SRC_FIELDS["encoder_displacement"]] = enc
                handle[SRC_FIELDS["disturbance"]] = residual.astype(np.float32)
                handle[SRC_FIELDS["time"]] = time_grid
            n_samples += 1
            if n_samples % 1000 == 0:
                print(
                    f"[build_residual] {n_samples} samples "
                    f"({time.perf_counter() - t0:.1f}s)",
                    flush=True,
                )

    derived_manifest = {
        "complete": True,
        "fields": {
            "force": {"path": "/actuators/force", "shape": [501, 4]},
            "encoder_displacement": {
                "path": "/sensors/encoder_displacement",
                "shape": [501, 4],
            },
            "disturbance": {"path": "/disturbance/force", "shape": [501, 2]},
            "time": {"path": "/time", "shape": [501]},
        },
        "splits": {
            split: [
                {"path": rec["path"]} for rec in manifest["splits"][split]
            ]
            for split in ("train", "val", "test")
        },
    }
    (out / "dataset_manifest.json").write_text(
        json.dumps(derived_manifest, indent=1)
    )
    print(
        f"[build_residual] {n_samples} derived samples -> {out} "
        f"({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
