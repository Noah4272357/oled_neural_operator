"""Prepare the fixed whitening matrix W for the pure-spectral model (M6-1).

W = C^{-1/2}: the Hermitian square root of the inverse per-pair joint
train Gram

    C = (1 / (N_samples * N_bin)) * sum_n sum_k x_ft[n,k]^H x_ft[n,k]

of the rfft features of the chosen problem (N_bin = N_pts // 2 + 1), so
z = x_ft W has identity covariance per (sample, bin) pair.  This
preconditioner is what makes the S5a pure-SGD recipe converge (see
m6_spike_spectral.py::train_whiten); without the /N_pairs normalization
the optimal map inflates ~1120x and first-order optimizers appear frozen.

Deterministic: one ordered pass over the train split, no RNG.  Output is a
CxC complex128 tensor (C = input channels; 13x13 for the 13ch problem,
8x8 for the 8ch recommended problem) (torch.save) loaded as a persistent
buffer by src/models/spectral.py::SpectralMap (whiten_path).

Usage (project .venv, CPU):
  .venv/bin/python scripts/analysis/spectral_whiten.py \
      [--data-root ~/data/neural_operator_3] \
      [--input-fields force encoder_displacement] \
      [--output outputs/spectral_whiten/whiten_8ch.pt]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import build_dataloaders


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default="~/data/neural_operator_3"
    )
    parser.add_argument(
        "--input-fields",
        nargs="+",
        default=["force", "displacement", "velocity", "acceleration"],
        help=(
            "Input fields to whiten (default: 13ch M6 problem; "
            "8ch recommended problem: force encoder_displacement)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="outputs/spectral_whiten/whiten.pt",
    )
    args = parser.parse_args()

    config = {
        "root": str(Path(args.data_root).expanduser()),
        "input_fields": list(args.input_fields),
        "target_fields": ["disturbance"],
        "time_start": 0,
        "time_stop": None,
        "time_stride": 1,
        "preprocessing": {"name": "none"},
        "memory_cache": False,
        "batch_size": 64,
        "eval_batch_size": 64,
        "num_workers": 0,
        "max_train_samples": None,
        "max_val_samples": None,
        "max_test_samples": None,
    }
    loaders = build_dataloaders(config, seed=20260810)
    train = loaders["train"].dataset
    n_samples = len(train)
    print(f"[whiten] train split: {n_samples} samples", flush=True)

    in_channels = None
    n_bin = None
    G = None
    t0 = time.perf_counter()
    for i in range(n_samples):
        x = train[i]["input"].numpy().astype(np.float64)  # (N_pts, 13)
        if G is None:
            n_pts = x.shape[0]
            in_channels = x.shape[1]
            n_bin = n_pts // 2 + 1  # 501 -> 251, derived from the data
            G = np.zeros((in_channels, in_channels), dtype=np.complex128)
            print(
                f"[whiten] n_pts={n_pts}, n_bin={n_bin}, channels={in_channels}",
                flush=True,
            )
        xft = np.fft.rfft(x, axis=0)  # (n_bin, 13)
        G += xft.conj().T @ xft
        if (i + 1) % 500 == 0:
            print(
                f"[whiten] {i + 1}/{n_samples} "
                f"({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )

    n_pairs = n_samples * n_bin
    evals, evecs = np.linalg.eigh(G / n_pairs)
    evals = np.clip(evals, 1e-10, None)
    w = (evecs * (1.0 / np.sqrt(evals))) @ evecs.conj().T
    out = torch.tensor(w, dtype=torch.cdouble)

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output)
    print(
        f"[whiten] W ({w.shape[0]}x{w.shape[1]} complex128, "
        f"normalized by {n_pairs} pairs) saved to {output} "
        f"({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
