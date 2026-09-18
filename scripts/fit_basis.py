"""Test STEP 3: fit the train-only spectral basis (whitening statistics).

Computes, from the **train split only**:

    sd  per-column std of the flattened float64 feature matrix
    Vd  top-d right singular vectors of (features / sd)
    Sd  their singular values
    sy  one scalar std of the flattened target

and writes them to ``<run-dir>/spectral_basis.pt`` for
:class:`src.models.spectral_dense.SpectralDenseMap`.

These are preprocessing statistics, not a fitted solution: no least squares or
pseudo-inverse of the targets against the features is ever taken.  The model
itself (``W``) is trained by gradient descent in ``scripts/train.py``.

The whitening by ``Sd`` is what makes plain SGD work: the differenced
features' Gram has condition number ~1e15, so without it the quadratic is
effectively unoptimizable by a first-order method.

The validation and test splits are never opened here.

Usage:
    python scripts/fit_basis.py --config configs/test.yaml \
        --run-dir runs/test_20260918-120000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.dataset import OLEDNeuralOperatorDataset  # noqa: E402
from src.data.preprocessing import build_preprocessor  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import apply_data_root  # noqa: E402

DTYPES = {"float32": torch.float32, "float64": torch.float64}


def build_train_dataset(config: Dict[str, Any]) -> OLEDNeuralOperatorDataset:
    """The train split, built exactly as the training pipeline builds it."""
    data = config["data"]
    return OLEDNeuralOperatorDataset(
        Path(data["root"]),
        "train",
        input_fields=data["input_fields"],
        target_fields=data["target_fields"],
        time_start=data.get("time_start", 0),
        time_stop=data.get("time_stop"),
        time_stride=data["time_stride"],
        dtype=DTYPES[str(data.get("dtype", "float32")).lower()],
        include_metadata=False,
        transform=build_preprocessor(data.get("preprocessing", {"name": "none"})),
        transform_dtype=(
            DTYPES[str(data["transform_dtype"]).lower()]
            if data.get("transform_dtype")
            else None
        ),
    )


def train_batches(dataset: OLEDNeuralOperatorDataset, seed: int) -> DataLoader:
    """One full-batch pass in the same order the trainer sees the train split."""
    return DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def features(x: torch.Tensor, f_in: int) -> torch.Tensor:
    """Per-bin [Re(all channels), Im(all channels)] for bins 0..f_in."""
    z = torch.fft.rfft(x.double(), dim=1)[:, : f_in + 1, :]
    return torch.cat([z.real, z.imag], dim=-1).reshape(x.shape[0], -1)


def targets(y: torch.Tensor) -> torch.Tensor:
    """Per-bin [Re(all channels), Im(all channels)] over the full spectrum."""
    z = torch.fft.rfft(y.double(), dim=1)
    return torch.cat([z.real, z.imag], dim=-1).reshape(y.shape[0], -1)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/test.yaml"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="test run directory (default: the config's)")
    parser.add_argument("--output", type=Path, default=None,
                        help="override the basis path (default <run-dir>/spectral_basis.pt)")
    parser.add_argument("--d", type=int, default=256,
                        help="retained singular directions (default 256)")
    parser.add_argument("--f-in", type=int, default=100,
                        help="highest retained input rfft bin (default 100)")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.data_root is not None:
        config["data"]["root"] = str(args.data_root)
    apply_data_root(config)
    data = config["data"]
    if not data.get("transform_dtype"):
        raise SystemExit(
            "data.transform_dtype must be set: the basis is invalid on "
            "float32-quantized difference features."
        )

    run_dir = args.run_dir if args.run_dir is not None else config["experiment"].get("run_dir")
    output = args.output if args.output is not None else (
        Path(run_dir) / "spectral_basis.pt" if run_dir else Path("spectral_basis.pt")
    )

    started = time.perf_counter()
    dataset = build_train_dataset(config)
    print(f"Fitting basis on the train split only: {len(dataset)} samples", flush=True)

    collected_features, collected_targets = [], []
    n_points = None
    for batch in train_batches(dataset, int(config["seed"])):
        n_points = int(batch["target"].shape[1])
        collected_features.append(features(batch["input"], args.f_in))
        collected_targets.append(targets(batch["target"]))
    F = torch.cat(collected_features)
    Y = torch.cat(collected_targets)

    n_bins = n_points // 2 + 1
    # The number of retained bins is capped by the spectrum actually available:
    # a short trace has fewer bins than --f-in asks for, and dividing by the
    # requested count would silently mis-derive the channel width.
    bins_used = min(args.f_in + 1, n_bins)
    if bins_used < args.f_in + 1:
        print(
            f"[basis] NOTE: --f-in {args.f_in} exceeds the {n_bins} bins of a "
            f"{n_points}-point trace; retaining {bins_used} bins."
        )
    if F.shape[1] % (2 * bins_used) != 0:
        raise SystemExit(
            f"Feature width {F.shape[1]} is not divisible by 2*{bins_used}; "
            "the basis cannot be laid out consistently."
        )
    in_channels = int(F.shape[1] // (2 * bins_used))
    out_channels = int(Y.shape[1] // (2 * n_bins))

    sd = F.std(0).clamp_min(1e-12)
    sy = Y.std().clamp_min(1e-12)
    Z = F / sd

    _, S, Vh = torch.linalg.svd(Z, full_matrices=False)
    d = min(args.d, int(S.numel()))
    basis = {
        "sd": sd,
        "Vd": Vh[:d].clone(),
        "Sd": S[:d].clone(),
        "sy": sy,
        "f_in": args.f_in,
        "d": d,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "out_flat": int(Y.shape[1]),
        "n_points": n_points,
        "n_bins": n_bins,
        "preprocessing": dict(data.get("preprocessing", {})),
        "root": str(data.get("root")),
        "transform_dtype": str(data.get("transform_dtype")),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(basis, output)

    elapsed = time.perf_counter() - started
    print(
        "\n".join(
            [
                "",
                "Train-only spectral basis fitted",
                f"  Train samples       : {F.shape[0]}",
                f"  Feature width       : {F.shape[1]}",
                f"  Target width        : {Y.shape[1]}",
                f"  Retained directions : {d}",
                f"  Input channels      : {in_channels}",
                f"  Output channels     : {out_channels}",
                f"  Elapsed             : {elapsed:.1f} s",
                f"  Wrote               : {output}",
            ]
        ),
        flush=True,
    )
    return basis


if __name__ == "__main__":
    main()
