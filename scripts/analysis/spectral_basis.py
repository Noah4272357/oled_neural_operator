"""Fixed whitening basis for the dense cross-frequency spectral model.

Computes, from the train split only:
  sd  per-column std of the flattened float64 feature matrix
  Vd  top-d right singular vectors of (features / sd)
  Sd  their singular values
  sy  one scalar std of the flattened target

Stored as a persistent buffer file consumed by ``SpectralDenseMap``.  These are
preprocessing statistics, not a fitted solution: no least-squares or pinv is
ever taken of the targets against the features.  The model itself (``W``) is
trained by SGD in ``scripts/train.py``.

The whitening by ``Sd`` is what makes plain SGD work: the differenced
features' Gram has condition number ~1e15, so without it the quadratic is
effectively unoptimizable by a first-order method.

Example:
    python scripts/analysis/spectral_basis.py --config configs/ld_spectral_dense.yaml \
        --d 1280 --output outputs/spectral_basis/basis_16ch_f64_d1280.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from src.data.dataloader import build_dataloaders  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def features(x: torch.Tensor, f_in: int) -> torch.Tensor:
    """Per-bin [Re(all channels), Im(all channels)] for bins 0..f_in."""
    z = torch.fft.rfft(x.double(), dim=1)[:, : f_in + 1, :]
    return torch.cat([z.real, z.imag], dim=-1).reshape(x.shape[0], -1)


def targets(y: torch.Tensor) -> torch.Tensor:
    """Per-bin [Re(all channels), Im(all channels)] over the full spectrum."""
    z = torch.fft.rfft(y.double(), dim=1)
    return torch.cat([z.real, z.imag], dim=-1).reshape(y.shape[0], -1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--d", type=int, default=1280)
    parser.add_argument("--f-in", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    config = load_config(args.config)
    data = config["data"]
    if not data.get("transform_dtype"):
        raise SystemExit(
            "data.transform_dtype must be set: the basis is invalid on "
            "float32-quantized difference features."
        )
    loaders = build_dataloaders(data, seed=args.seed)

    feats, targs = [], []
    n_points = None
    for batch in loaders["train"]:
        n_points = int(batch["target"].shape[1])
        feats.append(features(batch["input"], args.f_in))
        targs.append(targets(batch["target"]))
    F = torch.cat(feats)
    Y = torch.cat(targs)
    del feats, targs

    n_bins = n_points // 2 + 1
    in_channels = int(F.shape[1] // (2 * (args.f_in + 1)))
    out_channels = int(Y.shape[1] // (2 * n_bins))

    sd = F.std(0).clamp_min(1e-12)
    sy = Y.std().clamp_min(1e-12)
    Z = F / sd
    print(f"[basis] features {tuple(F.shape)}  targets {tuple(Y.shape)}", flush=True)

    _, S, Vh = torch.linalg.svd(Z, full_matrices=False)
    d = min(args.d, int(S.numel()))
    print(f"[basis] singular values: S[0]={float(S[0]):.4e} "
          f"S[d-1]={float(S[d - 1]):.4e} ratio={float(S[d - 1] / S[0]):.3e} "
          f"S[-1]={float(S[-1]):.3e}", flush=True)
    print("[basis] decay " + " ".join(
        f"{i}:{float(S[i] / S[0]):.2e}" for i in range(0, d, max(1, d // 10))), flush=True)

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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(basis, args.output)
    print(f"[basis] wrote {args.output} (d={d})", flush=True)


if __name__ == "__main__":
    main()
