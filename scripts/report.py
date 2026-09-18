"""Acceptance STEP 6/7: turn a finished run directory into figures.

Consumes ``history.json`` and ``predictions.npz`` from the run directory and
writes:

    figures/training_curve.png          training metric vs epoch
    figures/validation_curve.png        validation relative L2 vs epoch,
                                        with the 1e-4 acceptance limit drawn
    figures/prediction_sample_00{1,2,3}.png   Fx and Fy, truth vs prediction
    figures/error_sample_00{1,2,3}.png        residual time series

The three example samples are the **first, middle and last** sample of the test
split (indices 0, 100, 199 for the 200-sample split).  They are fixed by
position and were never selected by looking at the errors -- no cherry-picking.

No notebook, no dashboard, no web app.

Usage:
    python scripts/report.py --run-dir runs/acceptance_20260918-120000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ACCEPTANCE_THRESHOLD = 1.0e-4
# Fixed by position in the test split: first, middle, last.
EXAMPLE_INDICES: Tuple[int, ...] = (0, 100, 199)
CHANNEL_LABELS = ("Fx", "Fy")


def _history_series(history: Dict[str, Any]) -> Dict[str, List[float]]:
    records = history["history"]
    epochs = [int(record["epoch"]) for record in records]
    train_loss = [float(record["train"]["mse"]) for record in records]
    train_rl2 = [float(record["train"]["relative_l2"]) for record in records]
    val_epochs = [int(r["epoch"]) for r in records if r.get("validation")]
    val_rl2 = [float(r["validation"]["relative_l2"]) for r in records if r.get("validation")]
    val_mse = [float(r["validation"]["mse"]) for r in records if r.get("validation")]
    return {
        "epochs": epochs,
        "train_mse": train_loss,
        "train_rl2": train_rl2,
        "val_epochs": val_epochs,
        "val_rl2": val_rl2,
        "val_mse": val_mse,
    }


def plot_training_curve(series: Dict[str, List[float]], path: Path) -> None:
    figure, axes = plt.subplots(figsize=(7.5, 4.5))
    axes.semilogy(series["epochs"], series["train_mse"], color="#1f77b4", label="train MSE")
    axes.semilogy(series["val_epochs"], series["val_mse"], "o-", color="#d62728",
                  markersize=3.5, label="validation MSE")
    axes.set_xlabel("epoch")
    axes.set_ylabel("MSE (log scale)")
    axes.set_title("Training curve")
    axes.grid(True, which="both", alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_validation_curve(series: Dict[str, List[float]], path: Path) -> None:
    figure, axes = plt.subplots(figsize=(7.5, 4.5))
    axes.semilogy(series["val_epochs"], series["val_rl2"], "o-", color="#2ca02c",
                  markersize=4, label="validation global relative L2")
    axes.axhline(ACCEPTANCE_THRESHOLD, color="#d62728", linestyle="--",
                 label=f"acceptance limit {ACCEPTANCE_THRESHOLD:g}")
    axes.set_xlabel("epoch")
    axes.set_ylabel("relative L2 (log scale)")
    axes.set_title("Validation curve")
    axes.grid(True, which="both", alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_predictions(
    target: np.ndarray, prediction: np.ndarray, index: int, path: Path, seconds: np.ndarray
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(8.5, 6.0), sharex=True)
    for channel, label in enumerate(CHANNEL_LABELS):
        axes[channel].plot(seconds, target[:, channel], color="#1f77b4",
                           linewidth=1.6, label=f"{label} truth")
        axes[channel].plot(seconds, prediction[:, channel], color="#d62728",
                           linewidth=1.2, linestyle="--", label=f"{label} prediction")
        axes[channel].set_ylabel(f"{label} (N)")
        axes[channel].grid(True, alpha=0.3)
        axes[channel].legend(loc="upper right", fontsize=8)
    axes[1].set_xlabel("time (s)")
    figure.suptitle(f"Test sample {index:03d}: disturbance truth vs prediction")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_error(
    target: np.ndarray, prediction: np.ndarray, index: int, path: Path, seconds: np.ndarray
) -> None:
    residual = prediction - target
    figure, axes = plt.subplots(2, 1, figsize=(8.5, 6.0), sharex=True)
    for channel, label in enumerate(CHANNEL_LABELS):
        axes[channel].axhline(0.0, color="#7f7f7f", linewidth=0.8)
        axes[channel].plot(seconds, residual[:, channel], color="#9467bd", linewidth=1.2,
                           label=f"{label} residual")
        axes[channel].set_ylabel(f"{label} error (N)")
        axes[channel].grid(True, alpha=0.3)
        axes[channel].legend(loc="upper right", fontsize=8)
    axes[1].set_xlabel("time (s)")
    figure.suptitle(f"Test sample {index:03d}: prediction residual")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, default=None,
                        help="default: <run-dir>/figures")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> List[Path]:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    figures_dir = args.figures_dir or (run_dir / "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    series = _history_series(history)

    written: List[Path] = []
    path = figures_dir / "training_curve.png"
    plot_training_curve(series, path)
    written.append(path)

    path = figures_dir / "validation_curve.png"
    plot_validation_curve(series, path)
    written.append(path)

    archive = np.load(run_dir / "predictions.npz")
    predictions = archive["predictions"]
    targets = archive["targets"]
    n_samples, n_points, _ = targets.shape
    seconds = np.arange(n_points, dtype=np.float64) / 1000.0  # 1 kHz canonical grid

    for offset, index in enumerate(EXAMPLE_INDICES):
        if index >= n_samples:
            index = min(index, n_samples - 1)
        path = figures_dir / f"prediction_sample_{offset + 1:03d}.png"
        plot_predictions(targets[index], predictions[index], index, path, seconds)
        written.append(path)

        path = figures_dir / f"error_sample_{offset + 1:03d}.png"
        plot_error(targets[index], predictions[index], index, path, seconds)
        written.append(path)

    print(
        "\n".join(
            [
                "",
                f"Figures written to {figures_dir}",
                f"  Example test samples: {list(EXAMPLE_INDICES)}"
                " (first / middle / last of the test split, fixed by position)",
            ]
            + [f"  {item.name}" for item in written]
        ),
        flush=True,
    )
    return written


if __name__ == "__main__":
    main()
