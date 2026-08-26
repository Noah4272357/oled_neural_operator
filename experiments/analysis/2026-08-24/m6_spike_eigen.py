"""M6 spike: joint Gram spectrum of the pure-spectral problem.

Why: the S1 shared-map run (f64 AdamW lr=1e-3, 10 ep) barely moved (train rl2
0.9995) and SGD-m lr=0.1 diverged (3.4). For the convex quadratic
  L(M) = ||X M - D||^2 / ||D||^2   (complex, X: (N*bins, 13), D: (N*bins, 2))
GD convergence rate is governed by the eigenvalue spectrum of G = X^H X
(13x13 complex, 26 real) and how the target projects onto each eigendirection.
This script reports: eigenvalues (sorted), per-direction target projection,
and the predicted GD/Adam reachable rates for bins 3..250 (fit domain) and
bins 0..2 separately. Also prints per-channel input scales (13 channels) to
diagnose Adam's per-element normalization mismatch.

Usage: .venv/bin/python experiments/analysis/2026-08-24/m6_spike_eigen.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import build_dataloaders

DATA_ROOT = str(Path.home() / "data/neural_operator_3")
SEED = 20260810
N_PTS = 501
N_BIN = N_PTS // 2 + 1
N_IN = 13
N_OUT = 2


def main() -> None:
    config = {
        "root": DATA_ROOT,
        "input_fields": ["force", "displacement", "velocity", "acceleration"],
        "target_fields": ["disturbance"],
        "time_start": 0, "time_stop": None, "time_stride": 1,
        "preprocessing": {"name": "none"}, "memory_cache": False,
        "batch_size": 64, "eval_batch_size": 64, "num_workers": 4,
        "max_train_samples": None, "max_val_samples": None, "max_test_samples": None,
    }
    loaders = build_dataloaders(config, SEED)
    train = loaders["train"].dataset

    # accumulate Gram + rhs over bins (complex128)
    G = np.zeros((N_IN, N_IN), dtype=np.complex128)   # joint over all bins
    G_tone = np.zeros((N_IN, N_IN), dtype=np.complex128)
    rhs = np.zeros((N_IN, N_OUT), dtype=np.complex128)
    rhs_tone = np.zeros((N_IN, N_OUT), dtype=np.complex128)
    x2 = np.zeros(N_IN, dtype=np.float64)
    d2_total = 0.0  # ||D||^2 accumulated over all bins (frequency domain)
    n_pts = 0
    for i in range(len(train)):
        item = train[i]
        x = item["input"].numpy().astype(np.float64)
        d = item["target"].numpy().astype(np.float64)
        xft = np.fft.rfft(x, axis=0)
        dft = np.fft.rfft(d, axis=0)
        x2 += np.sum(np.abs(x) ** 2, axis=0)
        n_pts += x.shape[0]
        d2_total += np.sum(np.abs(dft) ** 2)
        for k in range(N_BIN):
            xk = xft[k][:, None]      # (13,1)
            dk = dft[k][:, None]      # (2,1)
            G += xk.conj() @ xk.T
            rhs += xk.conj() @ dk.T
            if 3 <= k <= 7:
                G_tone += xk.conj() @ xk.T
                rhs_tone += xk.conj() @ dk.T

    print(f"[data] n={len(train)}, {n_pts} points, per-channel RMS input:")
    for c in range(N_IN):
        print(f"  ch{c:2d}: rms = {np.sqrt(x2[c]/n_pts):.3e}")

    for label, Gg, rg in (("ALL bins 0..250", G, rhs), ("TONE bins 3..7", G_tone, rhs_tone)):
        evals, evecs = np.linalg.eigh(Gg)  # Hermitian complex Gram
        # contribution of each eigendirection to the optimal objective
        # (units of |d_ft|^2): |v^H r|^2 / lambda
        proj = np.zeros_like(evals)
        for j in range(len(evals)):
            v = evecs[:, j]
            proj[j] = np.sum(np.abs(v.conj() @ rg) ** 2) / max(evals[j], 1e-300)
        order = np.argsort(-evals)
        print(f"\n[{label}] eigvals (sorted desc):")
        print("  " + " ".join(f"{evals[j]:.2e}" for j in order))
        print("  cond =", f"{evals[order[0]]/max(evals[order[-1]],1e-300):.2e}")
        print("  proj_j/||D||^2 per direction (loss floor contribution frac, sorted with eigvals):")
        print("  " + " ".join(f"{proj[j]/d2_total:.2e}" for j in order))
        # if a direction's contribution exceeds 1e-8 of ||D||^2 (rl2 > 1e-4),
        # it must be optimized well; report its eigenvalue as the GD rate limiter
        print("  directions with proj > 1e-8*||D||^2 (must be fit for the 1e-4 gate):")
        for j in order:
            frac = proj[j] / d2_total
            if frac > 1e-8:
                print(f"    eigval {evals[j]:.3e}  proj_frac {frac:.3e}")


if __name__ == "__main__":
    main()
