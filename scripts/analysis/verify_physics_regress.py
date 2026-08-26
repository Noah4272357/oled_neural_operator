"""Data-level verification of the physics: f_dist = M q̈ - B u.

Diagnostic ladder:
  1. enc 2nd difference vs H @ q_ddot (true acceleration): does s̈ == H q̈?
  2. LS f_dist ~ [q_ddot, u] (TRUE acceleration + force) -> rl2: is the
     disturbance a time-domain linear function of the true state at all?
  3. LS f_dist ~ [s̈, u] (encoder-difference) -> rl2 (sanity check).

Usage (project .venv, CPU):
  .venv/bin/python scripts/analysis/verify_physics_regress.py
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import OLEDNeuralOperatorDataset, FIELD_PATHS
from src.data.dataloader import build_dataloaders

DATA_ROOT = Path("~/data/neural_operator_3").expanduser()


@contextmanager
def h5py_file(path):
    with h5py.File(path, "r") as f:
        yield f


def sdotdot(s):
    d = torch.zeros_like(s)
    d[1:-1] = (s[2:] - 2.0 * s[1:-1] + s[:-2]) / 1e-6
    d[0] = d[2]
    d[-1] = d[-3]
    return d


def main() -> None:
    manifest = json.loads((DATA_ROOT / "dataset_manifest.json").read_text())
    H = torch.tensor(manifest["model"]["encoder_measurement_matrix_H"], dtype=torch.float64)

    ds = OLEDNeuralOperatorDataset(
        DATA_ROOT, "train",
        input_fields=["force", "encoder_displacement"],
        target_fields=["disturbance"],
        include_metadata=False,
    )

    # ---- 1. s̈ vs H q̈ on a few samples ----
    rms_sdd = rms_hqdd = rms_diff = 0.0
    for i in range(5):
        rec = ds.records[i]
        with h5py_file(rec["_path"]) as f:
            enc = torch.as_tensor(np.asarray(f[FIELD_PATHS["encoder_displacement"]]),
                                  dtype=torch.float64)
            qdd = torch.as_tensor(np.asarray(f[FIELD_PATHS["acceleration"]]),
                                  dtype=torch.float64)
        sdd = sdotdot(enc)
        hqdd = qdd @ H.T
        rms_sdd += sdd.square().mean().item()
        rms_hqdd += hqdd.square().mean().item()
        rms_diff += (sdd - hqdd).square().mean().item()
    print(f"[1] rms(s̈)={np.sqrt(rms_sdd/5):.3e} rms(Hq̈)={np.sqrt(rms_hqdd/5):.3e} "
          f"rms(s̈-Hq̈)={np.sqrt(rms_diff/5):.3e}")

    # ---- 3. LS f_dist ~ [s̈, u] (differenced encoder), interior rows ----
    data_cfg = {
        "root": str(DATA_ROOT),
        "input_fields": ["force", "encoder_displacement"],
        "target_fields": ["disturbance"],
        "time_start": 0, "time_stop": None, "time_stride": 1,
        "preprocessing": {"name": "none"},
        "memory_cache": False,
        "batch_size": 128, "eval_batch_size": 128, "num_workers": 0,
        "max_train_samples": None, "max_val_samples": None, "max_test_samples": None,
    }
    loaders = build_dataloaders(data_cfg, seed=20260810)

    def ls_sdd(loader, max_fit=30, name="train"):
        X, Y = [], []
        for bi, b in enumerate(loader):
            if bi >= max_fit:
                break
            u = b["input"][:, :, 0:4].double()
            enc = b["input"][:, :, 4:8].double()
            f = b["target"].double()
            sdd = sdotdot(enc)
            X.append(torch.cat([sdd, u], -1)[:, 4:497, :].reshape(-1, 8))
            Y.append(f[:, 4:497, :].reshape(-1, 2))
        X = torch.cat(X)
        Y = torch.cat(Y)
        W = torch.linalg.lstsq(X, Y).solution
        num = den = 0.0
        for bi, b in enumerate(loader):
            if bi < max_fit or bi >= max_fit + 5:
                continue
            u = b["input"][:, :, 0:4].double()
            enc = b["input"][:, :, 4:8].double()
            f = b["target"].double()
            sdd = sdotdot(enc)
            Xt = torch.cat([sdd, u], -1)[:, 4:497, :].reshape(-1, 8)
            Yt = f[:, 4:497, :].reshape(-1, 2)
            p = Xt @ W
            num += (p - Yt).square().sum().item()
            den += Yt.square().sum().item()
        print(f"[3] {name}: LS [s̈,u] interior rl2 = "
              f"{np.sqrt(num/den) if den else float('nan'):.4f}")

    for name in ("train", "test"):
        ls_sdd(loaders[name], name=name)

    # ---- 2. LS f_dist ~ [q̈_true, u] on n train samples (HDF5 direct) ----
    n = 400
    Xq, Xu, Y = [], [], []
    for i in range(n):
        rec = ds.records[i]
        with h5py_file(rec["_path"]) as f:
            qdd = torch.as_tensor(np.asarray(f[FIELD_PATHS["acceleration"]]),
                                  dtype=torch.float64)
            u = torch.as_tensor(np.asarray(f[FIELD_PATHS["force"]]), dtype=torch.float64)
            dist = torch.as_tensor(np.asarray(f[FIELD_PATHS["disturbance"]]),
                                   dtype=torch.float64)
        Xq.append(qdd[4:497])
        Xu.append(u[4:497])
        Y.append(dist[4:497])
    Xq = torch.cat(Xq)
    Xu = torch.cat(Xu)
    Y = torch.cat(Y)
    W = torch.linalg.lstsq(torch.cat([Xq, Xu], -1), Y).solution
    p = torch.cat([Xq, Xu], -1) @ W
    rl2 = torch.sqrt((p - Y).square().sum() / Y.square().sum()).item()
    print(f"[2] LS f_dist ~ [q̈_true, u] interior rl2 (train {n} samples) = {rl2:.6e}")
    print(f"    W x-row: q̈ coeffs = {W[:3, 0].tolist()}")
    print(f"    W x-row: u coeffs  = {W[3:, 0].tolist()}")


if __name__ == "__main__":
    main()
