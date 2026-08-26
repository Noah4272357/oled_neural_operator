"""Closed-form least-squares initialization of the linear FNO1d.

The inverse disturbance operator d(t) = M q_ddot(t) - B u(t) is *pointwise*
linear, so it is per-bin diagonal in any orthonormal basis -- including the
FNO's padded (503-point) rfft basis.  A linear FNO (activation="none",
num_blocks=1) can represent ANY per-bin complex map M_k exactly:

    fc0 composite Q (128x14)  : zero rows 0,1; identity rows 2..15
    conv1d C                  : zero
    spectral weights1[:, :, k] : M_k embedded at (in cols 2..15, out rows 0,1)
    fc1 composite R (2x128)   : e_0, e_1 (reads rows 0,1)
    all biases                : zero

Then per bin k >= 1:  y_k = R (I + W_k + C) Q x_k = R W_k Q x_k = M_k x_k,
and the block residual R z_k = R Q x_k = 0.  The DC bin is also exact since
all biases vanish and the target is exactly zero-mean.

Three modes:

- ``analytic`` (default): the operator is known in closed form from the
  manifest -- d = M q_ddot - B u with A = [-B[:2], 0, 0, diag(m, m)] -- and
  is injected directly.  This is the exact inverse operator (linearity
  verified at rl2 ~ 1e-15), so the network reproduces the physics map to
  float32 arithmetic precision with zero training.
- ``ls``: fit per-padded-bin least-squares maps on the train split and
  inject those.  NOTE (2026-08-24): the LS fit is ill-conditioned at the
  wrench-tone bins (1-2): B_k singular values span ~19 decades there, and
  the min-norm pinv solution amplifies the (y - A x) residual along the
  tiny-sigma directions by 1/sigma_min, producing a garbage map at those
  bins (test rl2 ~ 2.3e-2) while the tone bins fit to ~3e-6.
- ``lspinv`` (M5, 2026-08-24): ridge-regularized LS map fit on the UNPADDED
  501-point rfft basis in float64 and injected into the same network
  structure.  M5 diagnosis (review-corrected 2026-08-24): the per-bin
  protocol of analyze_dataset_fourier.per_mode_ls_floor is UNIDENTIFIABLE,
  but its published floor HOLDS -- 4.8304e-6 on the 50-sample stride-4 eval
  subset and 4.8638e-6 on the full 200-sample test set (independent
  recheck).  An earlier claim that 4.83e-6 was a 50-sample subset artifact
  (the same fit scoring 9.2e2 total / 3.2e5 at bin 1 on all 200 samples)
  is RETRACTED: those numbers came from an abandoned intermediate per-bin
  implementation and are not reproducible from the floor protocol.  The
  unidentifiability manifests as TRAINING-subset instability: at the
  wrench-tone bins 1-2 the per-bin Gram has cond ~1e27-1e29 (real-stacked
  Gram ~1e31 at n=200; the padded 503-pt basis is NOT the fix -- the 503-pt
  basis is the source that sank ``ls`` only by making the conditioning
  worse), the tone-bin Gram has effective rank ~7-11 of 26 (~15 near-null
  directions), and a per-bin fit on a stride-25 training subset reaches
  ~6.7e-10 at the tone bins while the same fit on the consecutive 0..199
  training subset lands at 1.1e-2-1.6e-2 there: the per-bin map's action
  along the near-null directions of the recipe-limited input manifold is
  undetermined by any amount of per-bin data, and its Frobenius bias to
  the exact map GROWS with n (bin 4: 1.015e-3 -> 1.845e-3 -> 3.472e-3 for
  n=200/1000/5000).  Yet the floor's own bin-1 map (0.315 from A_true)
  still scores 5.4e-5 per-bin on the full 200-sample test set: the
  differences sit in directions the test does not probe (max|M|=1137.55
  at bins 1-2 is the mass coefficient, not a fit artifact).  The exact
  operator is FREQUENCY-INDEPENDENT (d = M q_ddot - B u is pointwise), so
  the correct data-driven model is a SINGLE shared complex 2x13 map fit
  jointly over the well-conditioned bins 3..250 (bins 0-2 excluded: their
  near-collinear ramp structure would dominate the joint conditioning;
  the shared map is then evaluated at every bin).  The joint fit recovers
  the exact map to ~5e-13 relative at n=5000 (data-confirmed LTI
  structure) and its per-bin test rl2 is ~1e-14 at the tone bins and
  ~1e-10 at bins 1-2; the total unpadded test rl2 is ~1e-10.  A ``--reg``
  grid (1e-12 / 1e-10 / 1e-8) is applied as ridge on the joint system
  (relative to its largest eigenvalue -- at these scales the ridge is
  numerically inert on the well-posed joint problem, so the three regs are
  expected to agree), each reg producing its own initialized checkpoint
  with per-bin Gram condition numbers and per-bin test-error decomposition.
  This is the P1 initializer for the M5 "train from an approximate LS
  start" path (F4 only disproved resume-finetuning the exact analytic
  optimum, not training from this start).

Run:
    python scripts/analysis/ls_init_fno.py \
        --config configs/f1_linear_13ch.yaml \
        --data-root ~/data/neural_operator_2 \
        --output-dir outputs/ls_init_f1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import FIELD_PATHS
from src.models.factory import build_model
from src.training.factories import build_optimizer, build_scheduler
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config

PAD = 2  # must match FNO1d.padding


def fit_ls_maps(root: Path, input_fields, n_train: int, device_note: str) -> np.ndarray:
    """Per-padded-bin LS maps M_k (out=2, in=14) fit on the train split.

    Returns complex128 array of shape (modes=128, 2, 14).
    """
    probe = h5py.File(root / "train" / "sample_000000.h5", "r")
    npts = probe[FIELD_PATHS["time"]].shape[0]
    probe.close()
    n_pad = npts + PAD
    n_bin = n_pad // 2 + 1
    print(f"[ls] {n_pad}-point padded basis, {n_bin} bins, fitting {n_train} samples")
    n_in = sum(_field_width(root, f) for f in input_fields) + 1  # + grid
    n_out = 2
    A = np.zeros((128, n_out, n_in), dtype=np.complex128)
    B = np.zeros((128, n_in, n_in), dtype=np.complex128)
    t0 = time.time()
    for i in range(n_train):
        path = root / "train" / f"sample_{i:06d}.h5"
        with h5py.File(path, "r") as fh:
            parts = [np.asarray(fh[FIELD_PATHS[f]], dtype=np.float32) for f in input_fields]
            y = np.asarray(fh[FIELD_PATHS["disturbance"]], dtype=np.float32)
            g = np.asarray(fh[FIELD_PATHS["time"]], dtype=np.float32)
        x = np.concatenate(parts, axis=1)                      # (npts, 13) float32
        x = np.pad(x, ((0, PAD), (0, 0)))                      # (503, 13)
        y = np.pad(y, ((0, PAD), (0, 0)))
        g = np.pad(g, (0, PAD))[..., None]                     # (503, 1)
        xft = np.fft.rfft(np.concatenate((x, g), axis=1), axis=0)[:128]  # (128, 14)
        yft = np.fft.rfft(y, axis=0)[:128]                                # (128, 2)
        B += xft[:, :, None] * np.conj(xft[:, None, :])
        A += yft[:, :, None] * np.conj(xft[:, None, :])
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{n_train}] {(time.time()-t0):.1f}s")
    M = np.empty((128, n_out, n_in), dtype=np.complex128)
    for k in range(128):
        M[k] = A[k] @ np.linalg.pinv(B[k])
    print(f"[ls] fit done in {(time.time()-t0):.1f}s")
    return M


def _field_width(root: Path, field: str) -> int:
    with h5py.File(root / "train" / "sample_000000.h5", "r") as fh:
        return int(np.asarray(fh[FIELD_PATHS[field]]).shape[1])


def fit_ridge_ls_maps(
    root: Path, input_fields, n_train: int, reg: float
):
    """Joint frequency-shared ridge-LS map on the UNPADDED 501-pt rfft basis.

    M5 diagnosis (2026-08-24, review-corrected): the per-bin protocol of
    ``analyze_dataset_fourier.per_mode_ls_floor`` (the 13ch "floor" 4.83e-6
    at reg=1e-10) is UNIDENTIFIABLE, but its published floor HOLDS:
    4.8304e-6 on the 50-sample stride-4 eval subset and 4.8638e-6 on the
    full 200-sample test set (independent recheck).  An earlier claim that
    the floor was an eval-subset artifact (same fit scoring 9.2e2 total /
    3.2e5 at bin 1 on all 200 samples) is RETRACTED: those numbers came
    from an abandoned intermediate per-bin implementation and are not
    reproducible from the floor protocol.  The unidentifiability manifests
    as TRAINING-subset instability instead:

    * bins 1-2: per-bin Gram cond ~1e27-1e29 (real-stacked Gram ~1e31 at
      n=200; identical for neural_operator_2 and _3; the padded 503-pt
      basis of ``ls`` is worse, not the same problem).  The
      complex-Hermitian solve (zgesv) is numerically destructive there.
      The floor's own bin-1 map (Frobenius 0.315 from A_true) still scores
      5.4e-5 per-bin on the full 200-sample test set -- the differences sit
      in directions the test inputs do not probe; max|M|=1137.55 at bins
      1-2 is the mass coefficient, not a fit artifact.
    * bins 4-7 (the tone bins, 99.98% of target energy): every per-bin fit
      (solve / lstsq / any rcond / any reg) lands ~1e-3 (Frobenius
      relative) from the exact map, and the bias GROWS with n (bin 4:
      1.015e-3 -> 1.845e-3 -> 3.472e-3 for n=200/1000/5000) -- the map's
      action along the ~15 near-null directions of the recipe-limited
      input manifold is undetermined by any amount of per-bin data.
      Per-bin test rl2 is then training-subset-dependent: ~6.7e-10 at the
      tone bins from a stride-25 training subset, 1.1e-2-1.6e-2 from the
      consecutive 0..199 training subset.

    The exact operator is FREQUENCY-INDEPENDENT (d = M q_ddot - B u is
    pointwise), so the correct data-driven model is a SINGLE shared complex
    2x13 map fit jointly over the well-conditioned bins 3..250 (bins 0-2
    excluded: their near-collinear ramp structure dominates the joint
    conditioning; the shared map is evaluated at every bin).  The joint fit
    recovers the exact map to ~5e-13 relative (data-confirmed LTI
    structure) with per-bin test rl2 ~1e-14 (tones) .. ~1e-10 (bins 1-2)
    and total ~1e-10 on the full 200-sample test set.  ``reg`` is applied
    as ridge relative to the joint system's largest eigenvalue (at the
    brief's 1e-12..1e-8 absolute scales the ridge is numerically inert on
    the well-posed joint problem).

    Feature layout matches the floor protocol: real-stacked
    [Re x_j, Im x_j] per channel (13 inputs, grid column NOT fit -- it is
    zeroed at injection like the analytic mode), target
    [Re d_0, Re d_1, Im d_0, Im d_1]; per-bin complex Gram B[k] is still
    accumulated for the condition report.

    Returns (M, conds): M complex128 (251, 2, 13) with the shared map
    replicated at every bin; conds = per-bin sigma_max / sigma_min of the
    complex Gram B[k].
    """
    probe = h5py.File(root / "train" / "sample_000000.h5", "r")
    npts = probe[FIELD_PATHS["time"]].shape[0]
    probe.close()
    n_bin = npts // 2 + 1  # 251
    n_in = sum(_field_width(root, f) for f in input_fields)  # 13, no grid
    n_out = 2
    width = 2 * n_in
    joint_bins = list(range(3, n_bin))  # 3..250; bins 0-2 excluded (see docstring)
    Bc = np.zeros((n_bin, n_in, n_in), dtype=np.complex128)  # cond numbers only
    Xj = np.empty((n_train * len(joint_bins), width))
    Yj = np.empty((n_train * len(joint_bins), 2 * n_out))
    t0 = time.time()
    row = 0
    for i in range(n_train):
        path = root / "train" / f"sample_{i:06d}.h5"
        with h5py.File(path, "r") as fh:
            parts = [np.asarray(fh[FIELD_PATHS[f]]) for f in input_fields]
            y = np.asarray(fh[FIELD_PATHS["disturbance"]])
        x = np.concatenate(parts, axis=1)  # (501, 13) float64
        xft = np.fft.rfft(x, axis=0)       # (251, 13)
        yft = np.fft.rfft(y, axis=0)       # (251, 2)
        Bc += xft[:, :, None] * np.conj(xft[:, None, :])
        T = np.concatenate((yft.real, yft.imag), axis=1)  # (251, 4)
        for k in joint_bins:
            Xj[row, 0::2] = xft[k].real    # [Re x_0, Im x_0, Re x_1, Im x_1, ...]
            Xj[row, 1::2] = xft[k].imag
            Yj[row] = T[k]
            row += 1
        if (i + 1) % 1000 == 0:
            print(f"  [lspinv reg={reg:g}] [{i+1}/{n_train}] {(time.time()-t0):.1f}s")
    lmax = np.linalg.norm(Xj) ** 2
    Xa = np.vstack([Xj, np.sqrt(reg / lmax) * np.eye(width)])
    Ya = np.vstack([Yj, np.zeros((width, 2 * n_out))])
    coeff, *_ = np.linalg.lstsq(Xa, Ya, rcond=1e-12)
    shared = coeff[0::2, :n_out].T - 1j * coeff[1::2, :n_out].T  # (2, 13)
    M = np.repeat(shared[None, ...], n_bin, axis=0)
    conds = np.empty(n_bin)
    for k in range(n_bin):
        s = np.linalg.svd(Bc[k], compute_uv=False)
        conds[k] = s[0] / s[-1] if s[-1] > 0 else float("inf")
    print(f"[lspinv reg={reg:g}] fit done in {(time.time()-t0):.1f}s | "
          f"max|M| {np.abs(shared).max():.3e} | "
          f"cond bins1-2: {conds[1]:.3e}/{conds[2]:.3e} | "
          f"cond median {np.median(conds):.3e} max {conds.max():.3e}")
    return M, conds


def per_bin_test_error(
    root: Path, input_fields, M: np.ndarray, n_test: int = 200
):
    """Per-bin squared-error / target-energy of the fitted map M on test.

    The map (13 inputs, no grid) is applied directly in the 501-point rfft
    basis (no network), so this localizes which bins dominate the residual --
    e.g. a single ill-conditioned wrench-tone bin carrying most of the error.
    """
    n_bin = M.shape[0]
    err_k = np.zeros(n_bin)
    tgt_k = np.zeros(n_bin)
    for i in range(n_test):
        path = root / "test" / f"sample_{i:06d}.h5"
        with h5py.File(path, "r") as fh:
            parts = [np.asarray(fh[FIELD_PATHS[f]]) for f in input_fields]
            y = np.asarray(fh[FIELD_PATHS["disturbance"]])
        x = np.concatenate(parts, axis=1)
        xft = np.fft.rfft(x, axis=0)
        yft = np.fft.rfft(y, axis=0)
        pred = np.einsum("koi,ki->ko", M, xft)
        err_k += np.sum(np.abs(pred - yft) ** 2, axis=1)
        tgt_k += np.sum(np.abs(yft) ** 2, axis=1)
    return err_k, tgt_k


def inject_ls_maps(model, M: np.ndarray) -> None:
    """Set model weights so per-bin map == M exactly (see module docstring)."""
    with torch.no_grad():
        # fc0 composite Q: rows 0,1 zero; rows 2..15 identity on the 14 inputs.
        fc0 = model.fc0
        w0 = torch.zeros_like(fc0[0].weight)          # (64, 14)
        w0[:14, :14] = torch.eye(14)
        fc0[0].weight.copy_(w0)
        fc0[0].bias.zero_()
        w1 = torch.zeros_like(fc0[1].weight)          # (128, 64)
        w1[2:16, :14] = torch.eye(14)
        fc0[1].weight.copy_(w1)
        fc0[1].bias.zero_()

        # conv1d branch: zero.
        conv = model.blocks[0].w
        conv.weight.zero_()
        conv.bias.zero_()

        # Spectral weights: M_k at (in cols 2..15, out rows 0,1).
        spec = model.blocks[0].conv.weights1           # (128, 128, 128) cfloat
        spec.zero_()
        n_modes = int(model.blocks[0].conv.modes)
        spec[2:16, 0:2, :n_modes] = torch.as_tensor(np.ascontiguousarray(M.transpose(2, 1, 0)))

        # fc1 composite R: e_0, e_1.
        b1, b2, b3 = model.fc1[0], model.fc1[1], model.fc1[2]
        wb1 = torch.zeros_like(b1.weight)             # (64, 128)
        wb1[:2, :2] = torch.eye(2)
        b1.weight.copy_(wb1); b1.bias.zero_()
        wb2 = torch.zeros_like(b2.weight)             # (128, 64)
        wb2[:2, :2] = torch.eye(2)
        b2.weight.copy_(wb2); b2.bias.zero_()
        wb3 = torch.zeros_like(b3.weight)             # (2, 128)
        wb3[:, :2] = torch.eye(2)
        b3.weight.copy_(wb3); b3.bias.zero_()


@torch.no_grad()
def evaluate_split(model, root: Path, split: str, indices, input_fields, batch: int = 64):
    model.eval()
    num = 0.0
    den = 0.0
    for b in range(0, len(indices), batch):
        idxs = indices[b : b + batch]
        xs, gs, ys = [], [], []
        for i in idxs:
            path = root / split / f"sample_{i:06d}.h5"
            with h5py.File(path, "r") as fh:
                parts = [np.asarray(fh[FIELD_PATHS[f]], dtype=np.float32) for f in input_fields]
                xs.append(torch.as_tensor(np.concatenate(parts, axis=1), dtype=torch.float32))
                ys.append(torch.as_tensor(np.asarray(fh[FIELD_PATHS["disturbance"]], dtype=np.float32), dtype=torch.float32))
                gs.append(torch.as_tensor(np.asarray(fh[FIELD_PATHS["time"]], dtype=np.float32), dtype=torch.float32))
        x = torch.stack(xs)
        g = torch.stack(gs)
        pred = model(x, g).float()
        y = torch.stack(ys)
        num += float(((pred - y) ** 2).sum())
        den += float((y**2).sum())
    return float(np.sqrt(num / den))


def evaluate_and_save(
    config: Dict[str, Any],
    model,
    root: Path,
    input_fields,
    out: Path,
    mode: str,
    n_train: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Shared tail: self-eval (train 200 / test 200) + v2 checkpoint + report."""
    train_rl2 = evaluate_split(model, root, "train", range(0, 200), input_fields)
    test_rl2 = evaluate_split(model, root, "test", range(0, 200), input_fields)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[eval] {mode} linear FNO ({n_params} params)")
    print(f"[eval] train relative_l2 (first 200) = {train_rl2:.6e}")
    print(f"[eval] test  relative_l2 (all 200)   = {test_rl2:.6e}")
    out.mkdir(parents=True, exist_ok=True)
    optimizer = build_optimizer(config["optimizer"], model)
    scheduler = build_scheduler(
        config["scheduler"], optimizer, int(config["training"]["epochs"])
    )
    save_checkpoint(
        out / "best_model.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0,
        validation={"train_relative_l2": train_rl2, "test_relative_l2": test_rl2},
        best_metric=test_rl2,
        config=config,
        legacy_arguments={},
    )
    report: Dict[str, Any] = {
        "train_relative_l2": train_rl2,
        "test_relative_l2": test_rl2,
        "mode": mode,
        "n_params": n_params,
        "n_train_fit": n_train,
        "n_modes": int(config["model"]["modes"]),
    }
    if mode == "lspinv":
        # the fit basis is the UNPADDED 501-pt rfft; PAD=2 is only the
        # network's forward padding, not the fit basis (review finding 5).
        # Only lspinv 模式写入：analytic/ls 的注入基底语义不同（无条件
        # 写入会误导下游读取方）。
        report["padded"] = False
        report["fit_basis"] = "unpadded_501_rfft"
    if extra:
        report.update(extra)
    with (out / "ls_init_report.json").open("w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[save] checkpoint -> {out / 'best_model.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "f1_linear_13ch.yaml")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "ls_init_f1")
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--mode", choices=("analytic", "ls", "lspinv"), default="analytic")
    parser.add_argument("--reg", type=float, nargs="+", default=[1e-10],
                        help="ridge reg grid for --mode lspinv (each reg -> its own checkpoint)")
    parser.add_argument("--modes", type=int, default=None, help="override modes; full-spectrum (=252 for 503 pts) required for the analytic map to be exact")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.modes is not None:
        config["model"]["modes"] = args.modes
    root = Path(config["data"]["root"] or args.data_root or str(Path.home() / "data/neural_operator_2"))
    input_fields = config["data"]["input_fields"]
    assert config["model"]["activation"] == "none", "LS init requires a linear FNO"
    assert config["model"]["num_blocks"] == 1, "LS init requires a single FNO block"

    if args.mode == "analytic":
        with (root / "dataset_manifest.json").open() as fh:
            manifest = json.load(fh)
        mass = float(manifest["model"]["mass_kg"])
        izz = float(manifest["model"]["Izz_kg_m2"])
        Bm = np.array(manifest["model"]["actuator_mapping_B"])
        A_true = np.zeros((2, 13))
        A_true[:, 0:4] = -Bm[:2, :]
        A_true[:, 10:13] = np.diag([mass, mass, izz])[:2, :]
        embed = np.zeros((2, 14), dtype=np.complex128)    # (out, in): in = 13 + grid
        embed[:, :13] = A_true                            # grid column stays zero
        n_modes = int(config["model"]["modes"])
        M = np.repeat(embed[None, ...], n_modes, axis=0)  # same map every bin
        assert n_modes * 2 >= 503, "analytic map is exact only at full spectrum"
        print("[analytic] A = [-B[:2], 0, 0, diag(m,m)] injected "
              f"(mass={mass}, Izz={izz})")
        model = build_model(config["model"], 13, 2)
        inject_ls_maps(model, M)
        evaluate_and_save(config, model, root, input_fields,
                          args.output_dir, "analytic", args.n_train)
    elif args.mode == "ls":
        assert int(config["model"]["modes"]) == 128, "ls fit is hardcoded to 128 modes"
        M = fit_ls_maps(root, input_fields, args.n_train, "")
        model = build_model(config["model"], 13, 2)
        inject_ls_maps(model, M)
        evaluate_and_save(config, model, root, input_fields,
                          args.output_dir, "ls", args.n_train)
    else:  # lspinv
        n_modes = int(config["model"]["modes"])
        assert n_modes == 252, "lspinv requires modes=252 (251 fitted bins + padded Nyquist)"
        for reg in args.reg:
            M, conds = fit_ridge_ls_maps(root, input_fields, args.n_train, reg)
            # Fitted map is (251, 2, 13); embed into the (252, 2, 14) network
            # map with the grid column zeroed (as in the analytic mode) and
            # carry the last fitted bin into the extra padded-basis Nyquist bin.
            M_net = np.zeros((n_modes, 2, 14), dtype=np.complex128)
            M_net[: M.shape[0], :, :13] = M
            M_net[M.shape[0] :] = M_net[M.shape[0] - 1]
            model = build_model(config["model"], 13, 2)
            inject_ls_maps(model, M_net)
            err_k, tgt_k = per_bin_test_error(root, input_fields, M, 200)
            per_bin_rl2 = np.sqrt(err_k / tgt_k.sum())
            top = np.argsort(per_bin_rl2)[-5:][::-1]
            print(f"[lspinv reg={reg:g}] per-bin test rl2 = "
                  f"{np.sqrt(err_k.sum() / tgt_k.sum()):.6e} | "
                  f"bins1-2 share {per_bin_rl2[1]:.3e}/{per_bin_rl2[2]:.3e} | "
                  f"top bin {top[0]} = {per_bin_rl2[top[0]]:.3e}")
            extra = {
                "reg": reg,
                "per_bin_test_rl2": float(np.sqrt(err_k.sum() / tgt_k.sum())),
                "condition_numbers": {
                    "bins_1_2": [float(conds[1]), float(conds[2])],
                    "median": float(np.median(conds)),
                    "max": float(conds.max()),
                    "min": float(conds.min()),
                },
                "top_error_bins": [[int(k), float(per_bin_rl2[k])] for k in top],
            }
            evaluate_and_save(config, model, root, input_fields,
                              args.output_dir / f"reg_{reg:g}", "lspinv",
                              args.n_train, extra)


if __name__ == "__main__":
    main()
