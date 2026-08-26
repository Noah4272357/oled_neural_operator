"""Post-training closed-form joint-LTI refinement (M5-3).

Given a trained FNO1d checkpoint, compute the training-set residual
r = d - model(x), fit a joint frequency-shared complex LTI map on the
UNPADDED 501-point rfft basis (bins 3..250, float64, ridge reg grid), and
wrap the trained model in ``RefinedFNO1d`` so that

    M_final(x) = model(x, grid) + irfft(M_shared @ rfft(x))

No gradient steps are taken (F4: Adam micro-tuning destroys the exact
map): the refine head is a closed-form ridge least-squares solution
(reusing the M5-1 joint-fit protocol of ``ls_init_fno.fit_ridge_ls_maps``,
with the target replaced by the trained model's residual).

Checkpoint protocol (evaluate.py): ``build_model(config["model"])`` +
strict ``load_state_dict`` -- no class-name deserialization.  The combined
checkpoint is therefore saved with ``config.model.name = "fno1d_refined"``
(the wrapper class registered in ``src/models/factory.py``) and the LTI map
in the persistent buffer ``lti_map``, so the UNMODIFIED official
``scripts/evaluate.sh`` can load and evaluate it.

Per-split diagnostics accumulate, in one forward pass, the residual target
energies (tgt_k), the target-disturbance energies (dft_k), the per-bin
cross Gram A_k = sum_i x_ik conj(r_ik)^T and the input Gram Bc[k] (also
the condition report); the after-correction per-bin energy is then exact
for any candidate shared map M:

    err_k = tgt_k - 2 Re(tr(M A_k)) + tr(M Bc[k] M^H)

M5-3 correction (2026-08-24): the M5-1 joint-fit protocol (real-stacked
4-column design, reconstruction ``coeff[0::2,:2].T - 1j*coeff[1::2,:2].T``)
is only valid for COMPLEX-CONSISTENT targets (the exact linear inverse
operator d = M q_ddot - B u).  For the trained model's NONLINEAR residual
the 4 columns [Re r1, Re r2, Im r1, Im r2] are independent projections;
the reconstruction rotates the real-part coefficients, and on the
ill-conditioned joint design (cond ~2e9, ~15 near-null directions whose
action is data-undetermined) the rotated map's output magnitude scales as
sigma_max * ||coeff|| -- the real parts fit to ~77% explained variance
while the reconstructed complex map EXPLODES (|Mx|^2 ~ 1e10 vs |r|^2 ~ 7).
The fit is therefore solved here as the COMPLEX-CONSTRAINED least-squares
problem (numpy lstsq handles complex natively: minimizing the squared
complex norm is exactly the constrained objective), which has no such
instability: max|M| ~ 7e2 was an artifact of the unconstrained columns,
not a signal.

Run:
    python scripts/analysis/refine_lspinv.py \
        --checkpoint outputs/job_30929/m52_p2a_hop1/best_model.pt \
        --data-root ~/data/neural_operator_3 \
        --output-dir outputs/job_30929/m53_refine \
        --reg 1e-12 1e-10 1e-8
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import FIELD_PATHS
from src.models.factory import build_model
from src.training.factories import build_optimizer, build_scheduler
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.config import load_config

from ls_init_fno import _field_width

N_BIN = 251  # 501-point unpadded rfft
FIT_BINS = (3, 250)  # joint fit domain; correction applied ONLY here


@torch.no_grad()
def _evaluate_split(model, root, split, indices, input_fields, batch=64):
    """Time-domain relative_l2 on the wrapper (device-aware evaluate_split)."""
    dev = next(model.parameters()).device
    num = 0.0
    den = 0.0
    for b0 in range(0, len(indices), batch):
        idxs = indices[b0 : b0 + batch]
        xs, gs, ys = [], [], []
        for i in idxs:
            path = root / split / f"sample_{i:06d}.h5"
            with h5py.File(path, "r") as fh:
                parts = [np.asarray(fh[FIELD_PATHS[f]], dtype=np.float32) for f in input_fields]
                xs.append(torch.as_tensor(np.concatenate(parts, axis=1), dtype=torch.float32))
                ys.append(torch.as_tensor(np.asarray(fh[FIELD_PATHS["disturbance"]], dtype=np.float32), dtype=torch.float32))
                gs.append(torch.as_tensor(np.asarray(fh[FIELD_PATHS["time"]], dtype=np.float32), dtype=torch.float32))
        pred = model(torch.stack(xs).to(dev), torch.stack(gs).to(dev)).float().cpu()
        y = torch.stack(ys)
        num += float(((pred - y) ** 2).sum())
        den += float((y**2).sum())
    return float(np.sqrt(num / den))


@torch.no_grad()
def _accumulate_split(
    root: Path,
    split: str,
    input_fields: List[str],
    model: torch.nn.Module,
    device: torch.device,
    indices: range,
    want_fit: bool,
    batch: int = 64,
) -> Dict[str, Any]:
    """One pass over ``indices``: residuals r = d - model(x) + LTI Gram terms.

    Returns Xc/Yc (complex fit rows over joint bins 3..250) when
    ``want_fit``, plus per-bin Bc (input Gram, cond report), A (cross
    Gram), tgt (residual energy) and dft (target disturbance energy).
    """
    n_in = sum(_field_width(root, f) for f in input_fields)
    n_out = 2
    joint_bins = list(range(3, N_BIN))
    Xc = np.empty((len(indices) * len(joint_bins), n_in), dtype=np.complex128) if want_fit else None
    Yc = np.empty((len(indices) * len(joint_bins), n_out), dtype=np.complex128) if want_fit else None
    Bc = np.zeros((N_BIN, n_in, n_in), dtype=np.complex128)
    A = np.zeros((N_BIN, n_in, n_out), dtype=np.complex128)
    tgt = np.zeros(N_BIN)
    dft = np.zeros(N_BIN)
    row = 0
    t0 = time.time()
    for b0 in range(0, len(indices), batch):
        idxs = indices[b0 : b0 + batch]
        xs, gs, ds = [], [], []
        for i in idxs:
            path = root / split / f"sample_{i:06d}.h5"
            with h5py.File(path, "r") as fh:
                parts = [np.asarray(fh[FIELD_PATHS[f]], dtype=np.float32) for f in input_fields]
                xs.append(np.concatenate(parts, axis=1))
                gs.append(np.asarray(fh[FIELD_PATHS["time"]], dtype=np.float32))
                ds.append(np.asarray(fh[FIELD_PATHS["disturbance"]], dtype=np.float32))
        xb = torch.as_tensor(np.stack(xs), dtype=torch.float32, device=device)
        gb = torch.as_tensor(np.stack(gs), dtype=torch.float32, device=device)
        db = torch.as_tensor(np.stack(ds), dtype=torch.float32, device=device)
        pred = model(xb, gb).float()
        r = (db.double() - pred.double()).cpu().numpy()   # (b, 501, 2) f64
        x64 = np.stack(xs).astype(np.float64)
        d64 = np.stack(ds).astype(np.float64)
        for i in range(xb.shape[0]):
            xft = np.fft.rfft(x64[i], axis=0)             # (251, 13)
            rft = np.fft.rfft(r[i], axis=0)               # (251, 2)
            dft_i = np.fft.rfft(d64[i], axis=0)
            Bc += xft[:, :, None] * np.conj(xft[:, None, :])
            A += xft[:, :, None] * np.conj(rft[:, None, :])
            tgt += (np.abs(rft) ** 2).sum(axis=1)
            dft += (np.abs(dft_i) ** 2).sum(axis=1)
            if want_fit:
                for k in joint_bins:
                    Xc[row] = xft[k]
                    Yc[row] = rft[k]
                    row += 1
        if (b0 // batch + 1) % 25 == 0:
            print(f"  [{split}] [{b0 + len(idxs)}/{len(indices)}] {(time.time()-t0):.1f}s")
    return {
        "Xc": Xc, "Yc": Yc, "Bc": Bc, "A": A,
        "tgt": tgt, "dft": dft, "n_in": n_in, "n_out": n_out,
    }


def solve_shared(Xc: np.ndarray, Yc: np.ndarray, reg: float) -> np.ndarray:
    """Complex-constrained ridge-LS joint fit -> shared (2, 13) map.

    Solves min ||Yc - Xc M^T||_F over complex M (numpy lstsq minimizes
    the squared complex norm -- the objective that enforces the
    complex-consistency a real-valued map must satisfy).  The M5-1
    real-stacked protocol is NOT reused here: for the nonlinear residual
    its independent [Re r1, Re r2, Im r1, Im r2] projections are not
    complex-consistent and the reconstructed map explodes (see module
    docstring).  ``reg`` is ridge relative to the joint system's largest
    eigenvalue, as in the M5-1 protocol.
    """
    lmax = np.linalg.norm(Xc) ** 2
    Xa = np.vstack([Xc, np.sqrt(reg / lmax) * np.eye(Xc.shape[1])])
    Ya = np.vstack([Yc, np.zeros((Xc.shape[1], Yc.shape[1]))])
    coeff, *_ = np.linalg.lstsq(Xa, Ya, rcond=1e-12)
    return coeff.T  # (2, 13) complex


def after_error(M: np.ndarray, acc: Dict[str, Any]) -> np.ndarray:
    """Per-bin residual energy after applying per-bin maps M (251,2,13).

    For any candidate per-bin map this is exact given the accumulated
    Gram terms (A_k = sum_i x_ik conj(r_ik)^T, Bc[k] = sum_i x_ik x_ik^H):
        err_k = tgt_k - 2 Re(tr(M_k A_k)) + tr(M_k Bc[k] M_k^H)
    """
    A, Bc, tgt = acc["A"], acc["Bc"], acc["tgt"]
    tr = np.einsum("koi,kio->k", M, A)
    quad = np.einsum("koi,kij,koj->k", M, Bc, np.conj(M))
    return tgt - 2 * np.real(tr) + quad


def build_refined_wrapper(
    trained_state: Dict[str, Any], combined_config: Dict[str, Any],
    M_bins: np.ndarray, device: torch.device,
) -> torch.nn.Module:
    """FNO1d state_dict -> RefinedFNO1d with the LTI map in its buffer.

    M_bins is the per-bin map (251, 2, 13) with rows 0..2 already zeroed:
    the correction is applied ONLY at the joint fit domain (bins 3..250).
    The trained residual is NOT pointwise-linear, so the frequency-shared
    map is only claimed where it was fit (evaluating it at the ramp bins
    0-2 exploded the error by orders of magnitude -- M5-3 finding, see
    m5_3_refine.md).
    """
    wrapper = build_model(combined_config["model"], 13, 2).to(device)
    wrapper.load_state_dict(trained_state, strict=False)  # base keys only
    wrapper.lti_map.copy_(torch.as_tensor(np.ascontiguousarray(M_bins), dtype=torch.cdouble))
    return wrapper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "m53_refine")
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--reg", type=float, nargs="+", default=[1e-12, 1e-10, 1e-8])
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("highest")  # residual forwards in full fp32
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else args.device if args.device != "auto" else "cpu")
    print(f"[refine] device={device}")

    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    trained_config = ckpt.get("config") or load_config(
        PROJECT_ROOT / "configs" / "config.yaml")
    root = Path(args.data_root or trained_config["data"]["root"])
    input_fields = list(trained_config["data"]["input_fields"])
    n_in = sum(_field_width(root, f) for f in input_fields)
    assert n_in == 13, f"refine protocol is 13ch, got {n_in}ch"

    model = build_model(trained_config["model"], 13, 2).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[refine] trained checkpoint epoch={ckpt['epoch']} "
          f"config.model={trained_config['model']}")

    # One residual pass over the full train split (fit rows + Gram terms).
    train_acc = _accumulate_split(root, "train", input_fields, model, device,
                                  range(0, args.n_train), want_fit=True)
    # One residual pass over val/test for the before/after decomposition.
    val_acc = _accumulate_split(root, "val", input_fields, model, device,
                                range(0, 500), want_fit=False)
    test_acc = _accumulate_split(root, "test", input_fields, model, device,
                                 range(0, 200), want_fit=False)

    def rl2(acc, err):
        # Frequency-domain rl2 in the rfft convention (bins 0..250, DC single
        # counted).  Differs from the official TIME-domain metric by a few
        # percent because of the DC-bin convention and Parseval half-counting
        # -- the gate is always the official evaluate.sh time-domain rl2.
        return float(np.sqrt(np.real(err).sum() / acc["dft"].sum()))

    m0 = {s: rl2(a, a["tgt"]) for s, a in
          (("train", train_acc), ("val", val_acc), ("test", test_acc))}
    print(f"[refine] M0 residual rl2 (freq-domain, full split): "
          f"train {m0['train']:.6e} | val {m0['val']:.6e} | test {m0['test']:.6e}")

    official_m0 = {}
    eval_json = args.checkpoint.parent / "official_test_eval.json"
    if eval_json.is_file():
        official_m0["test"] = float(
            json.loads(eval_json.read_text())["metrics"]["relative_l2"])
        print(f"[refine] official M0 test rl2 (time-domain) = {official_m0['test']:.6e}")

    out_root = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for reg in args.reg:
        print(f"\n[refine] reg={reg:g}: solving joint ridge LS ...")
        t0 = time.time()
        M_shared = solve_shared(train_acc["Xc"], train_acc["Yc"], reg)
        print(f"  solved in {(time.time()-t0):.1f}s | max|M| {np.abs(M_shared).max():.3e}")
        # The deployed correction zeroes bins 0..2; the decomposition must
        # describe the deployed model, so evaluate after_error with M_bins
        # (per-bin map: the shared map replicated at bins 3..250, zeroed at
        # 0..2 -- the deployed buffer content).
        M_bins = np.repeat(M_shared[None, ...], N_BIN, axis=0)  # (251, 2, 13)
        M_bins[: FIT_BINS[0]] = 0.0

        refined = {}
        for s, a in (("train", train_acc), ("val", val_acc), ("test", test_acc)):
            err = after_error(M_bins, a)
            refined[s] = rl2(a, err)
            if s == "train":
                err_k = err
        conds = np.empty(N_BIN)
        for k in range(N_BIN):
            sv = np.linalg.svd(train_acc["Bc"][k], compute_uv=False)
            conds[k] = sv[0] / sv[-1] if sv[-1] > 0 else float("inf")
        per_bin_after = np.sqrt(np.maximum(np.real(err_k / train_acc["dft"].sum()), 0.0))
        per_bin_before = np.sqrt(train_acc["tgt"] / train_acc["dft"].sum())
        top = np.argsort(per_bin_after)[-5:][::-1]
        print(f"[refine] reg={reg:g} refined rl2 (freq-domain): "
              f"train {refined['train']:.6e} | val {refined['val']:.6e} | "
              f"test {refined['test']:.6e}")
        print(f"[refine] train-in recovery (freq-domain): {m0['train']:.6e} -> "
              f"{refined['train']:.6e} ({(m0['train']-refined['train'])/m0['train']*100:.2f}%)")
        print(f"[refine] per-bin before top-5: "
              + ", ".join(f"bin{k}={per_bin_before[k]:.1e}" for k in np.argsort(per_bin_before)[-5:][::-1]))
        print(f"[refine] per-bin after  top-5: "
              + ", ".join(f"bin{k}={per_bin_after[k]:.1e}" for k in top))

        # Combined model: wrapper + checkpoint (config name fno1d_refined).
        combined_config = copy.deepcopy(trained_config)
        combined_config["model"] = copy.deepcopy(dict(trained_config["model"]))
        combined_config["model"]["name"] = "fno1d_refined"
        combined_config["model"]["refine_n_bin"] = N_BIN
        wrapper = build_refined_wrapper(ckpt["model_state_dict"], combined_config, M_bins, device)

        out_dir = out_root / f"reg_{reg:g}"
        out_dir.mkdir(parents=True, exist_ok=True)
        optimizer = build_optimizer(combined_config["optimizer"], wrapper)
        scheduler = build_scheduler(
            combined_config["scheduler"], optimizer,
            int(combined_config["training"]["epochs"]))
        save_checkpoint(
            out_dir / "best_model.pt",
            model=wrapper,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=int(ckpt["epoch"]),
            validation={"train_relative_l2": refined["train"],
                        "val_relative_l2": refined["val"],
                        "test_relative_l2": refined["test"]},
            best_metric=refined["test"],
            config=combined_config,
            legacy_arguments={},
        )
        print(f"[refine] checkpoint -> {out_dir / 'best_model.pt'}")

        # Independent time-domain cross-check on the wrapper (first 200 each).
        self_eval = {}
        for split, n in (("train", 200), ("val", 200), ("test", 200)):
            rl = _evaluate_split(wrapper, root, split, range(0, n), input_fields)
            self_eval[f"{split}_200"] = rl
        print(f"[refine] self-eval time-domain (first 200 each): {self_eval}")

        report = {
            "reg": reg,
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": int(ckpt["epoch"]),
            "domain": "rfft_frequency_domain",
            "fit_bins": f"{FIT_BINS[0]}..{FIT_BINS[1]}",
            "correction_bins": f"{FIT_BINS[0]}..{FIT_BINS[1]}",
            "m0_freq_rl2": {s: m0[s] for s in ("train", "val", "test")},
            "refined_freq_rl2": {s: refined[s] for s in ("train", "val", "test")},
            "refine_share_freq_domain": (m0["test"] - refined["test"]) / m0["test"],
            "official_m0_test_rl2": official_m0.get("test"),
            "self_eval_time_domain_first200": self_eval,
            "train_in_recovery_freq_rl2": refined["train"],
            "per_bin_before_rl2": [float(v) for v in per_bin_before],
            "per_bin_after_rl2": [float(v) for v in per_bin_after],
            "top_error_bins": [[int(k), float(per_bin_after[k])] for k in top],
            "max_abs_M": float(np.abs(M_shared).max()),
            "condition_numbers": {
                "bins_1_2": [float(conds[1]), float(conds[2])],
                "median": float(np.median(conds)),
                "max": float(conds.max()),
                "min": float(conds.min()),
            },
            "n_train_fit": args.n_train,
            "fit_basis": "unpadded_501_rfft",
            "joint_bins": "3..250",
            "grid_column_excluded": True,
        }
        with (out_dir / "refine_report.json").open("w") as fh:
            json.dump(report, fh, indent=2)
        summary[str(reg)] = report
        print(f"[refine] report -> {out_dir / 'refine_report.json'}")

    with (out_root / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n[refine] done.")


if __name__ == "__main__":
    main()
