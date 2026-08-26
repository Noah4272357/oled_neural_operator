"""Dataset-intrinsic Fourier analysis of the neural_operator_2 dataset.

Fourth optimization angle (complementing GPU performance / training dynamics /
code implementation): what the dataset itself says about the solvability of
the inverse disturbance operator at relative-L2 <= 1e-4.

Findings verified by this script (2026-08-23):
1. The operator (inputs) -> f_dist is *exactly linear*:
       f_dist = M q_ddot - B u          (rl2 ~ 1e-15, machine precision)
   since the dynamics are linear (M q_ddot = B u + f_dist), the encoder is an
   ideal linear measurement (s = H q, no noise), and the target is the pure
   disturbance.  Any nonlinearity in a surrogate only adds intermodulation
   error on top of a representable linear map.
2. The encoder displacement s is NOT bandlimited: the double-integrator
   start-up transient (K = C = 0, rest ICs) adds a secular linear drift to q
   (slope ~1.8e-3 m/s).  The drift's DFT decays as 1/omega and its energy
   across non-tone bins is ~5x the tone-bin energy of s.  The disturbance
   response in s (amplitude ~1e-7 m) is buried ~1e4x below the drift leakage.
3. Consequence: any *per-mode diagonal* operator on raw s (a linear FNO is
   exactly that in the Fourier basis) sees the disturbance at ~1e-4 SNR and
   its least-squares optimum shrinks toward the zero map -> rl2 ~ 1.0.
   This explains the observed "zero prediction" plateau of the 8ch problem.
4. Fix: differentiate the encoder signal twice (second numerical derivative).
   A ramp is annihilated exactly (in the interior by construction, at the
   edges by one-sided estimates exact on linear functions), tones stay tones
   at their own bins with a known per-bin scaling that the network absorbs.
   After differentiation the operator is again per-mode diagonal linear and
   the per-mode least-squares floor drops to ~1e-6 or below.
5. Per-mode least-squares linear maps (closed form, no training) give the
   achievable floor for a linear FNO on each problem definition.

Run:
    python scripts/analysis/analyze_dataset_fourier.py \
        --data-root ~/data/neural_operator_2 \
        --output-dir experiments/analysis/2026-08-23
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import h5py
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - headless guard
    plt = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# DFT bin spacing = 1 / window = 2 Hz; tone bins at 2*k Hz for k in TONE_BINS.
TONE_BINS = (1, 2, 4, 5, 6, 7)  # 2, 4, 8, 10, 12, 14 Hz
DISTURBANCE_BINS = (4, 5, 6, 7)  # 8, 10, 12, 14 Hz
BANDS_HZ = ((0.0, 10.0), (10.0, 18.0), (18.0, 250.0))
DT = 1e-3

# Validated categorical palette (dataviz reference palette, light mode).
C_BLUE = "#2a78d6"
C_ORANGE = "#eb6834"
C_AQUA = "#1baf7a"
C_MAGENTA = "#e87ba4"
C_INK = "#0b0b0b"
C_MUTED = "#52514e"
C_GRID = "#d9d7d2"


def load_sample(root: Path, split: str, index: int) -> Dict[str, np.ndarray]:
    with h5py.File(root / split / f"sample_{index:06d}.h5", "r") as f:
        return {
            "s": f["/sensors/encoder_displacement"][:],
            "u": f["/actuators/force"][:],
            "d": f["/disturbance/force"][:],
            "q": f["/states/q"][:],
            "qdot": f["/states/q_dot"][:],
            "qddot": f["/states/q_ddot"][:],
            "H": f["/sensors/measurement_matrix_H"][:],
        }


def rel_l2(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.sum((pred - target) ** 2) / np.sum(target**2)))


def second_derivative(x: np.ndarray, dt: float) -> np.ndarray:
    """Numerical 2nd derivative: ramp -> 0, tones -> same-bin tones.

    np.gradient with edge_order=2 uses central differences in the interior
    (exact per-bin scaling for discrete tones) and one-sided estimates at the
    edges (exact on linear functions, so the drift ramp is annihilated
    everywhere).
    """
    g1 = np.gradient(x, dt, axis=0, edge_order=2)
    return np.gradient(g1, dt, axis=0, edge_order=2)


def band_energy(series: np.ndarray, dt: float = DT) -> Dict[str, float]:
    """Per-band energy (sum |DFT|^2 / N, Parseval-consistent) of a signal."""
    dft = np.fft.rfft(series, axis=0)
    freqs = np.fft.rfftfreq(series.shape[0], dt)
    result: Dict[str, float] = {}
    for low, high in BANDS_HZ:
        mask = (freqs >= low) & (freqs < high)
        result[f"{low:g}-{high:g}Hz"] = float(np.sum(np.abs(dft[mask]) ** 2))
    return result


def leakage_ratio(series: np.ndarray, dt: float = DT) -> float:
    dft = np.fft.rfft(series, axis=0)
    freqs = np.fft.rfftfreq(series.shape[0], dt)
    tone = np.zeros_like(freqs, dtype=bool)
    for f_hz in (2.0, 4.0, 8.0, 10.0, 12.0, 14.0):
        tone |= np.isclose(freqs, f_hz, atol=0.75)
    e_tone = float(np.sum(np.abs(dft[tone]) ** 2))
    e_other = float(np.sum(np.abs(dft[~tone]) ** 2))
    return e_other / e_tone


def per_mode_ls_floor(
    split_train: str,
    split_test: str,
    root: Path,
    input_fields: Sequence[str],
    n_train: int = 200,
    n_test: int = 50,
    dt: float = DT,
    trim: int = 0,
    reg: float = 1e-10,
) -> float:
    """Least-squares per-mode linear map floor on test (train fit).

    ``trim`` removes that many samples from both ends before the DFT (used to
    discard numerical-differentiation boundary artifacts).
    """
    rng_train = list(range(0, 5000, 25))[:n_train]
    # stride-4 preserves the historical 50-sample eval subset; for n_test > 50
    # the stride-4 cap would silently truncate to 50 samples (review finding
    # 3: the floor holds either way -- 4.8304e-6 stride-4/50 vs 4.8638e-6
    # stride-1/200, same magnitude), so step down to stride-1.
    step = 4 if n_test <= 50 else 1
    rng_test = list(range(0, 200, step))[:n_test]
    probe = load_sample(root, split_train, 0)
    npts = probe["s"].shape[0] - 2 * trim
    nbin = npts // 2 + 1
    field_widths = {
        "s": probe["s"].shape[1],
        "sddot": probe["s"].shape[1],
        "u": probe["u"].shape[1],
        "q": probe["q"].shape[1],
        "qdot": probe["qdot"].shape[1],
        "qddot": probe["qddot"].shape[1],
    }
    width = 2 * sum(field_widths[f] for f in input_fields)
    gram = [np.zeros((width, width)) for _ in range(nbin)]
    rhs = [np.zeros((width, 4)) for _ in range(nbin)]

    def trimmed(sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if trim:
            return {k: v[trim:-trim] for k, v in sample.items()}
        return sample

    def features(sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        parts: Dict[str, np.ndarray] = {}
        for field in input_fields:
            if field == "s":
                parts["s"] = np.fft.rfft(sample["s"], axis=0)
            elif field == "sddot":
                parts["sddot"] = np.fft.rfft(
                    second_derivative(sample["s"], dt), axis=0
                )
            elif field == "u":
                parts["u"] = np.fft.rfft(sample["u"], axis=0)
            else:
                parts[field] = np.fft.rfft(sample[field], axis=0)
        parts["d"] = np.fft.rfft(sample["d"], axis=0)
        return parts

    # Train accumulation.
    for idx in rng_train:
        parts = features(trimmed(load_sample(root, split_train, idx)))
        cols = []
        for field in input_fields:
            cols.extend((parts[field].real, parts[field].imag))
        target = np.concatenate((parts["d"].real, parts["d"].imag), axis=1)
        for k in range(nbin):
            xk = np.concatenate([c[k] for c in cols])
            yk = target[k]
            gram[k] += np.outer(xk, xk)
            rhs[k] += np.outer(xk, yk)

    reg = reg * np.eye(width)
    num = den = 0.0
    for idx in rng_test:
        parts = features(trimmed(load_sample(root, split_test, idx)))
        cols = []
        for field in input_fields:
            cols.extend((parts[field].real, parts[field].imag))
        target = np.concatenate((parts["d"].real, parts["d"].imag), axis=1)
        for k in range(nbin):
            xk = np.concatenate([c[k] for c in cols])
            yk = target[k]
            coeff = np.linalg.solve(gram[k] + reg, rhs[k])
            pred = xk @ coeff
            num += float(np.sum((pred - yk) ** 2))
            den += float(np.sum(yk**2))
    return float(np.sqrt(num / den))


def make_spectrum_figure(
    series: np.ndarray,
    title: str,
    path: Path,
    freq_ticks: Sequence[float] = (2, 4, 8, 10, 12, 14),
) -> None:
    """Single-axis log-magnitude spectrum; tone bins marked, direct labels."""
    dft = np.fft.rfft(series, axis=0)
    freqs = np.fft.rfftfreq(series.shape[0], DT)
    mag = np.abs(dft).mean(axis=1)
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=130)
    ax.plot(freqs, mag, lw=1.2, color=C_BLUE)
    for f in freq_ticks:
        ax.axvline(f, color=C_GRID, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("|DFT| (mean over channels)")
    ax.set_title(title, color=C_INK, fontsize=10)
    ax.grid(True, which="both", color=C_GRID, lw=0.5)
    ax.tick_params(colors=C_INK)
    for spine in ax.spines.values():
        spine.set_color(C_GRID)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--reg", type=float, default=1e-10)
    args = parser.parse_args()

    root = args.data_root
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    with (root / "dataset_manifest.json").open() as f:
        manifest = json.load(f)
    mass = float(manifest["model"]["mass_kg"])
    izz = float(manifest["model"]["Izz_kg_m2"])
    M = np.diag([mass, mass, izz])
    B = np.array(manifest["model"]["actuator_mapping_B"])

    report: Dict[str, Any] = {"bands_hz": list(BANDS_HZ), "tone_bins": list(TONE_BINS)}

    # ---- 1. Linearity floor: f_dist = M q_ddot - B u ---------------------
    floors = {}
    for split, count in (("train", 20), ("val", 20), ("test", 20)):
        idxs = [min(i, {"train": 4999, "val": 5499, "test": 199}[split]) for i in range(0, count * 7, 7)]
        rls = []
        for i in idxs:
            s = load_sample(root, split, i)
            pred = (M @ s["qddot"].T).T - (B @ s["u"].T).T
            rls.append(rel_l2(pred[:, :2], s["d"]))
        floors[split] = max(rls)
    report["linearity_floor_rl2"] = floors

    # ---- 2. Drift & leakage ----------------------------------------------
    s0 = load_sample(root, "train", 0)
    drift = {}
    for j, name in enumerate(("x", "y", "theta_z")):
        slope, intercept = np.polyfit(
            np.linspace(0.0, 0.5, s0["q"].shape[0]), s0["q"][:, j], 1
        )
        drift[name] = {"slope": float(slope), "intercept": float(intercept)}
    report["drift_slope_train0"] = drift
    s_test = load_sample(root, "test", 0)
    report["leakage_ratio"] = {
        "s": leakage_ratio(s_test["s"]),
        "sddot": leakage_ratio(second_derivative(s_test["s"], DT)),
        "u": leakage_ratio(s_test["u"]),
        "d": leakage_ratio(s_test["d"]),
    }

    # ---- 3. Band energy decomposition ------------------------------------
    s = load_sample(root, "test", 0)
    report["band_energy_test0"] = {
        "encoder_displacement": band_energy(s["s"]),
        "encoder_sddot": band_energy(second_derivative(s["s"], DT)),
        "force": band_energy(s["u"]),
        "disturbance": band_energy(s["d"]),
    }

    # ---- 4. Analytic-inverse floor with differentiated encoder -----------
    sddot_floor = {}
    for split in ("train", "test"):
        idxs = [min(i, {"train": 4999, "test": 199}[split]) for i in range(0, 14 * 7, 7)]
        rls = []
        for i in idxs:
            s_ = load_sample(root, split, i)
            sddot = second_derivative(s_["s"], DT)
            Hp = np.linalg.pinv(s_["H"])
            pred = (M @ (Hp @ sddot.T)).T - (B @ s_["u"].T).T
            rls.append(rel_l2(pred[:, :2], s_["d"]))
        sddot_floor[split] = float(np.mean(rls))
    report["sddot_analytic_inverse_rl2"] = sddot_floor

    # ---- 5. Per-mode LS floors -------------------------------------------
    ls_floors = {
        "raw_8ch_s_u": per_mode_ls_floor("train", "test", root, ("s", "u")),
        "diff8ch_sddot_u": per_mode_ls_floor("train", "test", root, ("sddot", "u")),
        "diff8ch_sddot_u_trim2": per_mode_ls_floor(
            "train", "test", root, ("sddot", "u"), trim=2
        ),
        "diff8ch_sddot_u_trim4": per_mode_ls_floor(
            "train", "test", root, ("sddot", "u"), trim=4
        ),
        "13ch": per_mode_ls_floor("train", "test", root, ("q", "qdot", "qddot", "u")),
    }
    if args.reg != 1e-10:
        ls_floors = {
            f"{key}_reg{args.reg:g}": per_mode_ls_floor(
                "train", "test", root, fields, reg=args.reg
            )
            for key, fields in (
                ("diff8ch", ("sddot", "u")),
                ("13ch", ("q", "qdot", "qddot", "u")),
            )
        }
    report["per_mode_ls_floor_rl2"] = ls_floors

    # ---- 6. Figures -------------------------------------------------------
    if plt is not None:
        make_spectrum_figure(
            s_test["s"], "encoder displacement s — drift leakage dominates", out / "spectrum_s.png"
        )
        make_spectrum_figure(
            s_test["d"], "target disturbance d — clean tone bins", out / "spectrum_d.png"
        )
        make_spectrum_figure(
            second_derivative(s_test["s"], DT),
            "differentiated encoder d²s/dt² — drift annihilated, tones retained",
            out / "spectrum_sddot.png",
        )

    # ---- Save -------------------------------------------------------------
    with (out / "fourier_analysis.json").open("w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
