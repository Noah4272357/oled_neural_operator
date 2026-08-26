"""P2 同种子对照：预滚只平移 DC 偏移，斜坡/波形/音调能量一概不变（T4 决定性证据复现）。

沉淀自 /tmp/review_p2_sameseeds.py（评审独立复现脚本）+ p2_notes.md 方法：
- u（执行器力）新旧位级一致（maxdiff ~4e-14）→ 对照严格受控；
- q_new = q_old + 仿射（slope*t + intercept），残差 ~1e-9（机器精度，信号量级 1e-3）；
  斜率差 ~4e-9 为 substeps 5 vs 1 的积分数值精度差异（非"≡ 常数"字面含义，评审 F2）；
- 漂移斜率位级一致（x 通道 0.00177890631 → 0.00177891002）→ 斜坡是双积分器稳态固有性质，
  预滚只平移积分起点；
- 泄漏比（analyzer 同款公式 e_other/e_tone）不降反升，DFT 分解证明 e_other 增长 100% 来自
  DC bin（预滚引入的常数偏移计入非音调能量）→ 预滚不能消除泄漏，raw 8ch 路线出局；
- preroll 功能性检查：新档 qdot[0] != 0（预滚后非静止初值），旧档全零。

用法（代理侧任意 Python 解释器，需 h5py + numpy）：
  python experiments/analysis/2026-08-24/p2_same_seed_compare.py \
      [--old-root ~/data/neural_operator_2] [--new-root ~/data/neural_operator_3_pre] \
      [--split train] [--first-index 0] [--count 10] [--json <path>]

预期（2026-08-24 记录值，train0）：u maxdiff 3.9e-14；q 残差 1.6e-9；
斜率 0.00177890631 → 0.00177891002；leakage s 4.941 → 42.294；e_tone 相同（相对差 ~9e-6）；
e_other 剔除 DC 后相对差 ~1e-6、DC 增长占比 100.00%。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

DT = 1e-3
TONE_FREQS_HZ = (2.0, 4.0, 8.0, 10.0, 12.0, 14.0)  # 与 analyze_dataset_fourier.py 一致

_FIELD_PATHS = (
    ("s", "/sensors/encoder_displacement"),
    ("u", "/actuators/force"),
    ("d", "/disturbance/force"),
    ("q", "/states/q"),
    ("qdot", "/states/q_dot"),
    ("qddot", "/states/q_ddot"),
)


def load_sample(root: Path, split: str, idx: int):
    with h5py.File(root / split / f"sample_{idx:06d}.h5", "r") as f:
        return {k: f[p][:] for k, p in _FIELD_PATHS + (("t", "/time"),)}


def leakage_ratio(series: np.ndarray, dt: float = DT) -> float:
    """与 analyze_dataset_fourier.py::leakage_ratio 同款定义。"""
    dft = np.fft.rfft(series, axis=0)
    freqs = np.fft.rfftfreq(series.shape[0], dt)
    tone = np.zeros_like(freqs, dtype=bool)
    for f_hz in TONE_FREQS_HZ:
        tone |= np.isclose(freqs, f_hz, atol=0.75)
    e_tone = float(np.sum(np.abs(dft[tone]) ** 2))
    e_other = float(np.sum(np.abs(dft[~tone]) ** 2))
    return e_other / e_tone


def dft_decomp(series: np.ndarray, dt: float = DT):
    """e_tone / e_other / DC bin / e_other 剔除 DC，与 p2_notes 的 DFT 分解同法。"""
    dft = np.fft.rfft(series, axis=0)
    freqs = np.fft.rfftfreq(series.shape[0], dt)
    tone = np.zeros_like(freqs, dtype=bool)
    for f_hz in TONE_FREQS_HZ:
        tone |= np.isclose(freqs, f_hz, atol=0.75)
    e_tone = float(np.sum(np.abs(dft[tone]) ** 2))
    e_other = float(np.sum(np.abs(dft[~tone]) ** 2))
    dc = float(np.sum(np.abs(dft[0]) ** 2))  # 与 e_tone/e_other 同为全通道能量
    return {"e_tone": e_tone, "e_other": e_other, "dc": dc, "e_other_no_dc": e_other - dc}


def compare_sample(old_root: Path, new_root: Path, split: str, idx: int) -> dict:
    a = load_sample(old_root, split, idx)
    b = load_sample(new_root, split, idx)
    t = a["t"]
    q_names = ("x", "y", "theta_z")

    affine = []
    for j, nm in enumerate(q_names):
        dq = b["q"][:, j] - a["q"][:, j]
        sl, ic = np.polyfit(t, dq, 1)
        res = dq - (sl * t + ic)
        affine.append(
            {
                "channel": nm,
                "slope": float(sl),
                "intercept": float(ic),
                "resid_rms": float(np.sqrt(np.mean(res ** 2))),
                "max_abs_res": float(np.max(np.abs(res))),
            }
        )

    sl_old, _ = np.polyfit(t, a["q"][:, 0], 1)
    sl_new, _ = np.polyfit(t, b["q"][:, 0], 1)

    lr_old, lr_new = leakage_ratio(a["s"]), leakage_ratio(b["s"])
    dec_old, dec_new = dft_decomp(a["s"]), dft_decomp(b["s"])
    de_other = dec_new["e_other"] - dec_old["e_other"]
    d_dc = dec_new["dc"] - dec_old["dc"]

    return {
        "index": idx,
        "u_max_abs_diff": float(np.max(np.abs(a["u"] - b["u"]))),
        "q_affine": affine,
        "drift_slope_qx": {"old": float(sl_old), "new": float(sl_new)},
        "preroll_functional": {
            "old_qdot0": a["qdot"][0].tolist(),
            "new_qdot0": b["qdot"][0].tolist(),
            "old_q0": a["q"][0].tolist(),
            "new_q0": b["q"][0].tolist(),
        },
        "leakage_ratio_s": {"old": lr_old, "new": lr_new},
        "dft_decomp_s": {
            "old": dec_old,
            "new": dec_new,
            "e_tone_rel_diff": abs(dec_new["e_tone"] - dec_old["e_tone"]) / dec_old["e_tone"]
            if dec_old["e_tone"]
            else float("nan"),
            "e_other_no_dc_rel_diff": abs(dec_new["e_other_no_dc"] - dec_old["e_other_no_dc"])
            / dec_old["e_other_no_dc"]
            if dec_old["e_other_no_dc"]
            else float("nan"),
            "dc_growth_share": (d_dc / de_other) * 100.0 if de_other else float("nan"),
        },
    }


def fmt(r: dict) -> str:
    q = ", ".join(
        f"q_{a['channel']}: slope={a['slope']:.2e} intc={a['intercept']:.2e} "
        f"resid_rms={a['resid_rms']:.2e} max|res|={a['max_abs_res']:.2e}"
        for a in r["q_affine"]
    )
    d = r["dft_decomp_s"]
    return (
        f"== {r['index']} ==\n"
        f"u max|old-new| = {r['u_max_abs_diff']:.3e}\n"
        f"q affine (new-old): {q}\n"
        f"drift slope qx: old={r['drift_slope_qx']['old']:.12f} "
        f"new={r['drift_slope_qx']['new']:.12f}\n"
        f"preroll: old qdot[0]={r['preroll_functional']['old_qdot0']} "
        f"new qdot[0]={r['preroll_functional']['new_qdot0']}\n"
        f"leakage s: old={r['leakage_ratio_s']['old']:.4f} "
        f"new={r['leakage_ratio_s']['new']:.4f}\n"
        f"DFT s: old e_tone={d['old']['e_tone']:.4f} e_other={d['old']['e_other']:.4f} "
        f"(DC={d['old']['dc']:.4f}, no-DC={d['old']['e_other_no_dc']:.4f})\n"
        f"       new e_tone={d['new']['e_tone']:.4f} e_other={d['new']['e_other']:.4f} "
        f"(DC={d['new']['dc']:.4f}, no-DC={d['new']['e_other_no_dc']:.4f})\n"
        f"e_tone rel diff={d['e_tone_rel_diff']:.3e}  "
        f"e_other no-DC rel diff={d['e_other_no_dc_rel_diff']:.3e}  "
        f"DC growth share={d['dc_growth_share']:.2f}%"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old-root", default=str(Path.home() / "data/neural_operator_2"))
    ap.add_argument("--new-root", default=str(Path.home() / "data/neural_operator_3_pre"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--first-index", type=int, default=0)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--json", default=None, help="optional JSON output path")
    args = ap.parse_args()

    old_root, new_root = Path(args.old_root), Path(args.new_root)
    results = []
    for i in range(args.first_index, args.first_index + args.count):
        r = compare_sample(old_root, new_root, args.split, i)
        results.append(r)
        print(fmt(r))
    if args.json:
        Path(args.json).write_text(
            json.dumps(results, indent=1, sort_keys=True, ensure_ascii=False)
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
