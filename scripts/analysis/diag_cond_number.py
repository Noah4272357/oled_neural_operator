"""Condition number of the m67h Hessian H = X^T X / n (col-scaled raw feats).

Why: LBFGS grinds on the last ~15% of the way to the closed-form optimum
(1.055e-4 vs 9.18e-5 at ep880 of the f64 run).  If cond(H) is huge, only an
exact solve converges; if moderate (~1e4-1e6), a fresh quasi-Newton restart
with a longer history should break the stall.
"""
import sys
sys.path.insert(0, "/nishome/charliewang/forge-projects/oled-neural-operator")
import time
import torch

N_BIN, F_IN = 251, 100
DATA_CFG = {
    "root": "experiments/m63_residual_targets",
    "input_fields": ["force", "encoder_displacement"],
    "target_fields": ["disturbance"],
    "time_start": 0, "time_stop": None, "time_stride": 1,
    "preprocessing": {"name": "diff_features", "dt": 0.001, "channels": [4, 5, 6, 7]},
    "memory_cache": False, "batch_size": 128, "eval_batch_size": 128,
    "num_workers": 0,
}
from src.data.dataloader import build_dataloaders
loaders = build_dataloaders(DATA_CFG, seed=20260810)

def feats(b):
    z = torch.fft.rfft(b["input"], dim=1)[:, :F_IN + 1, :]
    return torch.cat([z.real, z.imag], dim=-1).reshape(len(b["input"]), -1).double()

t0 = time.perf_counter()
X = torch.cat([feats(b) for b in loaders["train"]])
sd = X.std(0).clamp_min(1e-12)
X = X / sd
print(f"[cond] X {tuple(X.shape)} col-scaled f64 ({time.perf_counter()-t0:.0f}s)", flush=True)

# eigen-decomposition of the 3232x3232 Gram matrix
t0 = time.perf_counter()
G = X.T @ X / X.shape[0]
evals = torch.linalg.eigvalsh(G)
print(f"[cond] Gram built + eigvalsh ({time.perf_counter()-t0:.0f}s)", flush=True)
lo, hi = evals[0].item(), evals[-1].item()
print(f"[cond] evals: min {lo:.3e}  max {hi:.3e}  cond(H) = {hi/max(lo,1e-30):.3e}")
# how many eigenvalues above a few thresholds (effective rank)
for thr in (1e-2 * hi, 1e-4 * hi, 1e-6 * hi):
    print(f"[cond] evals > {thr:.1e}: {(evals > thr).sum().item()} / {len(evals)}")
print(f"[cond] done ({time.perf_counter()-t0:.0f}s)")
