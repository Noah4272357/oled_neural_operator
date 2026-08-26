"""Projection property test: ||r - P r|| must be the MINIMUM over col(X).

LBFGS claims 0.997 train R^2 vs LS projection 0.638 -- mathematically
impossible if both are honest linear maps on the same feature space.
Check directly:  ||r - X W_lbfgs||^2  vs  ||r - P r||^2  vs  ||P(XW) - XW||.
"""
import sys
sys.path.insert(0, "/nishome/charliewang/forge-projects/oled-neural-operator")
import torch

N_POINTS, N_BIN, F_IN = 501, 251, 100
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
head = torch.load("outputs/lti_head_16ch.pt", map_location="cpu", weights_only=True)

def feats(b):
    z = torch.fft.rfft(b["input"], dim=1)[:, :F_IN + 1, :]
    return torch.cat([z.real, z.imag], dim=-1).reshape(len(b["input"]), -1).double()

X = torch.cat([feats(b) for b in loaders["train"]])
sd = X.std(0).clamp_min(1e-12)
ys = []
for b in loaders["train"]:
    z = torch.fft.rfft(b["target"].double(), dim=1)
    ys.append(torch.cat([z.real, z.imag], dim=-1).reshape(len(b["input"]), -1))
Y = torch.cat(ys)
s = 1.0 / Y.std().clamp_min(1e-12)

# --- real-space equivalent: solve in REAL domain for one output column ---
# LBFGS maps X -> Y (real flat 1004).  Take output column 0 only.
w = torch.load("outputs/m67h_lbfgs_f64/best.pt", map_location="cpu",
               weights_only=True)["weight"].double()[0]   # (3232,) row 0
y0 = (Y * s)[:, 0]                       # scaled target column 0
pred0 = (X / sd) @ w                    # LBFGS prediction, scaled space
# project pred0 onto col(X/sd):
U, S, Vh = torch.linalg.svd(X / sd, full_matrices=False)
k = int((S > 1e-14 * S[0]).sum())
P = U[:, :k] @ U[:, :k].t()
pproj = P @ pred0
print(f"[proj] rank = {k} / {X.shape[1]}")
print(f"[proj] ||pred0 - P pred0|| / ||pred0|| = "
      f"{((pred0 - pproj).square().sum() / pred0.square().sum()).sqrt().item():.3e}")
# LS fit for column 0:
w_ls = torch.linalg.lstsq(X / sd, y0).solution
pred_ls = (X / sd) @ w_ls
print(f"[proj] ||y0 - pred_lbfgs||^2 = {((y0 - pred0).square().sum()).item():.6e}")
print(f"[proj] ||y0 - pred_ls||^2    = {((y0 - pred_ls).square().sum()).item():.6e}")
print(f"[proj] ||y0||^2              = {y0.square().sum().item():.6e}")
print(f"[proj] ||y0 - P y0||^2       = {((y0 - P @ y0).square().sum()).item():.6e}")
