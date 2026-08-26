"""FACT-CHECK the time-domain rebuild path R (reshape -> complex -> irfft).

Claim under test: for ANY flat vector F, ||irfft(rebuild(F)) - rr||^2 MUST
equal ||F - Y||^2 / 501 (Parseval + orthonormal reshape).  The measured
time-domain error was 334x SMALLER than the flat-space error for the LBFGS
fit and 9x LARGER for the projection -- impossible if R is orthonormal.
Find where the energy goes.
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
rr_all = torch.cat([b["target"].double() for b in loaders["train"]])

W = torch.load("outputs/m67h_lbfgs_f64/best.pt", map_location="cpu",
               weights_only=True)["weight"].double()
Yfit = ((X / sd) @ W.t()) / s

def rebuild(F):
    v4 = F.reshape(-1, N_BIN, 4)
    return torch.fft.irfft(torch.complex(v4[..., :2], v4[..., 2:]),
                           n=N_POINTS, dim=1)

# 1. R o f == id ?  (rebuild the TRUE flat target, compare to time target)
rhat_true = rebuild(Y)
print(f"[rc] 1. R(f(rr)) vs rr: max|diff| = {(rhat_true - rr_all).abs().max().item():.3e}")
print(f"[rc]    ||R(f(rr)) - rr||^2 = {(rhat_true - rr_all).square().sum().item():.3e}")

# 2. energy through rebuild: ||R(F)||^2 vs ||F||^2/501 for both fits and Y
for name, F in [("Y_true", Y), ("Yfit_lbfgs", Yfit)]:
    e_in = F.square().sum().item()
    e_out = rebuild(F).square().sum().item()
    print(f"[rc] 2. {name}: ||F||^2={e_in:.6e} ||R(F)||^2={e_out:.6e} "
          f"x501={e_out*501:.6e} ratio={(e_out*501)/e_in:.6f}")

# 3. Parseval violation check: time err vs flat err/501
for name, F, T in [("lbfgs", Yfit, rr_all)]:
    num_t = (rebuild(F) - T).square().sum().item()
    num_f = (F - Y).square().sum().item()
    print(f"[rc] 3. {name}: time err = {num_t:.6e} | flat err/501 = {num_f/501:.6e} "
          f"| ratio = {num_t/(num_f/501):.6f}")

# 4. THE POINT: is the flat Yfit REALLY the Parseval image of the time rhat?
#    If R o f = id (check 1) and R orthonormal (check 2), this MUST be 1.
#    -> if not, the batching/order differs between the two paths.
