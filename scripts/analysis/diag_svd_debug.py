"""Debug the truncated-SVD eval discrepancy.

Claim to test: truncated fit rhat = X @ (V_k S_k^-1 U_k^H R) should equal
U_k U_k^H R on TRAIN, and on TEST should be the projection-ish reconstruction.
Measured rl2 for k=2042 was 647 -- impossible if the fit projects R.
Check: (1) train error of truncated fit vs ||(I-P_k)R||; (2) whether the
test eval uses the right Wt orientation; (3) same eval loop for the full
pinv solution (must reproduce 9.18e-5 from the reference script).
"""
import sys
sys.path.insert(0, "/nishome/charliewang/forge-projects/oled-neural-operator")
import time
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
X = X / sd
R = torch.cat([torch.fft.rfft(b["target"].double(), dim=1) for b in loaders["train"]])
U, S, Vh = torch.linalg.svd(X, full_matrices=False)

def rl2_test(W, name):
    num = den = 0.0
    with torch.no_grad():
        for b in loaders["test"]:
            xx = b["input"].double(); rr = b["target"].double()
            z = torch.fft.rfft(xx, dim=1)
            hp = torch.fft.irfft(torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
                                 n=N_POINTS, dim=1)
            dist = rr + hp
            rhat = torch.einsum("bi,koc->boc", feats(b) / sd, W.to(torch.complex128))
            num += (torch.fft.rfft(rr, dim=1) - rhat).abs().square().sum()
            den += torch.fft.rfft(dist, dim=1).abs().square().sum()
    print(f"  [{name}] test rl2 = {(num/den).sqrt().item():.6e}", flush=True)

# --- 1. reference: full pinv via torch (exact same eval loop) ---
Wp = torch.linalg.pinv(X.to(torch.complex128)) @ R.permute(1, 0, 2)
rl2_test(Wp.permute(1, 0, 2).reshape(3232, N_BIN, 2), "full pinv (reference path)")

# --- 2. truncated via my formula, per-part checks ---
for rcond, k in ((1e-4, 783), (1e-6, 2042)):
    UtR = torch.einsum("bi,bko->iko", U[:, :k].to(torch.complex128), R)
    A = (Vh[:k] / S[:k].unsqueeze(1)).conj().T.to(torch.complex128)
    Wt = torch.einsum("ik,koc->ioc", A, UtR)
    # train-side check: X Wt should equal U_k U_k^H R
    fit_train = torch.einsum("bi,koc->boc", X, Wt.to(torch.complex128))
    proj = torch.einsum("bk,koc->boc", U[:, :k].to(torch.complex128), UtR)
    err_fit = (fit_train - R).abs().square().sum().item()
    err_proj = (proj - R).abs().square().sum().item()
    print(f"[k={k}] train fit err {err_fit:.6e} vs projection err {err_proj:.6e} "
          f"| Wt shape {tuple(Wt.shape)}", flush=True)
    rl2_test(Wt, f"truncated k={k}")

# --- 3. whiten-space solution, explicit transform on both train and test ---
Ww = torch.linalg.lstsq(U.to(torch.complex128),
                        R.reshape(R.shape[0], -1)).solution.reshape(3232, N_BIN, 2)
num = den = 0.0
with torch.no_grad():
    for b in loaders["test"]:
        xx = b["input"].double(); rr = b["target"].double()
        z = torch.fft.rfft(xx, dim=1)
        hp = torch.fft.irfft(torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
                             n=N_POINTS, dim=1)
        dist = rr + hp
        xw = (feats(b) / sd) @ Vh.conj().T / S.unsqueeze(0)
        rhat = torch.einsum("bi,koc->boc", xw.to(torch.complex128), Ww)
        num += (torch.fft.rfft(rr, dim=1) - rhat).abs().square().sum()
        den += torch.fft.rfft(dist, dim=1).abs().square().sum()
print(f"  [whitened solve] test rl2 = {(num/den).sqrt().item():.6e}", flush=True)
print(f"[info] cond-ish: s_min/Smax = {S[-1].item()/S[0].item():.3e}", flush=True)
