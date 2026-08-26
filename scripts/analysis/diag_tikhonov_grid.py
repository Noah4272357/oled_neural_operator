"""Tikhonov (ridge / soft-truncation) grid for the linear residual head.

Hard truncation (diag_spectral_grid) failed: best rl2 = 1.545e-3 @ rcond 1e-2,
worse than the LBFGS trajectory (1.05e-4).  The LBFGS path is a SOFT
regularizer: it fits directions by gradient priority, never fully fitting
noise directions.  Tikhonov W(lam) = V diag(s/(s^2+lam)) U^H R is the
closed-form of ridge -- the standard soft regularizer, and its TRAINABLE
realization is plain weight decay (pure training, allowed).

Safe eval path (x-first contraction, proven 1e-12 accurate):
  rhat = (x V) @ diag(s/(s^2+lam)) @ (U^H R)
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

t0 = time.perf_counter()
X = torch.cat([feats(b) for b in loaders["train"]])
sd = X.std(0).clamp_min(1e-12)
X = X / sd
R = torch.cat([torch.fft.rfft(b["target"].double(), dim=1) for b in loaders["train"]])
U, S, Vh = torch.linalg.svd(X, full_matrices=False)
smax = S[0].item()
print(f"[tik] X {tuple(X.shape)} SVD done ({time.perf_counter()-t0:.0f}s) "
      f"s_min/smax = {S[-1].item()/smax:.2e}", flush=True)

def eval_tik(lam):
    g = (S / (S * S + lam))                     # gain per direction (3232,)
    Vg = Vh.conj().T * g.unsqueeze(0)          # (3232, 3232), safe column scale
    UtR = torch.einsum("bi,bko->iko", U.to(torch.complex128), R)
    num = den = 0.0
    with torch.no_grad():
        for b in loaders["test"]:
            xx = b["input"].double(); rr = b["target"].double()
            z = torch.fft.rfft(xx, dim=1)
            hp = torch.fft.irfft(
                torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
                n=N_POINTS, dim=1)
            dist = rr + hp
            xv = (feats(b) / sd) @ Vg                 # x-first, well-conditioned
            rhat = torch.einsum("bk,koc->boc", xv.to(torch.complex128), UtR)
            num += (torch.fft.rfft(rr, dim=1) - rhat).abs().square().sum()
            den += torch.fft.rfft(dist, dim=1).abs().square().sum()
    print(f"  lam {lam:.0e}: test rl2 = {(num/den).sqrt().item():.6e}", flush=True)

for lam in (1e2, 1e1, 1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10):
    eval_tik(lam)
