"""True closed-form family of the full-spectrum linear residual head.

Discovery 2026-08-26: the 9.18e-5 ceiling (diag_r_predictability /
pinpoint) was a CONTRACTION-ORDER ARTIFACT: evaluating the pinv solution
as einsum("bi,kio->bko", x, W) with W = V S^-1 U^H R loses catastrophic
precision (measured 2.75e6 vs 3.03e-3 for the safe path (x V S^-1) @ (U^H R)).
The true LS solution is TEST-BAD (3.03e-3): it overfits the high-gain
directions (the disturbance tones ride input directions with s ~ 1e-4..1e-6
of s_max; the map gains 1/s up to 1e8).

Here we sweep the SPECTRALLY-TRUNCATED LS family (rcond = spectral
regularization, ridge-analogue) with the SAFE eval path, to find the
best achievable test rl2 of the linear model family.

Safe eval:  rhat[b,k,o] = sum_i (x V S^-1)[b,i] * (U^H R)[i,k,o]
             -- contract x with V S^-1 FIRST (both are well-conditioned
                products; the equivalent W-then-x order cancels ~1e8 terms).
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
print(f"[grid] X {tuple(X.shape)} SVD done ({time.perf_counter()-t0:.0f}s) "
      f"s_min/smax = {S[-1].item()/smax:.2e}", flush=True)

# precompute per-rcond truncated U^H R and eval in one pass over test
def eval_trunc(rcond):
    k = int((S > rcond * smax).sum())
    if k == 0:
        print(f"  rcond {rcond:.0e}: k=0 (zero model) -- skip", flush=True)
        return
    UtR = torch.einsum("bi,bko->iko", U[:, :k].to(torch.complex128), R)
    VSinv = Vh.conj().T[:, :k] / S[:k].unsqueeze(0)  # truncated whitening
    num = den = 0.0
    with torch.no_grad():
        for b in loaders["test"]:
            xx = b["input"].double(); rr = b["target"].double()
            z = torch.fft.rfft(xx, dim=1)
            hp = torch.fft.irfft(
                torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
                n=N_POINTS, dim=1)
            dist = rr + hp
            xw = (feats(b) / sd) @ VSinv
            rhat = torch.einsum("bk,koc->boc", xw.to(torch.complex128), UtR)
            num += (torch.fft.rfft(rr, dim=1) - rhat).abs().square().sum()
            den += torch.fft.rfft(dist, dim=1).abs().square().sum()
    print(f"  rcond {rcond:.0e}: k={k:5d}  test rl2 = {(num/den).sqrt().item():.6e}",
          flush=True)

for rc in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 1e-12):
    eval_trunc(rc)
