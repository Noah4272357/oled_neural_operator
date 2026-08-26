"""SVD structure of the m67h feature matrix and the truncated-solve floor.

Finding: Hessian (X^T X / n) has cond ~1e33 -- numerically rank-deficient
(min Gram eval -1.7e-13, only ~617 evals > 1e-6*max).  LBFGS grinds on the
~2600 flat directions.  Fix: SVD-precondition (whiten, NO centering) so the
Hessian becomes I on the row space -- invertible transform, optimum unchanged.

This script measures what the TRAINED floor would be for a given rcond clamp:
  (a) full pinv        -> reference 9.18e-5 (5000 raw, col-scaled)
  (b) truncated SVD at rcond 1e-4 / 1e-6 / 1e-8 -> clamp cost
  (c) whitened feats, lstsq -> same as (a) up to numerics (validates the
      preconditioning keeps the optimum)
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
print(f"[svd] X {tuple(X.shape)} R {tuple(R.shape)} f64 ({time.perf_counter()-t0:.0f}s)", flush=True)

t0 = time.perf_counter()
U, S, Vh = torch.linalg.svd(X, full_matrices=False)
print(f"[svd] SVD done ({time.perf_counter()-t0:.0f}s): s_max={S[0].item():.3e} "
      f"s_rank~{(S > 1e-12 * S[0]).sum().item()} above 1e-12*smax, "
      f"s_nz={(S > 0).sum().item()}", flush=True)
smax = S[0].item()
for thr in (1e-2, 1e-4, 1e-6, 1e-8):
    print(f"  s > {thr:.0e}*smax: {(S > thr * smax).sum().item()}", flush=True)

for rcond in (1e-4, 1e-6, 1e-8):
    k = int((S > rcond * smax).sum())
    # truncated-SVD least squares: W = V_k S_k^-1 (U_k^H R)
    UtR = torch.einsum("bi,bko->iko", U[:, :k].to(torch.complex128), R)
    Wt = torch.einsum("ik,koc->ioc",
                      (Vh[:k] / S[:k].unsqueeze(1)).conj().T.to(torch.complex128),
                      UtR)
    E_lin = E_d = 0.0
    for b in loaders["test"]:
        xx = b["input"].double(); rr = b["target"].double()
        z = torch.fft.rfft(xx, dim=1)
        hp = torch.fft.irfft(torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
                             n=N_POINTS, dim=1)
        dist = rr + hp
        rhat = torch.einsum("bi,koc->boc", feats(b) / sd, Wt.to(torch.complex128))
        E_lin += (torch.fft.rfft(rr, dim=1) - rhat).abs().square().sum()
        E_d += torch.fft.rfft(dist, dim=1).abs().square().sum()
    print(f"[svd] rcond {rcond:.0e}: k={k}  test rl2 = {(E_lin/E_d).sqrt().item():.6e}",
          flush=True)

# (c) whitened features (no centering): X_w = X @ V S^-1 = U (unit columns) --
#     the SVD-preconditioned problem; optimum must match the full pinv value
Ww = torch.linalg.lstsq(U.to(torch.complex128),
                        R.reshape(R.shape[0], -1)).solution.reshape(3232, 251, 2)
E_lin = E_d = 0.0
for b in loaders["test"]:
    xx = b["input"].double(); rr = b["target"].double()
    z = torch.fft.rfft(xx, dim=1)
    hp = torch.fft.irfft(torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
                         n=N_POINTS, dim=1)
    dist = rr + hp
    xw = (feats(b) / sd) @ Vh.conj().T / S.unsqueeze(0)
    rhat = torch.einsum("bi,koc->boc", xw.to(torch.complex128), Ww)
    E_lin += (torch.fft.rfft(rr, dim=1) - rhat).abs().square().sum()
    E_d += torch.fft.rfft(dist, dim=1).abs().square().sum()
print(f"[svd] (c) whitened-feats solve: test rl2 = {(E_lin/E_d).sqrt().item():.6e} "
      f"(should equal full-pinv value)", flush=True)
