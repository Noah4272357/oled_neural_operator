"""ALIGNED re-test of the closed-form regularized family on TEST.

Previous closed-form numbers (1.52e-3 family, spectral/Tikhonov/bin grids)
were computed from scripts that re-iterated the shuffled loader across
different feature/target tensors -> row misalignment -> all numbers suspect
(proven by the projection: '3.03e-3' clean-aligned = 6.1e-2).

This script: single-pass row-aligned build per split, float64, SVD in
float64 (true numerical rank 2339, not the f32 3216 artifact), official
test rl2 vs dist, grid over:
  - truncated SVD  rcond in {1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12}
  - Tikhonov       lam  in {1e4, 1e3, 1e2, 1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4}
If any point lands at/below ~1.1e-4, the linear family closes the target
with a closed-form solution (zero training).
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

def build(split):
    xs, ys = [], []
    for b in loaders[split]:
        z_in = torch.fft.rfft(b["input"].double(), dim=1)[:, :F_IN + 1, :]
        xs.append(torch.cat([z_in.real, z_in.imag], dim=-1).reshape(len(b["input"]), -1))
        z_t = torch.fft.rfft(b["target"].double(), dim=1)
        ys.append(torch.cat([z_t.real, z_t.imag], dim=-1).reshape(len(b["input"]), -1))
    return torch.cat(xs), torch.cat(ys)

Xtr, Ytr = build("train")
Xte, Yte = build("test")
sd = Xtr.std(0).clamp_min(1e-12)
s = 1.0 / Ytr.std().clamp_min(1e-12)

def rebuild(F):
    v4 = F.reshape(-1, N_BIN, 4)
    return torch.fft.irfft(torch.complex(v4[..., :2], v4[..., 2:]), n=N_POINTS, dim=1)

def official_rl2(split, Yfit):
    num = den = 0.0
    lo = 0
    for b in loaders[split]:
        n = len(b["input"])
        rr = b["target"].double()
        z_in = torch.fft.rfft(b["input"].double(), dim=1)
        hp = torch.fft.irfft(torch.einsum("bki,koi->bko", z_in,
                                          head.to(torch.complex128)),
                             n=N_POINTS, dim=1)
        dist = rr + hp
        rhat = rebuild(Yfit[lo:lo + n])
        num += (hp + rhat - dist).square().sum().item()
        den += dist.square().sum().item()
        lo += n
    return (num / den) ** 0.5

Xt = Xtr / sd
U, S, Vh = torch.linalg.svd(Xt, full_matrices=False)
smax = S[0].item()
print(f"[grid] rank (1e-14) = {int((S > 1e-14 * smax).sum())} / {Xt.shape[1]}")

# --- truncated SVD: W = V_k S_k^-1 U_k^H (Y*s) ---
Ysc = Ytr * s
print(f"[grid] truncated SVD:")
for rcond in (1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
    k = int((S > rcond * smax).sum())
    W = Vh[:k].t() @ (S[:k].reciprocal().unsqueeze(1) * (U[:, :k].t() @ Ysc))
    rl2 = official_rl2("test", (Xte / sd) @ W / s)
    print(f"  rcond {rcond:.0e} (k={k:5d}): test rl2 = {rl2:.6e}", flush=True)

# --- Tikhonov: g = S/(S^2+lam), W = V diag(g) U^H (Y*s) ---
print(f"[grid] Tikhonov:")
for lam in (1e4, 1e3, 1e2, 1e1, 1.0, 1e-1, 1e-2, 1e-3, 1e-4):
    g = S / (S * S + lam)
    W = Vh.t() @ (g.unsqueeze(1) * (U.t() @ Ysc))
    rl2 = official_rl2("test", (Xte / sd) @ W / s)
    print(f"  lam {lam:.0e}: test rl2 = {rl2:.6e}", flush=True)
