"""Error structure of the best trajectory (31057 @ep890):

1. test per-bin error energy distribution (||e_k||^2 / ||dist||^2) -- where
   does the remaining 5% live?
2. train-side R^2 of the LS solution (clean path) -- if ~1.0, there is no
   representation limit: the gap is statistical/generalization, and
   ensembles/nonlinear inductive bias can help.  If < 1.0, the 8ch feature
   set cannot express r (info limit) and nothing helps.
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

W = torch.load("outputs/m67h_lbfgs_f64/best.pt", map_location="cpu",
               weights_only=True)["weight"].double()

# --- 1. train-side R^2 of the LS solution (CLEANEST path: U_k @ (U_k^H R),
#      the orthogonal projection -- no intermediate contraction) ---
U, S, Vh = torch.linalg.svd(X / sd, full_matrices=False)
smax = S[0].item()
r = torch.fft.rfft(torch.cat([b["target"].double() for b in loaders["train"]]), dim=1)
for rcond in (1e-2, 1e-4, 1e-6, 1e-8, 1e-12):
    k = int((S > rcond * smax).sum())
    UtR = torch.einsum("bi,bko->iko", U[:, :k].to(torch.complex128), r)
    fit = torch.einsum("bk,koc->boc", U[:, :k].to(torch.complex128), UtR)
    r2tr = 1 - (r - fit).abs().square().sum().item() / r.abs().square().sum().item()
    print(f"[err] LS (rcond {rcond:.0e}, k={k}) TRAIN R^2 vs r = {r2tr:.6f}", flush=True)

# --- 2. test per-bin error energy of the trajectory best ---
ebin = torch.zeros(N_BIN, 2, dtype=torch.float64)
den = torch.zeros(N_BIN, 2, dtype=torch.float64)
with torch.no_grad():
    for b in loaders["test"]:
        xx = b["input"].double(); rr = b["target"].double()
        z = torch.fft.rfft(xx, dim=1)
        hp = torch.fft.irfft(torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
                             n=N_POINTS, dim=1)
        dist = rr + hp
        out = (feats(b) / sd) @ W.t() / s
        v4 = out.reshape(len(b["input"]), N_BIN, 4)
        rhat = torch.fft.irfft(torch.complex(v4[..., :2], v4[..., 2:]),
                               n=N_POINTS, dim=1)
        e = torch.fft.rfft(hp + rhat - dist, dim=1)
        d = torch.fft.rfft(dist, dim=1)
        ebin += e.abs().square().sum(0)
        den += d.abs().square().sum(0)
ebin = ebin.sum(1)
den = den.sum(1)
frac = ebin / den.sum()
print(f"[err] test error energy by bin (fraction of total dist energy):")
print(f"[err]   bins 0-3:   {frac[:4].sum().item():.3e}")
print(f"[err]   bins 4-7:   {frac[4:8].sum().item():.3e}")
print(f"[err]   bins 8-100: {frac[8:101].sum().item():.3e}")
print(f"[err]   bins 101+:  {frac[101:].sum().item():.3e}")
top = frac.argsort(descending=True)[:8]
print(f"[err]   top bins: {[(i.item(), f'{frac[i].item():.3e}') for i in top]}")
