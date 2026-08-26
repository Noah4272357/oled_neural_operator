"""FINAL aligned evaluation of the LS projection vs LBFGS on TEST.

Single-pass, row-aligned construction for BOTH splits (no re-iterating
loaders: the loader shuffles differently per iteration -> row misalignment
was the source of every prior 'contradiction').

Projection fit (full rank k=3216) computed cleanly via LS weights from the
aligned train pair, evaluated through the SAME time-domain rebuild as the
trainer evaluate(), against the official test rl2 (vs dist).
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
    xs, ys, rrs = [], [], []
    for b in loaders[split]:
        z_in = torch.fft.rfft(b["input"].double(), dim=1)[:, :F_IN + 1, :]
        xs.append(torch.cat([z_in.real, z_in.imag], dim=-1).reshape(len(b["input"]), -1))
        z_t = torch.fft.rfft(b["target"].double(), dim=1)
        ys.append(torch.cat([z_t.real, z_t.imag], dim=-1).reshape(len(b["input"]), -1))
        rrs.append(b["target"].double())
    return torch.cat(xs), torch.cat(ys), torch.cat(rrs)

Xtr, Ytr, _ = build("train")
Xte, Yte, _ = build("test")
sd = Xtr.std(0).clamp_min(1e-12)
s = 1.0 / Ytr.std().clamp_min(1e-12)

def rebuild(F):
    v4 = F.reshape(-1, N_BIN, 4)
    return torch.fft.irfft(torch.complex(v4[..., :2], v4[..., 2:]), n=N_POINTS, dim=1)

def official_rl2(split, Yfit):
    """Official metric, row-aligned: iterate split ONCE, Yfit rows match
    batch order because build() iterated the same loader once."""
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

# --- LS weights from the aligned train pair (full-rank projection) ---
Xt = Xtr / sd
Uf, Sf, Vhf = torch.linalg.svd(Xt, full_matrices=False)
k = int((Sf > 1e-14 * Sf[0]).sum())
print(f"[final] rank k = {k} / {Xt.shape[1]}")
Wproj = Vhf[:k].t() @ (Sf[:k].reciprocal().unsqueeze(1) * (Uf[:, :k].t() @ (Ytr * s)))
Yfit_proj_tr = (Xtr / sd) @ Wproj / s
print(f"[final] TRAIN proj R^2 vs r = "
      f"{1 - (Yfit_proj_tr - Ytr).square().sum().item() / Ytr.square().sum().item():.6f}")
Yfit_proj_te = (Xte / sd) @ Wproj / s
print(f"[final] TEST  proj rl2 vs dist = {official_rl2('test', Yfit_proj_te):.6e}")

# --- LBFGS for reference, same aligned path ---
W = torch.load("outputs/m67h_lbfgs_f64/best.pt", map_location="cpu",
               weights_only=True)["weight"].double()
Yfit_lb_te = (Xte / sd) @ W.t() / s
print(f"[final] TEST  LBFGS rl2 vs dist = {official_rl2('test', Yfit_lb_te):.6e}")
