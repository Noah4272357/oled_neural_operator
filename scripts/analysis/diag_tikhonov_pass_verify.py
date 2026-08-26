"""ULTIMATE verification of the Tikhonov lambda=1e-3 closed-form PASS.

Zero alignment assumptions: for every test batch, build the prediction
INLINE from the same batch's features (X_b and rr are the same rows by
construction).  No pre-built Xte, no slicing, no reliance on loader order.

Also verify the flat-vs-time consistency INSIDE the batch: rfft(rhat) vs
the flat complex target must show the same error energy as the time path
(Parseval), so the result cannot be a rebuild artifact.

Outputs: test rl2 (official), per-batch flat/time error ratio, and the
same for LBFGS for comparison.
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
    z = torch.fft.rfft(b["input"].double(), dim=1)[:, :F_IN + 1, :]
    return torch.cat([z.real, z.imag], dim=-1).reshape(len(b["input"]), -1)

# ---- build training stats (single pass, only aggregate statistics) ----
Xtr_parts, Ytr_parts = [], []
for b in loaders["train"]:
    Xtr_parts.append(feats(b))
    z = torch.fft.rfft(b["target"].double(), dim=1)
    Ytr_parts.append(torch.cat([z.real, z.imag], dim=-1).reshape(len(b["input"]), -1))
Xtr = torch.cat(Xtr_parts)
Ytr = torch.cat(Ytr_parts)
sd = Xtr.std(0).clamp_min(1e-12)
s = 1.0 / Ytr.std().clamp_min(1e-12)

Xt = Xtr / sd
U, S, Vh = torch.linalg.svd(Xt, full_matrices=False)
Ysc = Ytr * s

def tikhonov_W(lam):
    g = S / (S * S + lam)
    return Vh.t() @ (g.unsqueeze(1) * (U.t() @ Ysc))

Wt = tikhonov_W(1e-3)
Wl = torch.load("outputs/m67h_lbfgs_f64/best.pt", map_location="cpu",
                weights_only=True)["weight"].double().t()

for name, W in (("tikhonov1e-3", Wt), ("lbfgs", Wl)):
    num = den = 0.0
    num_flat = den_flat = 0.0
    n_batches = 0
    for b in loaders["test"]:
        n = len(b["input"])
        rr = b["target"].double()
        z_in = torch.fft.rfft(b["input"].double(), dim=1)
        hp = torch.fft.irfft(torch.einsum("bki,koi->bko", z_in,
                                          head.to(torch.complex128)),
                             n=N_POINTS, dim=1)
        dist = rr + hp
        # prediction INLINE from this batch's features -- zero alignment risk
        out = (feats(b) / sd) @ W / s
        v4 = out.reshape(n, N_BIN, 4)
        rhat = torch.fft.irfft(torch.complex(v4[..., :2], v4[..., 2:]),
                               n=N_POINTS, dim=1)
        num += (hp + rhat - dist).square().sum().item()
        den += dist.square().sum().item()
        # flat-space error on the SAME batch (Parseval cross-check)
        z_t = torch.fft.rfft(rr, dim=1)
        Yb = torch.cat([z_t.real, z_t.imag], dim=-1).reshape(n, -1)
        num_flat += (out - Yb).square().sum().item()
        den_flat += Yb.square().sum().item()
        n_batches += 1
    rl2 = (num / den) ** 0.5
    print(f"[verify] {name}: test rl2 = {rl2:.6e} | batches={n_batches} | "
          f"flat/time ratio {num_flat/den_flat/(num/den):.4f} "
          f"(should be ~1.99 rfft factor)", flush=True)
