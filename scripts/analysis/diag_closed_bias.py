"""Closed-form with bias on the m67g parameterization (5000 samples, std
features): does the convex optimum with bias match m67g's 5.76e-4 (AdamW
converged) or the 1.25e-4 bias-free optimum (AdamW under-converged)?
Also: same on RAW features with bias (raw gives 9.18e-5 bias-free).
"""
import sys
sys.path.insert(0, "/nishome/charliewang/forge-projects/oled-neural-operator")
import torch
from src.data.dataloader import build_dataloaders

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
loaders = build_dataloaders(DATA_CFG, seed=20260810)
head = torch.load("outputs/lti_head_16ch.pt", map_location="cpu", weights_only=True)

def feats(b):
    z = torch.fft.rfft(b["input"], dim=1)[:, :F_IN + 1, :]
    return torch.cat([z.real, z.imag], dim=-1).reshape(len(b["input"]), -1)

Xtr, Rtr = [], []
for b in loaders["train"]:
    Xtr.append(feats(b).double())
    Rtr.append(torch.fft.rfft(b["target"].double(), dim=1))
X = torch.cat(Xtr); R = torch.cat(Rtr)

def run(name, std):
    Xf = X.clone()
    mu = sd = None
    if std:
        mu, sd = X.mean(0), X.std(0).clamp_min(1e-12)
        Xf = (X - mu) / sd
    Xbias = torch.cat([Xf, torch.ones(len(Xf), 1)], -1)      # (5000, 3233)
    W = torch.linalg.pinv(Xbias.to(torch.complex128)) @ R.permute(1, 0, 2)  # (3233, 251, 2)
    num = den = 0.0
    with torch.no_grad():
        for b in loaders["test"]:
            xx = b["input"].double(); rr = b["target"].double()
            z = torch.fft.rfft(xx, dim=1)
            hp = torch.fft.irfft(torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
                                 n=N_POINTS, dim=1)
            dist = rr + hp
            Xb = feats(b).double()
            if std:
                Xb = (Xb - mu) / sd
            Xbb = torch.cat([Xb, torch.ones(len(Xb), 1)], -1)
            rhat = torch.einsum("bi,kio->bko", Xbb.to(torch.complex128), W)
            Rf = torch.fft.rfft(rr, dim=1)
            num += (Rf - rhat).abs().square().sum().item()
            den += dist.square().sum().item()
    print(f"[{name}] closed-form WITH bias test rl2 = {(num/den)**0.5:.6e}")

run("5000 std+bias", True)
run("5000 raw+bias", False)
