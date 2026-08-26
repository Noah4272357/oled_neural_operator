"""Pinpoint why closed-form test rl2 differs between diag_r_predictability
(9.91e-5: 4000 train samples, RAW features) and my compare script (1.457e-3:
5000 train samples, standardized features).  Three variants, one script:
  (a) 4000 raw       -> should reproduce 9.91e-5 (diag's exact recipe)
  (b) 5000 raw       -> isolates the sample-count axis
  (c) 5000 std       -> isolates the standardization axis (matches m67g)
All evaluate identically: rhat = einsum(Xb, W) per-bin complex, rl2 over
the full dist energy (diag's per-bin evaluation, which avoids any
reshape-layout questions entirely).
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

def feats(b, mu=None, sd=None):
    z = torch.fft.rfft(b["input"], dim=1)[:, :F_IN + 1, :]
    x = torch.cat([z.real, z.imag], dim=-1).reshape(len(b["input"]), -1)
    if mu is not None:
        x = (x - mu) / sd
    return x

def run(name, n_train, std):
    Xtr, Rtr = [], []
    n = 0
    for b in loaders["train"]:
        Xtr.append(feats(b).double()); Rtr.append(torch.fft.rfft(b["target"].double(), dim=1))
        n += len(b["input"])
        if n >= n_train: break
    X = torch.cat(Xtr)[:n_train]; R = torch.cat(Rtr)[:n_train]
    mu = sd = None
    if std:
        mu, sd = X.mean(0), X.std(0).clamp_min(1e-12)
        X = (X - mu) / sd
    W = torch.linalg.pinv(X.to(torch.complex128)) @ R.permute(1, 0, 2)
    E_r = torch.zeros(N_BIN, dtype=torch.float64)
    E_lin = torch.zeros(N_BIN, dtype=torch.float64)
    E_d = torch.zeros(N_BIN, dtype=torch.float64)
    for b in loaders["test"]:
        xx = b["input"].double(); rr = b["target"].double()
        z = torch.fft.rfft(xx, dim=1)
        hp = torch.fft.irfft(torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
                             n=N_POINTS, dim=1)
        dist = rr + hp
        Rf = torch.fft.rfft(rr, dim=1); Df = torch.fft.rfft(dist, dim=1)
        E_r += Rf.abs().square().sum((0, 2)); E_d += Df.abs().square().sum((0, 2))
        rhat = torch.einsum("bi,kio->bko", feats(b, mu, sd).to(torch.complex128), W)
        E_lin += (Rf - rhat).abs().square().sum((0, 2))
    rl2 = (E_lin.sum().item() / E_d.sum().item()) ** 0.5
    print(f"[{name}] fit {n_train} std={std}: test rl2 = {rl2:.6e}")

run("a 4000 raw", 4000, False)
run("b 5000 raw", 5000, False)
run("c 5000 std", 5000, True)
