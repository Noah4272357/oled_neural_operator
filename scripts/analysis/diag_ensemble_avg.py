"""Ensemble average of existing LBFGS trajectories (early-stopped points).

Trajectories: 31057 (no restart, best @ep890) and 31059 (restart 50,
best @ep3960).  If the test error at the early-stopped point is dominated
by uncorrelated noise-direction weights, the average can beat both.

Clean eval: exact trainer evaluate() replica (v4 reshape, /s, /sd).
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

ckpts = ["outputs/m67h_lbfgs_f64/best.pt",       # 31057 @ep890
         "outputs/m67h_lbfgs_f64_r50/best.pt"]   # 31059 @ep3960
Ws = [torch.load(p, map_location="cpu", weights_only=True)["weight"].double()
      for p in ckpts]

def eval_w(W, name):
    num = den = 0.0
    with torch.no_grad():
        for b in loaders["test"]:
            xx = b["input"].double(); rr = b["target"].double()
            z = torch.fft.rfft(xx, dim=1)
            hp = torch.fft.irfft(torch.einsum("bki,koi->bko", z,
                                              head.to(torch.complex128)),
                                 n=N_POINTS, dim=1)
            dist = rr + hp
            out = (feats(b) / sd) @ W.t() / s
            v4 = out.reshape(len(b["input"]), N_BIN, 4)
            rhat = torch.fft.irfft(torch.complex(v4[..., :2], v4[..., 2:]),
                                   n=N_POINTS, dim=1)
            num += ((hp + rhat) - dist).square().sum()
            den += dist.square().sum()
    print(f"[ens] {name}: test rl2 = {(num/den).sqrt().item():.6e}", flush=True)
    return (num / den).sqrt().item()

for W, p in zip(Ws, ckpts):
    eval_w(W, p)
Wavg = sum(Ws) / len(Ws)
eval_w(Wavg, "avg(31057, 31059)")
# weighted average: weight each by 1/best_rl2^2 (inverse-variance-ish)
b = [1.055e-4, 1.104e-4]
w = torch.tensor([1.0 / x**2 for x in b])
Ww = sum(wi * Wi for wi, Wi in zip(w / w.sum(), Ws))
eval_w(Ww, "wavg(31057, 31059)")
