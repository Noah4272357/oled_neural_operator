"""LBFGS trajectory (job 31057 best.pt) per-bin R^2 relative to r.

After the contraction-order artifact nullified diag_r_predictability's
R^2>=0.997, measure the TRAINED model (clean v4-reshape eval):
  - per-bin R^2 vs residual r (SSE/SST per bin, both channels merged)
  - total rl2 vs dist (reference)
  - energy ratio ||r||^2 / ||dist||^2 (test)
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

CKPT = "outputs/m67h_lbfgs_f64/best.pt"
W = torch.load(CKPT, map_location="cpu", weights_only=True)["weight"].double()  # (1004, 3232)

def feats(b):
    z = torch.fft.rfft(b["input"], dim=1)[:, :F_IN + 1, :]
    return torch.cat([z.real, z.imag], dim=-1).reshape(len(b["input"]), -1).double()

# recompute the train-side column scaling + global target scale (same trainer)
X = torch.cat([feats(b) for b in loaders["train"]])
sd = X.std(0).clamp_min(1e-12)
ys = []
for b in loaders["train"]:
    z = torch.fft.rfft(b["target"].double(), dim=1)
    ys.append(torch.cat([z.real, z.imag], dim=-1).reshape(len(b["input"]), -1))
Y = torch.cat(ys)
s = 1.0 / Y.std().clamp_min(1e-12)

sse = torch.zeros(N_BIN, 2, dtype=torch.float64)
sst = torch.zeros(N_BIN, 2, dtype=torch.float64)
num = den = er = 0.0
with torch.no_grad():
    for b in loaders["test"]:
        xx = b["input"].double(); rr = b["target"].double()
        z = torch.fft.rfft(xx, dim=1)
        hp = torch.fft.irfft(torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
                             n=N_POINTS, dim=1)
        dist = rr + hp
        out = (feats(b) / sd) @ W.t() / s
        v4 = out.reshape(len(b["input"]), N_BIN, 4)
        rhat = torch.fft.irfft(torch.complex(v4[..., :2], v4[..., 2:]), n=N_POINTS, dim=1)
        r = torch.fft.rfft(rr, dim=1)
        rf = torch.fft.rfft(rhat, dim=1)
        sse += (r - rf).abs().square().sum(0)
        sst += r.abs().square().sum(0)
        num += (torch.fft.rfft(dist, dim=1) - torch.fft.rfft(hp + rhat, dim=1)).abs().square().sum()
        den += torch.fft.rfft(dist, dim=1).abs().square().sum()
        er += r.abs().square().sum()

r2 = 1 - sse.sum(1) / sst.sum(1)
print(f"[lbfgs-r2] ckpt {CKPT}")
print(f"[lbfgs-r2] test rl2 (vs dist) = {(num/den).sqrt().item():.6e}")
print(f"[lbfgs-r2] ||r||^2/||dist||^2 = {(er/den).item():.6e}")
print(f"[lbfgs-r2] per-bin R^2 vs r: median={r2.median():.5f} min={r2.min():.5f} "
      f"max={r2.max():.5f} | bins<0.99: {(r2<0.99).sum().item()}/251")
print(f"[lbfgs-r2] bins 4-7: {[f'{v:.5f}' for v in r2[4:8].tolist()]}")
print(f"[lbfgs-r2] tail bins 8-100: median={r2[8:101].median():.5f} min={r2[8:101].min():.5f}")
