"""Resolve the contradiction: LS projection explains only 64% of r energy
on TRAIN, but LBFGS explains 99.5% on TEST.  One must be wrong.

Measure on TRAIN, side by side:
  (a) LS projection fit R^2 (full rank, k = 3232/3216)
  (b) LBFGS model fit R^2 (same weights as the 99.5% test model)
  (c) r / dist energy ratio on train (should match test 2.33e-6 if iid)
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

r = torch.fft.rfft(torch.cat([b["target"].double() for b in loaders["train"]]), dim=1)

# proper dist on train: target + head @ input
dsum = torch.zeros((), dtype=torch.float64)
rsum = torch.zeros((), dtype=torch.float64)
with torch.no_grad():
    for b in loaders["train"]:
        z = torch.fft.rfft(b["input"].double(), dim=1)
        hp = torch.fft.irfft(torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
                             n=N_POINTS, dim=1)
        dist = b["target"].double() + hp
        dsum += dist.square().sum()
        rsum += b["target"].double().square().sum()
print(f"[cmp] TRAIN ||r||^2/||dist||^2 = {(rsum/dsum).item():.6e} "
      f"(test was 2.33e-6)", flush=True)

# (a) LS projection, full-ish rank
U, S, Vh = torch.linalg.svd(X / sd, full_matrices=False)
for rcond in (1e-12, 1e-14):
    k = int((S > rcond * S[0]).sum())
    UtR = torch.einsum("bi,bko->iko", U[:, :k].to(torch.complex128), r)
    fit = torch.einsum("bk,koc->boc", U[:, :k].to(torch.complex128), UtR)
    r2tr = 1 - (r - fit).abs().square().sum().item() / r.abs().square().sum().item()
    print(f"[cmp] (a) LS proj k={k}: TRAIN R^2 vs r = {r2tr:.6f}", flush=True)

# (b) LBFGS model on train
W = torch.load("outputs/m67h_lbfgs_f64/best.pt", map_location="cpu",
               weights_only=True)["weight"].double()
num = den = 0.0
with torch.no_grad():
    for b in loaders["train"]:
        xx = b["input"].double(); rr = b["target"].double()
        z = torch.fft.rfft(xx, dim=1)
        hp = torch.fft.irfft(torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
                             n=N_POINTS, dim=1)
        out = (feats(b) / sd) @ W.t() / s
        v4 = out.reshape(len(b["input"]), N_BIN, 4)
        rhat = torch.fft.irfft(torch.complex(v4[..., :2], v4[..., 2:]),
                               n=N_POINTS, dim=1)
        num += ((hp + rhat) - (rr + hp)).square().sum()
        den += rr.square().sum()
print(f"[cmp] (b) LBFGS TRAIN: ||err||^2/||r||^2 = {(num/den).item():.6e} "
      f"-> R^2 vs r = {1 - (num/den).item():.6f}", flush=True)

# also total-dist rl2 on train for reference
den2 = 0.0
with torch.no_grad():
    for b in loaders["train"]:
        z = torch.fft.rfft(b["input"].double(), dim=1)
        hp = torch.fft.irfft(torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
                             n=N_POINTS, dim=1)
        den2 += (b["target"].double() + hp).square().sum()
print(f"[cmp] (b) LBFGS TRAIN rl2 vs dist = {(num/den2).sqrt().item():.6e}")
