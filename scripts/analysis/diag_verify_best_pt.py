"""Verify best.pt train loss on the EXACT training-script code path."""
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

def feats(b):
    z = torch.fft.rfft(b["input"], dim=1)[:, :F_IN + 1, :]
    return torch.cat([z.real, z.imag], dim=-1).reshape(len(b["input"]), -1)

def ft_targets(b):
    z = torch.fft.rfft(b["target"], dim=1)
    return torch.cat([z.real, z.imag], dim=-1).reshape(len(b["input"]), -1)

xs, ys = [], []
for b in loaders["train"]:
    xs.append(feats(b)); ys.append(ft_targets(b))
Xtr = torch.cat(xs); Ytr = torch.cat(ys)
mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-12)
Xs = (Xtr - mu) / sd
s = 1.0 / Ytr.std().clamp_min(1e-12)
Yt = Ytr * s

m = torch.nn.Linear(3232, 1004)
m.load_state_dict(torch.load("outputs/m67g_adamw_scaled/best.pt",
                             map_location="cpu", weights_only=True))
with torch.no_grad():
    m.eval()
    loss = torch.nn.functional.mse_loss(m(Xs), Yt)
    print(f"[best.pt] train mse (float32 path) = {loss.item():.6e}   (log ep80: 1.265e-01)")
    out = m(Xs)
    print(f"[best.pt] out range {out.min().item():.3f}..{out.max().item():.3f} "
          f"| Yt range {Yt.min().item():.3f}..{Yt.max().item():.3f}")
    print(f"[best.pt] |weight| {m.weight.norm().item():.4e} |bias| {m.bias.norm().item():.4e}")
    # per-unit residual energy: which output units does it fail on?
    res = (out - Yt).square().mean(0)   # (1004,) scaled residual energy per unit
    unit = res.reshape(N_BIN, 4).sum(-1)  # (251,) per bin
    order = unit.argsort(descending=True)
    print("[best.pt] worst bins (scaled res energy):")
    for k in order[:10].tolist():
        print(f"  bin {k:3d}: {unit[k].item():.4e}")
    print(f"[best.pt] bins 0-100 res {unit[:101].sum().item():.4e} | "
          f"bins 101-250 res {unit[101:].sum().item():.4e}")
    # same for the closed form (bias-free) for comparison
    Wc = torch.linalg.pinv(Xs.double()) @ Yt.double()
    p = Xs.double() @ Wc
    res_c = (p - Yt.double()).square().mean(0)
    unit_c = res_c.reshape(N_BIN, 4).sum(-1)
    print(f"[closed] bins 0-100 res {unit_c[:101].sum().item():.4e} | "
          f"bins 101-250 res {unit_c[101:].sum().item():.4e}")
    print(f"[closed] train mse = {res_c.mean().item():.6e}")
