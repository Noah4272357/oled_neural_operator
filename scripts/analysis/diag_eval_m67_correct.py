"""Correct official-style evaluation of m67f / m67g best.pt.

The training scripts' eval rebuilt rhat via
  rhat_ft.reshape(B, N_BIN, 2, 2) -> complex(rhat_ft[..., 0], rhat_ft[..., 1])
but the flattened freq-target layout is per-bin [r0, r1, i0, i1], so
[..., 0] pulls r0/i0 interleaved and [..., 1] pulls r1/i1 -> real/imag
channel mixup in the reported test rl2.  Training loss itself is correct
(directly on the flattened vector).  Here the checkpoint is evaluated on
the exact same features/preprocessing, but rhat is rebuilt per-bin:
  v4 = out.reshape(B, N_BIN, 4);  rhat_c = complex(v4[..., :2], v4[..., 2:])
Official rl2 over dist energy on test.
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

def ft_targets(b):
    z = torch.fft.rfft(b["target"], dim=1)
    return torch.cat([z.real, z.imag], dim=-1).reshape(len(b["input"]), -1)

xs, ys = [], []
for b in loaders["train"]:
    xs.append(feats(b)); ys.append(ft_targets(b))
Xtr, Ytr = torch.cat(xs), torch.cat(ys)
mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-12)

def eval_ckpt(path, mode, name):
    m = torch.nn.Linear(3232, 1004)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    m.eval()
    if mode == "m67f":
        Ymu, Ysd = Ytr.mean(0), Ytr.std(0).clamp_min(1e-12)
    elif mode == "m67g":
        s = 1.0 / Ytr.std().clamp_min(1e-12)
    num = den = 0.0
    with torch.no_grad():
        for b in loaders["test"]:
            xx = b["input"].double(); rr = b["target"].double()
            z = torch.fft.rfft(xx, dim=1)
            hp = torch.fft.irfft(torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
                                 n=N_POINTS, dim=1)
            dist = rr + hp
            out = m((feats(b) - mu) / sd)
            if mode == "m67f":
                rft = out * Ysd + Ymu
            else:
                rft = out / s
            v4 = rft.reshape(len(b["input"]), N_BIN, 4)
            rhat = torch.fft.irfft(torch.complex(v4[..., :2].double(), v4[..., 2:].double()),
                                   n=N_POINTS, dim=1)
            num += (hp + rhat - dist).square().sum()
            den += dist.square().sum()
    rl2 = torch.sqrt(num / den).item()
    print(f"[{name}] CORRECT eval test rl2 = {rl2:.6e}  (was reported "
          f"{'1.453e-3' if mode=='m67f' else '1.438e-3'})  "
          f"{'PASS' if rl2 <= 1e-4 else 'FAIL'}")

eval_ckpt("outputs/m67f_adamw/best.pt", "m67f", "m67f AdamW+Ystd")
eval_ckpt("outputs/m67g_adamw_scaled/best.pt", "m67g", "m67g AdamW+scalar")
