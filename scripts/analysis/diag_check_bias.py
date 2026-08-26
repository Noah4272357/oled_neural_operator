"""Check m67g best.pt bias weights; re-evaluate with bias zeroed vs kept."""
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

xs = []
for b in loaders["train"]:
    xs.append(feats(b))
Xtr = torch.cat(xs)
mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-12)

st = torch.load("outputs/m67g_adamw_scaled/best.pt", map_location="cpu", weights_only=True)
m = torch.nn.Linear(3232, 1004)
m.load_state_dict(st); m.eval()
with torch.no_grad():
    w_n = m.weight.norm().item(); b_n = m.bias.norm().item()
print(f"[m67g best] |weight| = {w_n:.4e} |bias| = {b_n:.4e} "
      f"bias/w = {b_n/max(w_n,1e-30):.3f}")

def eval_model(bias_zero, name):
    with torch.no_grad():
        if bias_zero:
            m.bias.zero_()
        num = den = 0.0
        for b in loaders["test"]:
            xx = b["input"].double(); rr = b["target"].double()
            z = torch.fft.rfft(xx, dim=1)
            hp = torch.fft.irfft(torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
                                 n=N_POINTS, dim=1)
            dist = rr + hp
            out = m((feats(b) - mu) / sd)
            s = 1.0 / Xtr_dummy_std  # placeholder, replaced below
            num += 0.0; den += 0.0
    print(f"[{name}] (placeholder)")
    return None

Xtr_ft = []
for b in loaders["train"]:
    z = torch.fft.rfft(b["target"], dim=1)
    Xtr_ft.append(torch.cat([z.real, z.imag], dim=-1).reshape(len(b["input"]), -1))
Ytr = torch.cat(Xtr_ft)
s = 1.0 / Ytr.std().clamp_min(1e-12)
print(f"[m67g] target scale s = {s:.4e}")

def eval_bias(bias_zero):
    with torch.no_grad():
        if bias_zero:
            m.bias.zero_()
        num = den = 0.0
        for b in loaders["test"]:
            xx = b["input"].double(); rr = b["target"].double()
            z = torch.fft.rfft(xx, dim=1)
            hp = torch.fft.irfft(torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
                                 n=N_POINTS, dim=1)
            dist = rr + hp
            out = m((feats(b) - mu) / sd) / s
            v4 = out.reshape(len(b["input"]), N_BIN, 4)
            rhat = torch.fft.irfft(torch.complex(v4[..., :2].double(), v4[..., 2:].double()),
                                   n=N_POINTS, dim=1)
            num += (hp + rhat - dist).square().sum()
            den += dist.square().sum()
        print(f"[m67g best] bias {'zeroed' if bias_zero else 'kept'}: "
              f"test rl2 = {torch.sqrt(num/den).item():.6e}")

eval_bias(False)
eval_bias(True)
