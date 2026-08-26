"""Debug: reproduce train_m67h evaluate() exactly vs the manual matmul.

Job 31057 log says best 1.0545e-4 @ep890; manual W eval says 6.44e-4.
Find the discrepancy by diffing intermediate tensors on one test batch.
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
sd_ = torch.load(CKPT, map_location="cpu", weights_only=True)
print("state_dict keys:", list(sd_.keys()), "| w shape:", tuple(sd_["weight"].shape),
      "| dtype:", sd_["weight"].dtype)

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
print(f"sd: {tuple(sd.shape)} | s = {s:.6e}")

W = sd_["weight"].double()
model = torch.nn.Linear(X.shape[1], Y.shape[1], bias=False, dtype=torch.float64)
model.weight.data.copy_(W)

b = next(iter(loaders["test"]))
xx = b["input"].double()
z = torch.fft.rfft(xx, dim=1)
hp = torch.fft.irfft(torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
                     n=N_POINTS, dim=1)
fx = feats(b) / sd
out_manual = (fx @ W.t()) / s
out_model = model(fx) / s
print(f"out manual {out_manual.shape} vs model: max diff = "
      f"{(out_manual - out_model).abs().max().item():.3e}")

v4 = out_model.reshape(len(b["input"]), N_BIN, 4)
rhat = torch.fft.irfft(torch.complex(v4[..., :2], v4[..., 2:]), n=N_POINTS, dim=1)
pred = hp + rhat
dist = b["target"].double() + hp
rl2 = ((pred - dist).square().sum() / dist.square().sum()).sqrt()
print(f"[repro] one batch rl2 = {rl2.item():.6e}")

# full test set, exact trainer loop
num = den = 0.0
with torch.no_grad():
    for b in loaders["test"]:
        xx = b["input"].double(); rr = b["target"].double()
        z = torch.fft.rfft(xx, dim=1)
        hp = torch.fft.irfft(torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
                             n=N_POINTS, dim=1)
        dist = rr + hp
        out = model(feats(b) / sd) / s
        v4 = out.reshape(len(b["input"]), N_BIN, 4)
        rhat = torch.fft.irfft(torch.complex(v4[..., :2], v4[..., 2:]), n=N_POINTS, dim=1)
        num += ((hp + rhat) - dist).square().sum()
        den += dist.square().sum()
print(f"[repro] full test rl2 = {(num / den).sqrt().item():.6e}")
