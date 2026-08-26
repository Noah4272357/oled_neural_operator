"""Correct per-bin evaluation of m67d (per-pair whiten) and m67e (global
whiten) best.pt checkpoints, rebuilding each script's exact preprocessing."""
import sys
sys.path.insert(0, "/nishome/charliewang/forge-projects/oled-neural-operator")
import torch
from src.data.dataloader import build_dataloaders

N_POINTS, N_BIN, F_IN, N_CH = 501, 251, 100, 16
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

def feats_raw(b):
    z = torch.fft.rfft(b["input"], dim=1)[:, :F_IN + 1, :]
    return torch.cat([z.real, z.imag], dim=-1).reshape(len(b["input"]), -1)

def ft_targets(b):
    z = torch.fft.rfft(b["target"], dim=1)
    return torch.cat([z.real, z.imag], dim=-1).reshape(len(b["input"]), -1)

def xft(b):
    return torch.fft.rfft(b["input"], dim=1).to(torch.complex128)

# --- build preps ---
Xr = torch.cat([feats_raw(b) for b in loaders["train"]])
mu, sd = Xr.mean(0), Xr.std(0).clamp_min(1e-12)

# m67d per-pair whiten
G = torch.zeros((N_CH, N_CH), dtype=torch.complex128)
n_pairs = 0
for b in loaders["train"]:
    z = xft(b)
    G += torch.einsum("btc,btd->cd", z.conj(), z)
    n_pairs += z.shape[0] * z.shape[1]
evals, evecs = torch.linalg.eigh(G / n_pairs)
evals = evals.clamp_min(1e-10)
Wd = (evecs * (1.0 / evals.sqrt())) @ evecs.conj().T

def feats_d(b):
    z = (xft(b) @ Wd)[:, :F_IN + 1, :]
    return torch.cat([z.real, z.imag], dim=-1).reshape(len(b["input"]), -1).float()

Xd = torch.cat([feats_d(b) for b in loaders["train"]])
mu_d, sd_d = Xd.mean(0), Xd.std(0).clamp_min(1e-12)

# m67e global whiten
n = Xr.shape[0]
Xs = (Xr.double() - mu.double()) / sd.double()
C = Xs.T @ Xs / n
ev, evec = torch.linalg.eigh(C)
ev = ev.clamp_min(ev.max().item() * 1e-10)
We = (evec * (1.0 / ev.sqrt())) @ evec.T
def feats_e(b):
    return (((feats_raw(b).double() - mu.double()) / sd.double()) @ We)

Ytr = torch.cat([ft_targets(b) for b in loaders["train"]])
Ymu, Ysd = Ytr.mean(0), Ytr.std(0).clamp_min(1e-12)

def eval_ckpt(path, prep, name):
    m = torch.nn.Linear(3232, 1004)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    m.eval()
    num = den = 0.0
    with torch.no_grad():
        for b in loaders["test"]:
            xx = b["input"].double(); rr = b["target"].double()
            z = torch.fft.rfft(xx, dim=1)
            hp = torch.fft.irfft(torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
                                 n=N_POINTS, dim=1)
            dist = rr + hp
            out = m(prep(b).float())
            rft = out * Ysd + Ymu
            v4 = rft.reshape(len(b["input"]), N_BIN, 4)
            rhat = torch.fft.irfft(torch.complex(v4[..., :2].double(), v4[..., 2:].double()),
                                   n=N_POINTS, dim=1)
            num += (hp + rhat - dist).square().sum()
            den += dist.square().sum()
    print(f"[{name}] CORRECT eval test rl2 = {torch.sqrt(torch.tensor(num/den)).item():.6e}")

eval_ckpt("outputs/m67d_linear_sgd/best.pt",
          lambda b: (feats_d(b) - mu_d) / sd_d, "m67d per-pair whiten SGD")
eval_ckpt("outputs/m67e_linear_sgd/best.pt", feats_e, "m67e global whiten SGD")
