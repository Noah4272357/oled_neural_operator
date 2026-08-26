"""Debug why whitening stalls training (train rl2 ~1.0, no movement)."""
import sys

from pathlib import Path

PROJECT = str(Path(__file__).resolve().parents[4])  # repo root (m6_logs -> ... -> repo)
sys.path.insert(0, PROJECT)

import importlib.util

spec = importlib.util.spec_from_file_location(
    "m6_spike_spectral",
    PROJECT + "/experiments/analysis/2026-08-24/m6_spike_spectral.py",
)
m = importlib.util.module_from_spec(spec)
sys.modules["m6_spike_spectral"] = m
spec.loader.exec_module(m)

import numpy as np
import torch

loaders = m.make_loaders(batch_size=64)
whit = m.train_whiten(loaders)
print("W: abs mean", whit.abs().mean().item(), "abs max", whit.abs().max().item())

batch = next(iter(loaders["train"]))
x = batch["input"].to(torch.float64)
d = batch["target"].to(torch.float64)
xft = torch.fft.rfft(x, dim=1)
z = torch.einsum("bki,ij->bkj", xft, whit)
g = z.conj().mT @ z
print("z abs mean", z.abs().mean().item(), "abs max", z.abs().max().item())
print("z Gram trace/(B*K*13):", g.trace().real.item() / (z.shape[0] * z.shape[1]) / 13)
dg = g.diagonal().real
print("z Gram diag range:", dg.min().item(), "..", dg.max().item())
dft = torch.fft.rfft(d, dim=1)
print("target d abs mean:", d.abs().mean().item(), "abs max:", d.abs().max().item())

torch.manual_seed(20260810)
model = m.SpectralSharedMap(13, 2, dtype=torch.cdouble)
model.set_whiten(whit)
opt = torch.optim.AdamW(model.parameters(), lr=1e-1, weight_decay=0.0)

denom = torch.linalg.vector_norm(d)
pred = model(x)
loss = torch.linalg.vector_norm(pred - d) / denom
print("init loss:", loss.item())
loss.backward()
gm = model.M.grad
print("grad abs mean", gm.abs().mean().item(), "abs max", gm.abs().max().item(),
      "NaN?", torch.isnan(gm).any().item())

# per-batch closed-form optimum in whitened frame
r = z.conj().mT @ dft  # (13,2)
G = z.conj().mT @ z
Mstar = torch.linalg.solve(G, r).conj().mT  # (2,13)
print("M* abs mean", Mstar.abs().mean().item(), "abs max", Mstar.abs().max().item())
print("M init abs mean", model.M.detach().abs().mean().item())

pred_star = torch.fft.irfft(torch.einsum("bki,oi->bko", z, Mstar), n=501, dim=1)
print("loss at M*:", torch.linalg.vector_norm(pred_star - d).item() / denom.item())

# 20 AdamW steps on this single batch
for i in range(20):
    opt.zero_grad(set_to_none=True)
    pred = model(x)
    loss = torch.linalg.vector_norm(pred - d) / denom
    loss.backward()
    opt.step()
print("loss after 20 adamw steps (lr 1e-1, one batch):", loss.item())
print("M after: abs mean", model.M.detach().abs().mean().item())
