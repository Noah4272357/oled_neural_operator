"""Structural anatomy of the Tikhonov solution (aligned, f64).

Question: what is the minimal EXPRESSIBLE structure that reaches
test rl2 <= 1e-4, so that a small-parameter SGD (M6 style) can converge?

Facts established so far:
  - per-bin independent: rl2 = 1.526e-3  (== ||rr||/||dist|| with rhat=0:
    no per-bin 8->2 map predicts its own bin; cross-frequency is required)
  - Toeplitz shared kernel (K=2..50): WORSE (5.5e-2..0.49) -- the map from
    input neighborhood to output bin is NOT translation-invariant
  - full flat linear (3232->1004), Tikhonov lam=1e-3: 9.42e-5 PASS but
    3.24M params -> too big for fast SGD

This script dissects W (1004, 3232):
  1. SVD effective rank of W (how compressible is the solution?)
  2. per-output-bin weight-norm distribution (which output bins matter?)
  3. rr spectrum energy distribution (which output bins need prediction?)
  4. truncated-rank W: test rl2 vs rank r (rank needed to stay <= 1e-4)
  5. per-output-bin weight distribution over input bins (localization?)
"""
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

ckpt = torch.load("outputs/tikhonov_linear_1e-3/model.pt", map_location="cpu", weights_only=True)
W = ckpt["weight"]  # (3232, 1004): prediction = (X/sd) @ W / s
sd = ckpt["sd"]
s = ckpt["s"]

# ---- data (aligned single-pass) ----
loaders = build_dataloaders(DATA_CFG, seed=20260810)
head = torch.load("outputs/lti_head_16ch.pt", map_location="cpu", weights_only=True)

def feats(b):
    z = torch.fft.rfft(b["input"].double(), dim=1)[:, :F_IN + 1, :]
    return torch.cat([z.real, z.imag], dim=-1).reshape(len(b["input"]), -1)

def ft_flat(b):
    z = torch.fft.rfft(b["target"].double(), dim=1)
    return torch.cat([z.real, z.imag], dim=-1).reshape(len(b["input"]), -1)

def build(split):
    Xs, Ys = [], []
    for b in loaders[split]:
        Xs.append(feats(b)); Ys.append(ft_flat(b))
    return torch.cat(Xs), torch.cat(Ys)

Xtr, Ytr = build("train")
Xte, Yte = build("test")

def rl2_from_flat(Fte, split="test"):
    num = den = 0.0
    lo = 0
    for b in loaders[split]:
        n = len(b["input"])
        rr = b["target"].double()
        z_in = torch.fft.rfft(b["input"].double(), dim=1)
        hp = torch.fft.irfft(torch.einsum("bki,koi->bko", z_in,
                                          head.to(torch.complex128)),
                             n=N_POINTS, dim=1)
        dist = rr + hp
        v4 = Fte[lo:lo + n].reshape(-1, N_BIN, 4)
        rhat = torch.fft.irfft(torch.complex(v4[..., :2], v4[..., 2:]),
                               n=N_POINTS, dim=1)
        num += (hp + rhat - dist).square().sum().item()
        den += dist.square().sum().item()
        lo += n
    return (num / den) ** 0.5

# ---- 1. SVD effective rank of W ----
U, S, Vh = torch.linalg.svd(W, full_matrices=False)
print(f"[1] W {tuple(W.shape)} SVD singular spectrum:")
for thr in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
    r = int((S > thr * S[0]).sum())
    print(f"    rank @ >{thr:g}*Smax = {r}  (energy kept "
          f"{S[:r].square().sum() / S.square().sum():.6f})")
cum = torch.cumsum(S.square(), 0) / S.square().sum()
for frac in (0.5, 0.9, 0.99, 0.999, 0.9999):
    r = int((cum <= frac).sum())
    print(f"    rank for {frac:.4f} energy = {r}")

# ---- 2. per-output-bin weight norms (which output bins matter) ----
# W is (3232, 1004); output bin k occupies columns [4k, 4k+4)
wnorm = W.T.reshape(N_BIN, 4, -1).norm(dim=(1, 2))
print("\n[2] per-output-bin weight norm (flat [r0,r1,i0,i1]):")
for k in (0, 1, 10, 50, 100, 150, 200, 250):
    print(f"    bin {k:3d}: ||W_k|| = {wnorm[k]:.4e}")
print(f"    bins 0-100 total {wnorm[:101].sum():.4e} | "
      f"bins 101-250 total {wnorm[101:].sum():.4e}")

# ---- 3. rr spectrum energy distribution ----
Z = torch.fft.rfft(torch.fft.irfft(
    torch.complex(Yte.reshape(-1, N_BIN, 2, 2)[..., 0],
                  Yte.reshape(-1, N_BIN, 2, 2)[..., 1]), n=N_POINTS, dim=1),
    dim=1)
ek = Z.abs().square().sum(dim=(0, 2))
print("\n[3] rr test spectrum energy per bin:")
print(f"    bins 0-100 {ek[:101].sum()/ek.sum():.6f} | "
      f"bins 101-250 {ek[101:].sum()/ek.sum():.6f}")
for k in (0, 10, 50, 100, 101, 150, 200, 250):
    print(f"    bin {k:3d}: {ek[k]/ek.sum():.6e}")

# ---- 4. truncated-rank W: test rl2 vs r ----
print("\n[4] truncated-rank W test rl2 (rank scan):")
for r in (16, 32, 64, 128, 256, 512, 1024, 2339):
    Wr = (U[:, :r] * S[:r]) @ Vh[:r]
    Fte_r = (Xte / sd) @ Wr / s
    v = rl2_from_flat(Fte_r)
    flag = "PASS" if v <= 1e-4 else ""
    print(f"    rank {r:4d}: test rl2 = {v:.6e} {flag}")
    if r >= 2339:
        break

# ---- 5. per-output-bin input localization (energy of W_k over input bins) ----
# feature layout: per input bin 32 reals = 16ch real + 16ch imag (interleaved
# within each bin's block: [re_0..re_15, im_0..im_15]? -- NO: rfft is complex,
# feats() cats [z.real, z.imag] along last dim => [re0..re15, im0..im15] of
# bin t occupies rows [32t, 32t+32)?? verify below from actual layout)
print("\n[5] per-output-bin input-spectrum localization (input bins 0-100):")
# NOTE: feats() = cat([z.real, z.imag], -1).reshape(B, -1) with z (B,101,16)
# => layout is [bin0 real(16), bin0 imag(16), bin1 real(16), ...]
Wd = W.T.reshape(N_BIN, 4, F_IN + 1, 2, 16)  # (251,4,101,real/imag,16ch)
E = Wd.square().sum(dim=(1, 3, 4)).sqrt()  # (251, 101): output k x input bin
E = E / E.sum(dim=1, keepdim=True).clamp_min(1e-30)
for k in (0, 50, 100, 150, 250):
    top = E[k].topk(6).indices
    print(f"    out bin {k:3d}: top input bins {top.tolist()} "
          f"(share {E[k][top].sum().item():.3f})")

# ---- 6. input-side PCA projection + full-connection LS (closed form) ----
# X (5000,3232) has numerical rank 2339.  Project input to d principal
# components (no whitening, just truncation), fit a dense LS map d->1004.
# If d small (~512) reaches <= 1e-4, the structure "input PCA + dense head"
# is expressible with d*1004 params, and SGD on PCA coordinates (decorrelated,
# unit variance) converges fast (M6-style whitening).
print("\n[6] input PCA projection + dense LS (closed form, train rl2 | test rl2):")
Xc = Xtr / sd
Ux, Sx, Vhx = torch.linalg.svd(Xc, full_matrices=False)  # Xc = U S Vh
# principal scores: Xc @ Vhx[:d].t() = (U*S)[:, :d]  (5000, d)
for d in (64, 128, 256, 512, 768, 1024, 1500, 2339):
    Xd = Xc @ Vhx[:d].t()  # (5000, d)
    Wd = torch.linalg.lstsq(Xd, Ytr * s).solution  # (d, 1004)
    Fte_d = (Xte / sd) @ Vhx[:d].t() @ Wd / s
    v = rl2_from_flat(Fte_d)
    Ftr_d = Xd @ Wd / s
    vt = rl2_from_flat(Ftr_d, "train")
    flag = "PASS" if v <= 1e-4 else ""
    print(f"    d={d:4d} (params {d*1004:>9}): train {vt:.6e} | test {v:.6e} {flag}")
    if d >= 2339:
        break

# ---- 7. PCA-domain Tikhonov (lambda scan in the compressed space) ----
# d=1024 LS without regularization is 1.02e-4 (2.3% off).  Adding ridge in
# the PCA domain should recover the full-space Tikhonov behaviour.
print("\n[7] PCA-domain Tikhonov (d, lambda) scan -- test rl2:")
for d in (1024, 1280, 1536, 1792, 2048, 2339):
    Xd = Xc @ Vhx[:d].t()  # (5000, d), scores (NOT whitened)
    for lam in (1e-3, 3e-3, 1e-2):
        Uq, Sq, Vhq = torch.linalg.svd(Xd, full_matrices=False)
        g = Sq / (Sq * Sq + lam)
        Wd = Vhq.t() @ (g.unsqueeze(1) * (Uq.t() @ (Ytr * s)))
        Fte_d = (Xte / sd) @ Vhx[:d].t() @ Wd / s
        v = rl2_from_flat(Fte_d)
        flag = "PASS" if v <= 1e-4 else ""
        print(f"    d={d:4d} lambda={lam:.0e}: test {v:.6e} {flag}")
