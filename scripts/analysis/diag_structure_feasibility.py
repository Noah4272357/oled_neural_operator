"""Structure feasibility scan for the SGD-training campaign (aligned, f64).

Question: which LINEAR structure on the 8ch input spectrum can reach
test rl2 <= 1e-4 in closed form (i.e. is expressible)?  Only structures
that are expressible matter -- SGD then only has to CONVERGE, and small
parameter counts converge fast (M6 lesson: 26 params).

Structures:
  A. per-bin independent (no cross-frequency coupling): for each output
     bin k, a 2x8 complex map from input bin k's 8 channels.  Input has
     energy only in bins 0-100 -> output bins 101-250 are unpredicted.
     Expectation: FAILS (this is the head/SpectralConv structure that
     motivated the campaign).
  B. frequency-domain convolution (Toeplitz): output bin k depends on
     input bins in [k-K, k+K] (clamped to 0-100).  Kernel length K scan.
     Cross-frequency coupling with O(K*8*2) complex params per output
     (K*16 total for a shared kernel).  Expectation: PASSES for K large
     enough -- then SGD trains a small Toeplitz kernel.
  C. full flat linear (3232->1004) already known: Tikhonov 9.42e-5 PASS
     but 3.24M params -> SGD slow.  Not retested here.
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

# ---- aligned single-pass data: complex input spectra Z (B,251,8) and
#      complex target spectra R (B,251,2) + flat Y for scaling stats ----
def build(split):
    zs, rs = [], []
    for b in loaders[split]:
        zs.append(torch.fft.rfft(b["input"].double(), dim=1))
        rs.append(torch.fft.rfft(b["target"].double(), dim=1))
    return torch.cat(zs), torch.cat(rs)
Ztr, Rtr = build("train")
Zte, Rte = build("test")
# target scale for the flat (B,1004) parameterization (training compat)
Ytr_flat = torch.cat([Rtr.real, Rtr.imag], dim=-1).reshape(-1, N_BIN * 4)
s = 1.0 / Ytr_flat.std().clamp_min(1e-12)

def rl2(Z, R, rhat_c):
    """official: ||(hp+rhat) - dist|| / ||dist||  (aligned, per batch inline)"""
    num = den = 0.0
    lo = 0
    for b in loaders["test"]:
        n = len(b["input"])
        rr = b["target"].double()
        z_in = torch.fft.rfft(b["input"].double(), dim=1)
        hp = torch.fft.irfft(torch.einsum("bki,koi->bko", z_in,
                                          head.to(torch.complex128)),
                             n=N_POINTS, dim=1)
        dist = rr + hp
        rhat = torch.fft.irfft(rhat_c[lo:lo + n], n=N_POINTS, dim=1)
        num += (hp + rhat - dist).square().sum().item()
        den += dist.square().sum().item()
        lo += n
    return (num / den) ** 0.5

# ---------- A. per-bin independent: R_hat[k] = Z[k] @ M_k, M_k in C^{8x2} ----------
print("[A] per-bin independent complex LS (per-bin 8->2):")
for k in (0, 4, 50, 100, 101, 200):
    Mk = torch.linalg.lstsq(Ztr[:, k], Rtr[:, k]).solution
    res = ((Ztr[:, k] @ Mk - Rtr[:, k]).abs().square().sum()
           / Rtr[:, k].abs().square().sum()).sqrt().item()
    print(f"  bin {k}: ls residual ||Z M - R||/||R|| = {res:.3e}")
rhat = torch.zeros_like(Rte)
for k in range(N_BIN):
    Mk = torch.linalg.lstsq(Ztr[:, k], Rtr[:, k]).solution
    rhat[:, k] = Zte[:, k] @ Mk
print(f"  [A] per-bin independent test rl2 = {rl2(Zte, Rte, rhat):.6e}")

# ---------- B. frequency-domain convolution (Toeplitz), kernel K ----------
#  rhat[k, c] = sum_{d=0}^{1} sum_{j=max(0,k-K)}^{min(100,k+K)} Z[j, d] Kc[j-(k-K)...]
#  shared kernel: 2 input ch -> 2 output ch, support 2K+1, per output bin k
#  parameterization: linear map from Z bins [k-K, k+K] x 8ch -> R bin k x 2ch
#  For a SHARED kernel this is a (2K+1)*8 -> 2 map, SAME for every k.
print("[B] frequency-domain convolution (shared kernel, cross-frequency):")
for K in (0, 2, 5, 10, 20, 50, 100):
    if K == 0:
        # reuse per-bin independent (K=0 = no neighbors)
        continue
    # build design matrix: for each (k, t) with t in [k-K, k+K] n [0,100],
    # collect Ztr[:, t, :] -> feature dim (2K+1)*8 (zero-padded)
    W_sup = 2 * K + 1
    A = torch.zeros(len(Ztr), N_BIN, W_sup * 16, dtype=torch.complex128)
    mask = torch.zeros(N_BIN, W_sup, dtype=torch.bool)
    for k in range(N_BIN):
        t0 = max(0, k - K); t1 = min(100, k + K)
        for t in range(t0, t1 + 1):
            j = t - (k - K)
            A[:, k, j * 16:(j + 1) * 16] = Ztr[:, t]
            mask[k, j] = True
    # solve for a SHARED W (W_sup*16 -> 2) on all (k, sample) pairs
    Amat = A.reshape(-1, W_sup * 16)          # (B*251, (2K+1)*16), zero-padded
    Rmat = Rtr.reshape(-1, 2)                # (B*251, 2)
    Wk = torch.linalg.lstsq(Amat, Rmat).solution
    # apply
    Ate = torch.zeros(len(Zte), N_BIN, W_sup * 16, dtype=torch.complex128)
    for k in range(N_BIN):
        t0 = max(0, k - K); t1 = min(100, k + K)
        for t in range(t0, t1 + 1):
            j = t - (k - K)
            Ate[:, k, j * 16:(j + 1) * 16] = Zte[:, t]
    rhat = (Ate.reshape(-1, W_sup * 16) @ Wk).reshape(-1, N_BIN, 2)
    n_params = Wk.numel()
    print(f"  K={K:3d} (kernel {W_sup}, params {n_params}): "
          f"test rl2 = {rl2(Zte, Rte, rhat):.6e}")
