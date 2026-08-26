"""Per-bin linear predictability R^2, CLEAN eval path.

The 9.91e-5 / R^2>=0.997 claims in diag_r_predictability.py were computed
via the direct einsum eval (contraction-order artifact, 2026-08-26).  Here
we recompute per-bin R^2 with the SAFE path (x V S^-1 first), on test.

Per bin k: rhat_k = U_test[k-col] @ (U^H R)_k over the truncated directions.
R^2_k = 1 - ||r_k - rhat_k||^2 / ||r_k||^2 (test set, residual targets).
"""
import sys
sys.path.insert(0, "/nishome/charliewang/forge-projects/oled-neural-operator")
import time
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

def feats(b):
    z = torch.fft.rfft(b["input"], dim=1)[:, :F_IN + 1, :]
    return torch.cat([z.real, z.imag], dim=-1).reshape(len(b["input"]), -1).double()

t0 = time.perf_counter()
X = torch.cat([feats(b) for b in loaders["train"]])
sd = X.std(0).clamp_min(1e-12)
X = X / sd
R = torch.cat([torch.fft.rfft(b["target"].double(), dim=1) for b in loaders["train"]])
U, S, Vh = torch.linalg.svd(X, full_matrices=False)
smax = S[0].item()
print(f"[r2] X {tuple(X.shape)} SVD done ({time.perf_counter()-t0:.0f}s)", flush=True)

# per-bin clean-path fit at the best rcond found by the grid (1e-2, k=164)
for rcond in (1e-2, 1e-4, 1e-6):
    k = int((S > rcond * smax).sum())
    UtR = torch.einsum("bi,bko->iko", U[:, :k].to(torch.complex128), R)
    VSinv = Vh.conj().T[:, :k] / S[:k].unsqueeze(0)
    # accumulate per-bin SSE / SST over test
    sse = torch.zeros(N_BIN, 2, dtype=torch.float64)
    sst = torch.zeros(N_BIN, 2, dtype=torch.float64)
    with torch.no_grad():
        for b in loaders["test"]:
            r = torch.fft.rfft(b["target"].double(), dim=1)
            xw = (feats(b) / sd) @ VSinv
            rhat = torch.einsum("bk,koc->boc", xw.to(torch.complex128), UtR)
            sse += (r - rhat).abs().square().sum(0)   # (251, 2)
            sst += r.abs().square().sum(0)            # (251, 2)
    r2 = 1 - sse.sum(1) / sst.sum(1)                  # per-bin, channels merged
    print(f"[r2] rcond {rcond:.0e} (k={k}): per-bin R^2 median={r2.median():.5f} "
          f"min={r2.min():.5f} | bins<0.99: {(r2<0.99).sum().item()}/251 | "
          f"bins<0.9: {(r2<0.9).sum().item()}/251", flush=True)
    # tonal bins 4-7 and the broadband tail band 8-100
    print(f"[r2]   bins 4-7: {[f'{v:.5f}' for v in r2[4:8].tolist()]}")
    print(f"[r2]   tail bins 8-100 median R^2 = {r2[8:101].median():.5f}")
