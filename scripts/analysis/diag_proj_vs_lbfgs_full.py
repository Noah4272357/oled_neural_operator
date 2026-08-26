"""Full 1004-column comparison on the SAME error metric:

  (A) LS projection fit:  P Y  (U U^H Y, full-rank projection)
  (B) LBFGS fit:          (X/sd) @ W.t() / s

Decisive question: ||Y - P Y||^2 vs ||Y - Y_lbfgs||^2 (same flat space).
Projection minimum requires (A) <= (B).  Any violation => one of the two
fit paths is not what it claims to be.  Also verify P Y_lbfgs == Y_lbfgs
(all columns, not just column 0), and rebuild the LBFGS time-domain path
to cross-check the (B,251,4) reshape/irfft rebuild used by evaluate().
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

U, S, Vh = torch.linalg.svd(X / sd, full_matrices=False)
k = int((S > 1e-14 * S[0]).sum())
P = U[:, :k] @ U[:, :k].t()
Yfit_proj = (P @ (Y * s)) / s
W = torch.load("outputs/m67h_lbfgs_f64/best.pt", map_location="cpu",
               weights_only=True)["weight"].double()
Yfit_lbfgs = ((X / sd) @ W.t()) / s

ssq = lambda t: t.square().sum().item()
num_p = ssq(Y - Yfit_proj); num_l = ssq(Y - Yfit_lbfgs); den = ssq(Y)
print(f"[full] X {tuple(X.shape)} rank k={k} | projection error  = {num_p:.6e}"
      f"  R^2 = {1 - num_p/den:.6f}")
print(f"[full] LBFGS error      = {num_l:.6e}  R^2 = {1 - num_l/den:.6f}")
print(f"[full] proj - lbfgs     = {num_p - num_l:.6e} "
      f"(>0 => LBFGS beats the projection => MATHS VIOLATED)")

# LBFGS preds must lie in col(X): P Y_lbfgs == Y_lbfgs?
resid = ssq(Yfit_lbfgs - P @ Yfit_lbfgs)
print(f"[full] ||Yfit_lbfgs - P Yfit_lbfgs||^2 = {resid:.3e} "
      f"(of ||Yfit_lbfgs||^2 = {ssq(Yfit_lbfgs):.3e})")

# per-column: on how many columns does LBFGS actually beat the projection?
per_col_proj = (Y - Yfit_proj).square().sum(0)
per_col_lbfgs = (Y - Yfit_lbfgs).square().sum(0)
beats = (per_col_lbfgs < per_col_proj)
print(f"[full] columns where LBFGS < projection: {beats.sum().item()}/1004")
if beats.any():
    worst = torch.where(beats)[0]
    print(f"[full] worst such columns: "
          f"{[(i.item(), f'{per_col_lbfgs[i].item()/per_col_proj[i].item():.4f}x') for i in worst[:6]]}")

# time-domain rebuild of LBFGS pred (the evaluate() path), error vs r:
num_t = den_t = 0.0
with torch.no_grad():
    for b in loaders["train"]:
        rr = b["target"].double()
        out = ((feats(b) / sd) @ W.t()) / s
        v4 = out.reshape(len(b["input"]), N_BIN, 4)
        rhat = torch.fft.irfft(torch.complex(v4[..., :2], v4[..., 2:]),
                               n=N_POINTS, dim=1)
        num_t += (rhat - rr).square().sum().item()
        den_t += rr.square().sum().item()
print(f"[full] LBFGS time-domain ||err||^2/||r||^2 = {num_t/den_t:.6e} "
      f"-> R^2 = {1 - num_t/den_t:.6f}   (den_t = {den_t:.3e})")

# and the projection through the SAME time-domain rebuild:
num_p2 = 0.0
with torch.no_grad():
    for i, b in enumerate(loaders["train"]):
        rr = b["target"].double()
        n = len(b["input"])
        yf = Yfit_proj[i * n:(i + 1) * n]
        v4 = yf.reshape(n, N_BIN, 4)
        rhat = torch.fft.irfft(torch.complex(v4[..., :2], v4[..., 2:]),
                               n=N_POINTS, dim=1)
        num_p2 += (rhat - rr).square().sum().item()
print(f"[full] projection time-domain ||err||^2/||r||^2 = {num_p2/den_t:.6e} "
      f"-> R^2 = {1 - num_p2/den_t:.6f}")
