"""Input bin truncation grid: does the disturbance broadband tail depend
only on LOW input frequencies?

If yes, high bins are pure noise directions that hurt generalization (3232
dims -> far fewer noise dirs -> better test rl2).  Compare closed-form LS
and the best Tikhonov/truncation per F_IN -- but the decisive signal is the
TRAINED-trajectory analogue, so first map the closed-form landscape.

Hypothesis to test: cross-band coupling r depends on input bands 0..F_IN;
if the tail drives the gap, test rl2 vs F_IN will show a clear minimum.
"""
import sys
sys.path.insert(0, "/nishome/charliewang/forge-projects/oled-neural-operator")
import time
import torch

N_POINTS, N_BIN = 501, 251
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

def feats(b, fin):
    z = torch.fft.rfft(b["input"], dim=1)[:, :fin + 1, :]
    return torch.cat([z.real, z.imag], dim=-1).reshape(len(b["input"]), -1).double()

t0 = time.perf_counter()
X = torch.cat([feats(b, 100) for b in loaders["train"]])
R = torch.cat([torch.fft.rfft(b["target"].double(), dim=1) for b in loaders["train"]])
print(f"[bin] full X {tuple(X.shape)} built ({time.perf_counter()-t0:.0f}s)", flush=True)

for fin in (10, 15, 20, 30, 40, 60, 100):
    # column scale on the truncated feature set
    Xf = X[:, :(fin + 1) * 32]
    sd = Xf.std(0).clamp_min(1e-12)
    Xf = Xf / sd
    U, S, Vh = torch.linalg.svd(Xf, full_matrices=False)
    smax = S[0].item()
    best = None
    for rc in (1e-2, 1e-4, 1e-6, 1e-8):
        k = int((S > rc * smax).sum())
        if k == 0:
            continue
        UtR = torch.einsum("bi,bko->iko", U[:, :k].to(torch.complex128), R)
        VSinv = Vh.conj().T[:, :k] / S[:k].unsqueeze(0)
        num = den = 0.0
        with torch.no_grad():
            for b in loaders["test"]:
                xx = b["input"].double(); rr = b["target"].double()
                z = torch.fft.rfft(xx, dim=1)
                hp = torch.fft.irfft(
                    torch.einsum("bki,koi->bko", z, head.to(torch.complex128)),
                    n=N_POINTS, dim=1)
                dist = rr + hp
                xw = (feats(b, fin) / sd) @ VSinv
                rhat = torch.einsum("bk,koc->boc", xw.to(torch.complex128), UtR)
                num += (torch.fft.rfft(rr, dim=1) - rhat).abs().square().sum()
                den += torch.fft.rfft(dist, dim=1).abs().square().sum()
        rl2 = (num / den).sqrt().item()
        best = rl2 if best is None or rl2 < best else best
    print(f"[bin] F_IN={fin:3d} dim={Xf.shape[1]:4d} best trunc rl2 = {best:.6e}", flush=True)
