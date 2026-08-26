"""Independent verification of the SGD-trained PCA-1280 model (official
evaluation protocol, aligned inline per batch, all splits).

Checks:
  - every seed's test rl2 reproduces 9.480689e-05 (PASS < 1e-4)
  - val rl2 (manifest order, same protocol) -- for the record (Tikhonov
    reference showed val 1.63e-4 > test 9.42e-5, worth re-measuring here)
  - model vs closed-form PCA-Tikhonov reference: weight relative diff
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


def feats(b):
    z = torch.fft.rfft(b["input"].double(), dim=1)[:, :F_IN + 1, :]
    return torch.cat([z.real, z.imag], dim=-1).reshape(len(b["input"]), -1)


def rebuild(F):
    v4 = F.reshape(-1, N_BIN, 4)
    return torch.fft.irfft(torch.complex(v4[..., :2], v4[..., 2:]),
                           n=N_POINTS, dim=1)


def main() -> None:
    loaders = build_dataloaders(DATA_CFG, seed=20260810)
    head = torch.load("outputs/lti_head_16ch.pt", map_location="cpu", weights_only=True)

    def rl2(split, W, Vd, sd, s):
        num = den = 0.0
        for b in loaders[split]:
            n = len(b["input"])
            rr = b["target"].double()
            z_in = torch.fft.rfft(b["input"].double(), dim=1)
            hp = torch.fft.irfft(torch.einsum("bki,koi->bko", z_in,
                                              head.to(torch.complex128)),
                                 n=N_POINTS, dim=1)
            dist = rr + hp
            f = feats(b)
            out = ((f / sd) @ Vd.t()) @ W / s
            rhat = rebuild(out)
            num += (hp + rhat - dist).square().sum().item()
            den += dist.square().sum().item()
        return (num / den) ** 0.5

    print(f"{'seed':>8} | {'train rl2':>12} | {'val rl2':>12} | {'test rl2':>12}")
    for seed in (20260810, 12345, 98765):
        p = Path(f"outputs/sgd_pca_1280/seed_{seed}/model.pt")
        ck = torch.load(p, map_location="cpu", weights_only=True)
        tr = rl2("train", ck["W"], ck["Vd"], ck["sd"], ck["s"])
        va = rl2("val", ck["W"], ck["Vd"], ck["sd"], ck["s"])
        te = rl2("test", ck["W"], ck["Vd"], ck["sd"], ck["s"])
        flag = "PASS" if te <= 1e-4 else "FAIL"
        print(f"{seed:>8} | {tr:.6e} | {va:.6e} | {te:.6e}  {flag}")

    # closed-form PCA-domain Tikhonov reference (diag_tikhonov_structure.py [7])
    Xp, Yp = [], []
    for b in loaders["train"]:
        Xp.append(feats(b))
        z = torch.fft.rfft(b["target"].double(), dim=1)
        Yp.append(torch.cat([z.real, z.imag], dim=-1).reshape(len(b["input"]), -1))
    Xtr, Ytr = torch.cat(Xp), torch.cat(Yp)
    sd = Xtr.std(0).clamp_min(1e-12)
    s = 1.0 / Ytr.std().clamp_min(1e-12)
    _, Sx, Vhx = torch.linalg.svd(Xtr / sd, full_matrices=False)
    for d, lam in ((1280, 1e-3),):
        Xd = (Xtr / sd) @ Vhx[:d].t()
        Uq, Sq, Vhq = torch.linalg.svd(Xd, full_matrices=False)
        g = Sq / (Sq * Sq + lam)
        Wref = Vhq.t() @ (g.unsqueeze(1) * (Uq.t() @ (Ytr * s)))
        ck = torch.load("outputs/sgd_pca_1280/seed_20260810/model.pt",
                        map_location="cpu", weights_only=True)
        rel = (ck["W"] - Wref).norm().item() / Wref.norm().item()
        print(f"\n[ref] closed-form d={d} lam={lam} test rl2 "
              f"{rl2('test', Wref, Vhx[:d], sd, s):.6e}")
        print(f"[cmp] SGD W vs closed-form W: rel diff {rel:.3e}")


if __name__ == "__main__":
    main()
