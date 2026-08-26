"""M6 spike: pure-spectral (frequency-domain) models — feasibility & float64 pipeline.

Questions (low cost, high information, before the M6 plan):
  S1  shared-map spectral model (LTI: single 2x13 complex map across bins) —
      SGD from scratch vs closed-form LS ceiling. M5-1 showed the joint LS fit
      recovers A_true to 5.5e-13 (per-bin test rl2 2.26e-11); is the convex
      quadratic optimizable by SGD to the same level? The exact inverse
      operator d = M qdd - B u is frequency-independent, so a shared map is
      the exactly-right model class (26 complex params).
  S2  per-bin spectral model (M6 brief "P2d": 251 independent 2x13 complex
      maps, ~6.5k complex params). Convex per-bin but subset-sensitive per
      M5-1 evidence (per-bin fits ranged 1e-10 .. 1e-2 test rl2 across
      training subsets).
  S3  float64 vs float32 (F4 rounding-gradient spike check) and
      AdamW vs SGD+momentum on the quadratic.
  REF closed-form LS shared map on bins 3..250 (numpy lstsq, float64) — the
      convex optimum ceiling, evaluated on the full 501-pt basis.

Metrics: src.training.validate.validate = the exact code path of the official
scripts/evaluate.py (global relative_l2 aggregation, full precision, official
~/data/neural_operator_3 splits). No TF32 involved (CPU).

Usage (project .venv, CPU):
  .venv/bin/python experiments/analysis/2026-08-24/m6_spike_spectral.py [--epochs 50] [--smoke]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import build_dataloaders
from src.training.factories import build_loss
from src.training.validate import validate

DATA_ROOT = str(Path.home() / "data/neural_operator_3")
SEED = 20260810
N_PTS = 501
N_BIN = N_PTS // 2 + 1  # 251
N_IN = 13
N_OUT = 2


class _SpectralBase(nn.Module):
    """Shared plumbing: optional fixed per-channel input scaling (buffer).

    The 13 input channels span 4 orders of magnitude in RMS (force ~15,
    states ~1e-4..5e-2) — the joint Gram of the bare quadratic has eigenvalues
    from ~1e11 down to ~1e-4 with the loss-relevant directions at 1.7e3-2.4e4
    (m6_spike_eigen.py).  A fixed per-channel division tightens that spread
    by ~1e8; since the map is linear the scaling folds into M at eval time.
    """

    def set_input_scale(self, scale: torch.Tensor) -> None:
        self.register_buffer("in_scale", scale.detach().clone())

    def set_whiten(self, w: torch.Tensor) -> None:
        """Fixed whitening matrix W (13x13 complex) applied to the rfft bins.

        W = C^{-1/2} of the joint train Gram C = E[x_ft^H x_ft]; makes the
        probed input subspace near-identity so first-order optimizers converge
        on the quadratic (m6_spike_eigen.py: bare Gram has loss-relevant
        eigenvalues spanning ~5 orders). Linear => folds into M at eval.
        """
        self.register_buffer("W", w.detach().clone())

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "in_scale"):
            return x / self.in_scale.to(x.dtype)
        return x


class SpectralSharedMap(_SpectralBase):
    """Direct complex-linear map, ONE 2x13 matrix shared across all bins (LTI)."""

    def __init__(self, in_channels: int, out_channels: int, dtype=torch.cdouble):
        super().__init__()
        scale = 1.0 / in_channels
        self.M = nn.Parameter(
            (scale * torch.rand(out_channels, in_channels, dtype=dtype))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 501, 13) -> (B, 501, 2)
        xft = torch.fft.rfft(self._norm(x), dim=1)  # (B, 251, 13) complex
        if hasattr(self, "W"):
            xft = torch.einsum("bki,ij->bkj", xft, self.W.to(xft.dtype))
        out_ft = torch.einsum("bki,oi->bko", xft, self.M)  # (B, 251, 2)
        return torch.fft.irfft(out_ft, n=x.shape[1], dim=1)


class SpectralPerBin(_SpectralBase):
    """Per-bin complex-linear maps (M6 P2d design): 251 independent 2x13 maps."""

    def __init__(self, in_channels: int, out_channels: int, n_bin: int, dtype=torch.cdouble):
        super().__init__()
        scale = 1.0 / in_channels
        self.M = nn.Parameter(
            (scale * torch.rand(n_bin, out_channels, in_channels, dtype=dtype))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xft = torch.fft.rfft(self._norm(x), dim=1)  # (B, 251, 13)
        if hasattr(self, "W"):
            xft = torch.einsum("bki,ij->bkj", xft, self.W.to(xft.dtype))
        out_ft = torch.einsum("bki,koi->bko", xft, self.M)
        return torch.fft.irfft(out_ft, n=x.shape[1], dim=1)


class GridAdapter(nn.Module):
    """map (inputs, grid) -> model(inputs), matching the trainer/evaluate call.

    Casts inputs to the model's dtype (float64 when the spectral map is
    complex128), so official-metric evaluation matches the trained dtype.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        p = next(model.parameters())
        self.in_dtype = torch.float64 if p.dtype == torch.cdouble else torch.float32

    def forward(self, inputs: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return self.model(inputs.to(self.in_dtype))


def make_loaders(batch_size: int = 64):
    config = {
        "root": DATA_ROOT,
        "input_fields": ["force", "displacement", "velocity", "acceleration"],
        "target_fields": ["disturbance"],
        "time_start": 0,
        "time_stop": None,
        "time_stride": 1,
        "preprocessing": {"name": "none"},
        "memory_cache": False,
        "batch_size": batch_size,
        "eval_batch_size": 64,
        "num_workers": 4,
        "max_train_samples": None,
        "max_val_samples": None,
        "max_test_samples": None,
    }
    return build_dataloaders(config, SEED)


def train_epoch(model, loader, opt, dtype_float, device="cpu"):
    """One epoch, per-batch relative_l2 (project-standard training loss)."""
    model.train()
    total = 0.0
    for batch in loader:
        x = batch["input"].to(device).to(dtype_float)
        d = batch["target"].to(device).to(dtype_float)
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        denom = torch.linalg.vector_norm(d).clamp_min(torch.finfo(dtype_float).tiny)
        loss = torch.linalg.vector_norm(pred - d) / denom
        loss.backward()
        opt.step()
        total += float(loss)
    return total / max(len(loader), 1)


def train_epoch_lbfgs(model, loader, opt, dtype_float, device="cpu"):
    """One L-BFGS 'epoch' = one full-batch closure step over the whole set."""
    model.train()
    xs, ds = [], []
    for batch in loader:
        xs.append(batch["input"].to(device).to(dtype_float))
        ds.append(batch["target"].to(device).to(dtype_float))
    x = torch.cat(xs)
    d = torch.cat(ds)

    def closure():
        opt.zero_grad()
        pred = model(x)
        denom = torch.linalg.vector_norm(d).clamp_min(torch.finfo(dtype_float).tiny)
        loss = torch.linalg.vector_norm(pred - d) / denom
        loss.backward()
        return loss

    loss = opt.step(closure)
    return float(loss)


def official_rl2(model, loader, device="cpu"):
    """Exact official metric: src.training.validate.validate (global rel_l2)."""
    adapter = GridAdapter(model)
    return validate(adapter, loader, build_loss({"name": "relative_l2"}), device)


def closed_form_reference(loaders, bins=(3, 251), reg=None):
    """Shared-map numpy lstsq on bins b0..b1 over all 5000 train samples.

    Matches M5-1's joint fit (recovered A_true to 5.5e-13). Evaluate the
    fixed map in float64 numpy on the official test split as the convex-
    optimum ceiling (the official float32 metric floors around ~1e-7, so the
    reference ceiling is measured in float64 directly).  ``reg`` adds a ridge
    to the joint Gram when not None.
    """
    train = loaders["train"].dataset
    G = np.zeros((N_IN, N_IN), dtype=np.complex128)
    rhs = np.zeros((N_IN, N_OUT), dtype=np.complex128)
    for i in range(len(train)):
        item = train[i]
        x = item["input"].numpy().astype(np.float64)
        d = item["target"].numpy().astype(np.float64)
        xft = np.fft.rfft(x, axis=0)  # (251, 13)
        dft = np.fft.rfft(d, axis=0)  # (251, 2)
        sel = slice(bins[0], bins[1])
        Xb, Db = xft[sel], dft[sel]  # (248, 13), (248, 2)
        G += Xb.conj().T @ Xb
        rhs += Xb.conj().T @ Db
    if reg is not None:
        G = G + reg * np.eye(N_IN, dtype=np.complex128)
    M, *_ = np.linalg.lstsq(G, rhs, rcond=None)  # (13, 2) complex

    # float64 numpy evaluation on the full test split (exact metric semantics)
    test = loaders["test"].dataset
    err2 = np.zeros(N_BIN, dtype=np.float64)
    tgt2 = np.zeros(N_BIN, dtype=np.float64)
    for i in range(len(test)):
        item = test[i]
        x = item["input"].numpy().astype(np.float64)
        d = item["target"].numpy().astype(np.float64)
        xft = np.fft.rfft(x, axis=0)
        dft = np.fft.rfft(d, axis=0)
        pred_ft = xft @ M  # (251, 2)
        err2 += np.sum(np.abs(pred_ft - dft) ** 2, axis=1)
        tgt2 += np.sum(np.abs(dft) ** 2, axis=1)
    # group decomposition (bins 0-2 ramp / 3-7 tone / 8-250 tail)
    def grp(a, lo, hi):
        return float(np.sqrt(a[lo:hi].sum() / tgt2.sum()))
    return M, {
        "relative_l2": float(np.sqrt(err2.sum() / tgt2.sum())),
        "rl2_bins0_2": grp(err2, 0, 3),
        "rl2_bins3_7": grp(err2, 3, 8),
        "rl2_bins8_250": grp(err2, 8, 251),
        "e_frac_bins0_2": float(tgt2[0:3].sum() / tgt2.sum()),
        "e_frac_bins3_7": float(tgt2[3:8].sum() / tgt2.sum()),
        "e_frac_bins8_250": float(tgt2[8:251].sum() / tgt2.sum()),
    }


def per_bin_target_energy(loaders, n=100):
    """Sum |d_ft|^2 per bin over the first n train samples (float64)."""
    train = loaders["train"].dataset
    E = np.zeros(N_BIN, dtype=np.float64)
    for i in range(min(n, len(train))):
        d = train[i]["target"].numpy().astype(np.float64)
        dft = np.fft.rfft(d, axis=0)
        E += np.sum(np.abs(dft) ** 2, axis=1)
    return E


def train_channel_scale(loaders) -> torch.Tensor:
    """Per-channel RMS of the train split (time domain), for input scaling."""
    train = loaders["train"].dataset
    x2 = np.zeros(N_IN, dtype=np.float64)
    n = 0
    for i in range(len(train)):
        x = train[i]["input"].numpy().astype(np.float64)
        x2 += np.sum(x ** 2, axis=0)
        n += x.shape[0]
    return torch.tensor(np.sqrt(x2 / max(n, 1)), dtype=torch.float64)


def train_whiten(loaders) -> torch.Tensor:
    """PER-PAIR C^{-1/2} of the joint train Gram C = sum_n sum_k x_ft[n,k]^H x_ft[n,k].

    Hermitian square root via eigendecomposition (W = U Lambda^{-1/2} U^H),
    complex128, normalized so the covariance of z = x_ft @ W is IDENTITY PER
    (sample, bin) PAIR (not per aggregate).  Why: G aggregates 5000x251 pairs;
    without the sqrt(N_pairs) factor the whitened z has covariance I/1.25e6,
    the optimal map M* inflates ~1120x, and AdamW's fixed per-element step
    (F4) needs ~50k steps just to traverse parameter space — training looks
    frozen (round-2 S1w: train rl2 0.995 in 30 ep).
    """
    train = loaders["train"].dataset
    G = np.zeros((N_IN, N_IN), dtype=np.complex128)
    for i in range(len(train)):
        x = train[i]["input"].numpy().astype(np.float64)
        xft = np.fft.rfft(x, axis=0)
        G += xft.conj().T @ xft
    n_pairs = len(train) * N_BIN
    evals, evecs = np.linalg.eigh(G / n_pairs)
    evals = np.clip(evals, 1e-10, None)
    w = (evecs * (1.0 / np.sqrt(evals))) @ evecs.conj().T
    return torch.tensor(w, dtype=torch.cdouble)


def run_spike(epochs: int = 50, smoke: bool = False) -> None:
    torch.manual_seed(SEED)
    loaders = make_loaders()
    print(f"[data] train={len(loaders['train'].dataset)} val={len(loaders['val'].dataset)} test={len(loaders['test'].dataset)}")
    print(f"[data] batch 64 -> {len(loaders['train'])} batches/epoch")

    # --- per-bin target energy check (verify M5-1 tone 99.98% claim) ---
    E = per_bin_target_energy(loaders, n=min(100, smoke and 10 or 100))
    E_total = E.sum()
    tone = E[4:8].sum() / E_total
    low = E[0:3].sum() / E_total
    print(f"[data] per-bin target energy (100 train samples): tone bins 4-7 = {tone:.6f}, bins 0-2 = {low:.3e}")

    # --- REF: closed-form ceilings (fit bins 3..250 vs all bins w/ ridge) ---
    t0 = time.perf_counter()
    M, ref_metrics = closed_form_reference(loaders, bins=(3, 251), reg=None)
    print(
        f"[REF-A] shared map fit bins 3..250: test rl2 = {ref_metrics['relative_l2']:.3e} "
        f"(bins0-2 {ref_metrics['rl2_bins0_2']:.2e} | bins3-7 {ref_metrics['rl2_bins3_7']:.2e} | "
        f"bins8-250 {ref_metrics['rl2_bins8_250']:.2e})  ({time.perf_counter()-t0:.1f}s)"
    )
    print(
        f"[REF-A] target energy frac: bins0-2 {ref_metrics['e_frac_bins0_2']:.3e} | "
        f"bins3-7 {ref_metrics['e_frac_bins3_7']:.6f} | bins8-250 {ref_metrics['e_frac_bins8_250']:.3e}"
    )
    t0 = time.perf_counter()
    _, ref2 = closed_form_reference(loaders, bins=(0, 251), reg=1e-10)
    print(
        f"[REF-B] shared map fit ALL bins 0..250 (ridge 1e-10): test rl2 = {ref2['relative_l2']:.3e} "
        f"(bins0-2 {ref2['rl2_bins0_2']:.2e} | bins3-7 {ref2['rl2_bins3_7']:.2e} | "
        f"bins8-250 {ref2['rl2_bins8_250']:.2e})  ({time.perf_counter()-t0:.1f}s)"
    )
    del M

    # --- S-grid: scale / whiten x float64/32 x optimizer ---
    scale = train_channel_scale(loaders)
    whit = train_whiten(loaders)
    grid = [
        # (name, cls, pre, f64, opt, lr, epochs)
        ("S1  shared bare      f64 adamw 1e-3", SpectralSharedMap, "none", True, "adamw", 1e-3, epochs),
        ("S1n shared norm      f64 adamw 1e-2", SpectralSharedMap, "scale", True, "adamw", 1e-2, epochs),
        ("S1n shared norm      f64 sgd-m 1e-1", SpectralSharedMap, "scale", True, "sgd", 1e-1, epochs),
        ("S1w shared whiten    f64 adamw 1e-1", SpectralSharedMap, "whiten", True, "adamw", 1e-1, 30),
        ("S1w shared whiten    f64 adamw 1e-2", SpectralSharedMap, "whiten", True, "adamw", 1e-2, 30),
        ("S1w shared whiten    f64 sgd-m 1e-1", SpectralSharedMap, "whiten", True, "sgd", 1e-1, 30),
        ("S1w shared whiten    f32 adamw 1e-2", SpectralSharedMap, "whiten", False, "adamw", 1e-2, 30),
        ("S1w shared whiten    f64 lbfgs 1e-0", SpectralSharedMap, "whiten", True, "lbfgs", 1.0, 20),
    ]
    if smoke:
        grid = grid[:2]
    results = {}
    for name, cls, pre, f64, opt_name, lr, n_ep in grid:
        torch.manual_seed(SEED)
        dc, df = (torch.cdouble, torch.float64) if f64 else (torch.cfloat, torch.float32)
        model = cls(N_IN, N_OUT, N_BIN, dtype=dc) if cls is SpectralPerBin else cls(N_IN, N_OUT, dtype=dc)
        if pre == "scale":
            model.set_input_scale(scale)
        elif pre == "whiten":
            model.set_whiten(whit.to(dc))
        if opt_name == "adamw":
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        elif opt_name == "sgd":
            opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            opt = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=4, history_size=20)
        t0 = time.perf_counter()
        for ep in range(n_ep):
            if opt_name == "lbfgs":
                tr_loss = train_epoch_lbfgs(model, loaders["train"], opt, df)
            else:
                tr_loss = train_epoch(model, loaders["train"], opt, df)
            if ep % 10 == 9 or ep == n_ep - 1:
                val = official_rl2(model, loaders["val"])
                print(f"  [{name}] ep {ep+1:3d} train {tr_loss:.3e} val rl2 {val['relative_l2']:.3e}")
        test = official_rl2(model, loaders["test"])
        dt = time.perf_counter() - t0
        results[name] = test["relative_l2"]
        print(f"[{name}] FINAL official test rl2 = {test['relative_l2']:.3e} ({n_ep} ep, {dt:.1f}s, {dt/max(n_ep,1):.2f}s/ep)")

    print("\n=== SUMMARY ===")
    for name, v in results.items():
        print(f"  {name}: official test rl2 = {v:.3e}")


def train_whiten_perbin(loaders) -> torch.Tensor:
    """Per-bin per-pair whitening W_k, shape (251, 13, 13), frozen buffers.

    Global whitening (train_whiten) makes only the AGGREGATE pair covariance
    identity; the bin mixture (ramp bins 0-2 / tone bins 4-7 / noise bins
    8-250) leaves the batch covariance spread ~6e4 (round-3 check: diag
    5e-3..315).  Per-bin: each bin k whitens its own per-pair Gram
    G_k = sum_n x_nk^H x_nk / N (clip 1e-10), so EVERY bin's z_k = x_k W_k
    has covariance I and the batch Hessian is ~I (plus the target-energy
    weighting the loss applies).  Null directions (rank-deficient tone bins)
    collapse to ~0 variance; their map rows stay at init, harmless.
    """
    train = loaders["train"].dataset
    Ws = np.zeros((N_BIN, N_IN, N_IN), dtype=np.complex128)
    G = np.zeros((N_IN, N_IN), dtype=np.complex128)
    for i in range(len(train)):
        x = train[i]["input"].numpy().astype(np.float64)
        xft = np.fft.rfft(x, axis=0)
        for k in range(N_BIN):
            G = np.outer(xft[k].conj(), xft[k])
            evals, evecs = np.linalg.eigh(G / len(train))
            evals = np.clip(evals, 1e-10, None)
            Ws[k] = (evecs * (1.0 / np.sqrt(evals))) @ evecs.conj().T
    return torch.tensor(Ws, dtype=torch.cdouble)


class SpectralSharedMapPerBinW(_SpectralBase):
    """Shared 2x13 map over PER-BIN-whitened bins (26 trained params, cond~1)."""

    def __init__(self, in_channels: int, out_channels: int, dtype=torch.cdouble):
        super().__init__()
        scale = 1.0 / in_channels
        self.M = nn.Parameter(
            (scale * torch.rand(out_channels, in_channels, dtype=dtype))
        )

    def set_whiten_perbin(self, w: torch.Tensor) -> None:
        self.register_buffer("Wk", w.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 501, 13) -> (B, 501, 2)
        xft = torch.fft.rfft(self._norm(x), dim=1)  # (B, 251, 13) complex
        if hasattr(self, "Wk"):
            xft = torch.einsum("bki,ijk->bkj", xft, self.Wk.to(xft.dtype))
        out_ft = torch.einsum("bki,oi->bko", xft, self.M)  # (B, 251, 2)
        return torch.fft.irfft(out_ft, n=x.shape[1], dim=1)


class SpectralPerBinPerBinW(_SpectralBase):
    """P2d: per-bin maps over per-bin-whitened bins (6.5k trained params)."""

    def __init__(self, in_channels: int, out_channels: int, n_bin: int, dtype=torch.cdouble):
        super().__init__()
        scale = 1.0 / in_channels
        self.M = nn.Parameter(
            (scale * torch.rand(n_bin, out_channels, in_channels, dtype=dtype))
        )

    def set_whiten_perbin(self, w: torch.Tensor) -> None:
        self.register_buffer("Wk", w.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xft = torch.fft.rfft(self._norm(x), dim=1)  # (B, 251, 13)
        if hasattr(self, "Wk"):
            xft = torch.einsum("bki,ijk->bkj", xft, self.Wk.to(xft.dtype))
        out_ft = torch.einsum("bki,koj->bko", xft, self.M)
        return torch.fft.irfft(out_ft, n=x.shape[1], dim=1)


def run_spike4(epochs: int = 100, smoke: bool = False) -> None:
    """Round 4: per-bin whitening + shared / per-bin maps.

    Round-3 diagnosis: the ~8e-3 plateau is CONDITIONING (L-BFGS line-search-
    exact also stalls), not the F4 step floor; global whitening leaves the
    batch covariance spread ~6e4 because one matrix cannot whiten the
    ramp/tone/null bin mixture.  Per-bin whitening makes every bin's
    covariance I -> batch Hessian ~I: first-order should converge, L-BFGS
    should solve the 26-param shared quadratic in a few full-batch steps.
    """
    torch.manual_seed(SEED)
    loaders = make_loaders()
    print(f"[data] train={len(loaders['train'].dataset)} val={len(loaders['val'].dataset)} test={len(loaders['test'].dataset)}")

    wk = train_whiten_perbin(loaders)

    batch = next(iter(loaders["train"]))
    x = batch["input"].to(torch.float64)
    xft = torch.fft.rfft(x, dim=1)
    z = torch.einsum("bki,ijk->bkj", xft, wk)
    cov = torch.einsum("bki,bkj->bij", z.conj(), z).mean(dim=0)  # (13,13)
    print(f"[check] per-bin whitened z covariance: diag {cov.diagonal().real.min().item():.3e}..{cov.diagonal().real.max().item():.3e}, "
          f"off-diag abs max {cov.abs().fill_diagonal_(0).abs().max().item():.3e}")

    grid = [
        ("R4a sharedW f64 lbfgs", SpectralSharedMapPerBinW, "whiten", "lbfgs", 1.0, 30),
        ("R4b sharedW f64 sgd-m 1e-1", SpectralSharedMapPerBinW, "whiten", "sgd", 1e-1, epochs),
        ("R4c sharedW f64 adamw 1e-2 cosine", SpectralSharedMapPerBinW, "whiten", "adamw-cos", 1e-2, epochs),
        ("R4d perbinW f64 lbfgs", SpectralPerBinPerBinW, "whiten", "lbfgs", 1.0, 30),
        ("R4e perbinW f64 sgd-m 1e-1", SpectralPerBinPerBinW, "whiten", "sgd", 1e-1, epochs),
        ("R4f perbinW f64 adamw 1e-2 cosine", SpectralPerBinPerBinW, "whiten", "adamw-cos", 1e-2, epochs),
    ]
    if smoke:
        grid = grid[:2]
    results = {}
    for name, cls, pre, opt_name, lr, n_ep in grid:
        torch.manual_seed(SEED)
        dc, df = (torch.cdouble, torch.float64)
        model = cls(N_IN, N_OUT, N_BIN, dtype=dc) if cls is SpectralPerBinPerBinW else cls(N_IN, N_OUT, dtype=dc)
        model.set_whiten_perbin(wk)
        if opt_name == "adamw" or opt_name == "adamw-cos":
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep) if opt_name == "adamw-cos" else None
        elif opt_name == "sgd":
            opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            sched = None
        else:
            opt = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=4, history_size=20)
            sched = None
        t0 = time.perf_counter()
        for ep in range(n_ep):
            if opt_name == "lbfgs":
                tr_loss = train_epoch_lbfgs(model, loaders["train"], opt, df)
            else:
                tr_loss = train_epoch(model, loaders["train"], opt, df)
            if sched is not None:
                sched.step()
            if ep % 10 == 9 or ep == n_ep - 1:
                val = official_rl2(model, loaders["val"])
                print(f"  [{name}] ep {ep+1:3d} train {tr_loss:.3e} val rl2 {val['relative_l2']:.3e}")
        test = official_rl2(model, loaders["test"])
        dt = time.perf_counter() - t0
        results[name] = test["relative_l2"]
        print(f"[{name}] FINAL official test rl2 = {test['relative_l2']:.3e} ({n_ep} ep, {dt:.1f}s, {dt/max(n_ep,1):.2f}s/ep)")

    print("\n=== SUMMARY ===")
    for name, v in results.items():
        print(f"  {name}: official test rl2 = {v:.3e}")


def run_spike3(epochs: int = 100, smoke: bool = False) -> None:
    """Round 3: fixed whitening (per-pair identity) + L-BFGS on scale/whiten.

    Round-2 diagnosis: S1w whiten stalled because W lacked the sqrt(N_pairs)
    normalization (M* inflated ~1120x -> AdamW fixed-step F4 can't traverse).
    Round 3 tests: (a) L-BFGS on the scale-only problem — a 26-param convex
    quadratic, line search should handle residual conditioning; (b) L-BFGS on
    the corrected whitened problem (cond ~1, exact solve in a few iterations);
    (c) first-order paths on the corrected whitened problem: SGD-m, AdamW
    fixed-lr, AdamW + cosine decay — the "SGD 训练" leg of the gate.
    """
    torch.manual_seed(SEED)
    loaders = make_loaders()
    print(f"[data] train={len(loaders['train'].dataset)} val={len(loaders['val'].dataset)} test={len(loaders['test'].dataset)}")

    scale = train_channel_scale(loaders)
    whit = train_whiten(loaders)

    # sanity: per-pair whitened covariance should be ~I
    batch = next(iter(loaders["train"]))
    x = batch["input"].to(torch.float64)
    xft = torch.fft.rfft(x, dim=1)
    z = torch.einsum("bki,ij->bkj", xft, whit)
    cov = torch.einsum("bki,bkj->bij", z.conj(), z).mean(dim=0)  # (13,13)
    print(f"[check] per-pair whitened z covariance: diag {cov.diagonal().real.min().item():.3e}..{cov.diagonal().real.max().item():.3e}, "
          f"off-diag abs max {cov.abs().fill_diagonal_(0).abs().max().item():.3e}")

    grid = [
        ("R3a shared scale    f64 lbfgs", SpectralSharedMap, "scale", True, "lbfgs", 1.0, 50),
        ("R3b shared whiten   f64 lbfgs", SpectralSharedMap, "whiten", True, "lbfgs", 1.0, 50),
        ("R3c shared whiten   f64 sgd-m 1e-1", SpectralSharedMap, "whiten", True, "sgd", 1e-1, epochs),
        ("R3d shared whiten   f64 adamw 1e-2", SpectralSharedMap, "whiten", True, "adamw", 1e-2, epochs),
        ("R3e shared whiten   f64 adamw 1e-2 cosine", SpectralSharedMap, "whiten", True, "adamw-cos", 1e-2, epochs),
    ]
    if smoke:
        grid = grid[:2]
    results = {}
    for name, cls, pre, f64, opt_name, lr, n_ep in grid:
        torch.manual_seed(SEED)
        dc, df = (torch.cdouble, torch.float64) if f64 else (torch.cfloat, torch.float32)
        model = cls(N_IN, N_OUT, dtype=dc)
        if pre == "scale":
            model.set_input_scale(scale)
        elif pre == "whiten":
            model.set_whiten(whit.to(dc))
        if opt_name == "adamw" or opt_name == "adamw-cos":
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep) if opt_name == "adamw-cos" else None
        elif opt_name == "sgd":
            opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            sched = None
        else:
            opt = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=4, history_size=20)
            sched = None
        t0 = time.perf_counter()
        for ep in range(n_ep):
            if opt_name == "lbfgs":
                tr_loss = train_epoch_lbfgs(model, loaders["train"], opt, df)
            else:
                tr_loss = train_epoch(model, loaders["train"], opt, df)
            if sched is not None:
                sched.step()
            if ep % 10 == 9 or ep == n_ep - 1:
                val = official_rl2(model, loaders["val"])
                print(f"  [{name}] ep {ep+1:3d} train {tr_loss:.3e} val rl2 {val['relative_l2']:.3e}")
        test = official_rl2(model, loaders["test"])
        dt = time.perf_counter() - t0
        results[name] = test["relative_l2"]
        print(f"[{name}] FINAL official test rl2 = {test['relative_l2']:.3e} ({n_ep} ep, {dt:.1f}s, {dt/max(n_ep,1):.2f}s/ep)")

    print("\n=== SUMMARY ===")
    for name, v in results.items():
        print(f"  {name}: official test rl2 = {v:.3e}")


def run_spike5(epochs: int = 120, smoke: bool = False) -> None:
    """Round 5: SGD-m decay schedules on the fixed-whitened shared map.

    R3c (SGD-m lr 0.1, fixed) reached 1.81e-5 and plateaued with momentum
    noise 1.5-2.4e-5 (vs REF-B closed-form floor 4.6e-7).  Decay schedules
    push the noise floor down: step decay 0.1->0.01->0.001 and a hard drop
    after the conditioning cliff (~ep 45-50).
    """
    torch.manual_seed(SEED)
    loaders = make_loaders()
    print(f"[data] train={len(loaders['train'].dataset)} val={len(loaders['val'].dataset)} test={len(loaders['test'].dataset)}")
    whit = train_whiten(loaders)

    schedules = [
        ("S5a step 0.1->0.01->0.001", [(60, 1e-1), (40, 1e-2), (20, 1e-3)]),
        ("S5b drop 0.1->0.001@60", [(60, 1e-1), (40, 1e-3)]),
        ("S5c sgd 0.1 fixed 120", [(120, 1e-1)]),
    ]
    results = {}
    for name, sched in schedules:
        torch.manual_seed(SEED)
        model = SpectralSharedMap(N_IN, N_OUT, dtype=torch.cdouble)
        model.set_whiten(whit)
        opt = torch.optim.SGD(model.parameters(), lr=sched[0][1], momentum=0.9)
        t0 = time.perf_counter()
        step = 0
        for n_ep, lr in sched:
            for g in opt.param_groups:
                g["lr"] = lr
            for _ in range(n_ep):
                tr_loss = train_epoch(model, loaders["train"], opt, torch.float64)
                step += 1
                if step % 10 == 0:
                    val = official_rl2(model, loaders["val"])
                    print(f"  [{name}] ep {step:3d} train {tr_loss:.3e} val rl2 {val['relative_l2']:.3e}")
        test = official_rl2(model, loaders["test"])
        dt = time.perf_counter() - t0
        results[name] = test["relative_l2"]
        print(f"[{name}] FINAL official test rl2 = {test['relative_l2']:.3e} ({sum(n for n, _ in sched)} ep, {dt:.1f}s)")

    print("\n=== SUMMARY ===")
    for name, v in results.items():
        print(f"  {name}: official test rl2 = {v:.3e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--round", type=int, default=2)
    args = parser.parse_args()
    if args.round == 3:
        run_spike3(epochs=args.epochs, smoke=args.smoke)
    elif args.round == 5:
        run_spike5(epochs=args.epochs, smoke=args.smoke)
    elif args.round == 4:
        run_spike4(epochs=args.epochs, smoke=args.smoke)
    else:
        run_spike(epochs=args.epochs, smoke=args.smoke)

