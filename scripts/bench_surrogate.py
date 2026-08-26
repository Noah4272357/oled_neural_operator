"""M4: single-sample surrogate inference wall-time benchmark.

Protocol mirrors the generation-side reference (oled-microstage-simulation
``scripts/bench_solver.py``): median of N repeats (default 5), reporting per-
repeat t_ms plus speedup against the P0 solver reference lines.

Input construction is identical to the official ``scripts/evaluate.py``: the
dataset is assembled from the checkpoint's stored config (13ch input =
force(4) + displacement(3) + velocity(3) + acceleration(3), concatenated in
that order on the last axis; 501 time points), loaded from the neural_operator_3
test split, and the model is invoked as ``model(inputs, grid)``.

Primary acceptance is CPU (project ``.venv``, CPU torch); the same script with
``--device cuda`` in the CUDA-capable base env produces the GPU reference
column.  Timing wraps a single forward pass (batch 1) -- the apples-to-apples
analogue of one solver run -- under ``eval`` mode and ``torch.no_grad``, with a
warmup phase before the measured repeats.  On CUDA, float32 matmul precision is
set to "highest" to match the official evaluation semantics (no TF32).

Usage::

    .venv/bin/python scripts/bench_surrogate.py \\
        --checkpoint outputs/ls_init_f1/best_model.pt --device cpu
    ~/envs/base/bin/python scripts/bench_surrogate.py \\
        --checkpoint experiments/fno1d_w64_b1_seed20260810_20260824-161144/best_model.pt \\
        --device cuda
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.dataloader import build_dataloaders, unwrap_dataset
from src.models.factory import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config

# P0 solver reference lines (generation side, CPU): see 战役报告 §9 / bench_solver.py.
T_EXUDYN_MS = 22.075
T_DOP853_MS = 818.548

DEFAULT_DATA_ROOT = "~/data/neural_operator_3"  # 正式档（M3 扩产）


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT).expanduser())
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def _full_precision() -> None:
    """Match official evaluation semantics: no TF32 on CUDA (validate.py)."""
    if torch.cuda.is_available() and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    config = copy.deepcopy(checkpoint.get("config") or {})
    if "data" not in config:
        raise SystemExit("checkpoint carries no parsed config; refuse to guess")
    config["data"]["root"] = str(args.data_root)
    config["device"] = args.device

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")

    # --- input assembly identical to scripts/evaluate.py ----------------------
    loaders = build_dataloaders(config["data"], int(config["seed"]))
    dataset = unwrap_dataset(loaders[args.split].dataset)
    model = build_model(
        config["model"], dataset.input_channels, dataset.target_channels
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sample = dataset[args.sample_index]
    x = sample["input"][None].to(device)   # [1, 501, C_in]
    g = sample["grid"][None].to(device)    # [1, 501]
    target = sample["target"]              # CPU, sanity only

    n_params = sum(p.numel() for p in model.parameters())
    info = {
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "device_name": (torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu"),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "model": {k: config["model"][k] for k in
                  ("modes", "width", "embed_dim", "lift_dim", "num_blocks", "activation")},
        "n_params": n_params,
        "input_channels": dataset.input_channels,
        "target_channels": dataset.target_channels,
        "points": int(x.shape[1]),
        "split": args.split,
        "sample_index": args.sample_index,
        "data_root": str(args.data_root),
    }
    print(f"[info] {json.dumps(info, sort_keys=True)}")

    # --- timed section: one forward pass per repeat, batch 1 ------------------
    _full_precision()
    with torch.no_grad():
        for _ in range(args.warmup):
            model(x, g)
        if args.device == "cuda":
            torch.cuda.synchronize()

        times: List[float] = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            with torch.no_grad():
                y = model(x, g)
            if args.device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1e3)
            if not torch.isfinite(y).all():
                raise FloatingPointError("non-finite surrogate output during benchmark")

    median_ms = statistics.median(times)
    # sanity: single-sample rl2 vs stored target (not the official aggregate metric)
    y_cpu = y.detach().cpu()
    sanity_rl2 = float(
        torch.sqrt(((y_cpu - target[None]) ** 2).sum() / (target ** 2).sum())
    )

    result = {
        "model": info,
        "repeats_ms": times,
        "median_ms": median_ms,
        "speedup_vs_exudyn": T_EXUDYN_MS / median_ms,
        "speedup_vs_dop853": T_DOP853_MS / median_ms,
        "t_exudyn_ms": T_EXUDYN_MS,
        "t_dop853_ms": T_DOP853_MS,
        "sanity_rl2_sample0": sanity_rl2,
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"[bench] repeats_ms={[f'{t:.4f}' for t in times]} median={median_ms:.4f} ms")
    print(
        f"[bench] speedup vs t_exudyn({T_EXUDYN_MS}ms) = {T_EXUDYN_MS / median_ms:.1f}x"
        f" | vs t_dop853({T_DOP853_MS}ms) = {T_DOP853_MS / median_ms:.1f}x"
    )
    print(f"[sanity] sample-0 rl2 = {sanity_rl2:.3e} (smoke check only)")


if __name__ == "__main__":
    main()
