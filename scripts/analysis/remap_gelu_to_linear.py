"""Remap a GELU FNO1d checkpoint into the activation="none" (linear) key layout.

M5-2 Step 4 (P2c) 互调假说：从 P2a 第 1 跳（GELU）checkpoint 切到线性块续训。
GELU 与线性变体仅激活函数不同：FNO1d 的 fc0/fc1 中 GELU 是 Functional
（无参数），但 nn.Sequential 的模块索引不同——GELU 变体 fc0 = [Linear, GELU,
Linear]（键 fc0.0/fc0.2），线性变体 fc0 = [Linear, Linear]（键 fc0.0/fc0.1）；
fc1 同理（fc1.0/fc1.2/fc1.4 → fc1.0/fc1.1/fc1.2）。权重形状一一对应，
本脚本仅按此映射重排 state_dict 键，并同步 checkpoint 内 config 的
model.activation = "none"。优化器 state 按参数索引对齐（两变体参数序相同），
无需改动；调度器在 resume 时本就重建。

用法:
    python scripts/analysis/remap_gelu_to_linear.py \
        --checkpoint <gelu_best_model.pt> --output <linear_best_model.pt>
"""

from __future__ import annotations

import argparse
import copy
import warnings
from pathlib import Path
from typing import Dict

import numpy
import torch

# v2 checkpoint 含 numpy RNG/优化器状态：weights_only=True 需显式允许
# numpy 数组重建类型（仅构造数组，不执行任意代码）。numpy 1.x/2.x 路径兼容。
_NUMPY_CORE = getattr(numpy, "_core", numpy.core)  # numpy>=2: _core; 1.x: core
_SAFE_GLOBALS = [
    _NUMPY_CORE.multiarray._reconstruct,
    numpy.ndarray,
    numpy.dtype,
]
if hasattr(numpy, "dtypes"):  # numpy>=2 的 dtype 子类
    _SAFE_GLOBALS.append(numpy.dtypes.UInt32DType)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    torch.serialization.add_safe_globals(_SAFE_GLOBALS)

# GELU-variant state_dict key prefix -> linear-variant key prefix
# (shape-preserving). Keys are like "fc0.2.weight" (module.index.suffix), so
# the mapping must match the dotted prefix "fc0.2.".
KEY_MAP: Dict[str, str] = {
    "fc0.2.": "fc0.1.",
    "fc1.2.": "fc1.1.",
    "fc1.4.": "fc1.2.",
}


def remap_state_dict(state_dict: Dict[str, object]) -> Dict[str, object]:
    new_state: Dict[str, object] = {}
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in KEY_MAP.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix) :]
                break
        new_state[new_key] = value
    return new_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_config = checkpoint["config"].get("model", {})
    if model_config.get("activation", "gelu") != "gelu":
        raise SystemExit(
            f"unexpected source activation: {model_config.get('activation')!r}"
        )
    model_config = dict(model_config)
    model_config["activation"] = "none"

    checkpoint = copy.deepcopy(checkpoint)
    checkpoint["model_state_dict"] = remap_state_dict(checkpoint["model_state_dict"])
    checkpoint["config"] = dict(checkpoint["config"])
    checkpoint["config"]["model"] = model_config

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    keys = sorted(checkpoint["model_state_dict"].keys())
    print(f"[remap] wrote {args.output} ({len(keys)} params)")
    print("[remap] keys:", keys)


if __name__ == "__main__":
    main()
