#!/usr/bin/env bash
#SBATCH --job-name=oled_opt_abl
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# oled-neural-operator 训练成本优化对照实验（全量数据，各 10 epoch）。
# R1  基线语义（无缓存、FP32）          —— vs 历史 30842（3.1 s/epoch，同步 metrics）
# R2  实现+硬件（整表缓存 + AMP-BF16）  —— 语义保持层净收益
# R3  R2 + batch32/lr2e-3               —— 动力学层（sqrt 缩放）
# R4  R2 + batch64/lr2.8e-3             —— 动力学层（sqrt 缩放）
# R5  R2 + torch.compile                —— 编译探索（首 epoch 含编译时间）
# 评估保持 FP32（validate 不 autocast），指标跨组可比。

set -euo pipefail

cd "$(dirname "$0")/.."

source "$HOME/envs/base/bin/activate"

export OUTPUT_BASE="outputs/job_${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT_BASE}"

echo "========== Job Info =========="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
echo "Start Time: $(date)"
echo "=============================="

run_probe() {
    name=$1
    shift
    echo "---- probe: ${name} ----"
    python scripts/train.py --output-dir "${OUTPUT_BASE}/${name}" --epochs 10 --device auto "$@"
}

# 100-epoch dynamics validation: batch64+lr2.83e-3 vs batch8+lr1e-3 (fp32).
run_probe r8_b64_100ep      --amp none --batch-size 64 --learning-rate 2.83e-3 --epochs 100
run_probe r10_b8_100ep_ref  --amp none --epochs 100
# r5 torch.compile: NEGATIVE — envs/base lacks python3-dev (no Python.h for
# Triton) and inductor lacks codegen for complex weights; falls back to eager.

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort

echo "End Time: $(date)"
echo "Job finished."
