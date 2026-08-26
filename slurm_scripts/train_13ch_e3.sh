#!/usr/bin/env bash
#SBATCH --job-name=oled_e3
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E3: 13ch b64 1500-epoch 长训练（cosine T_max=1500）。
# 依据：E1 (500ep) test rl2 0.000592（evaluate.py 全精度），train 曲线未收敛；
# b64 每 epoch 权重更新数 = b8 的 1/8，需要更长训练逼近 1e-4 目标。

set -euo pipefail

cd "$(dirname "$0")/.."

source "$HOME/envs/base/bin/activate"

export OUTPUT_BASE="outputs/job_${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT_BASE}"

echo "========== Job Info =========="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
echo "Start Time: $(date)"
echo "=============================="

echo "---- E3: 13ch b64 1500ep ----"
python scripts/train.py --output-dir "${OUTPUT_BASE}/e3_13ch_b64_1500ep" \
    --input-fields force displacement velocity acceleration \
    --target-fields disturbance --amp none --no-memory-cache \
    --batch-size 64 --learning-rate 2.83e-3 --epochs 1500

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort

echo "End Time: $(date)"
echo "Job finished."
