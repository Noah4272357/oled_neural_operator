#!/usr/bin/env bash
#SBATCH --job-name=oled_e19
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E19: 第四轮分段微调 —— 从 E17 best 出发，b64 / lr 1e-4，800ep
# 链式：E13(2nd) → E17(3rd) → E19(4th)。验证分段微调收益曲线。

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

echo "---- E19: 4th-stage finetune from E17 best, lr 1e-4, 800ep ----"
python scripts/train.py --output-dir "${OUTPUT_BASE}/e19_finetune4" \
    --resume "outputs/job_30867/e17_finetune3/best_model.pt" \
    --input-fields force displacement velocity acceleration \
    --target-fields disturbance \
    --learning-rate 1e-4 --epochs 800 \
    --amp none --no-memory-cache

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort

echo "End Time: $(date)"
echo "Job finished."
