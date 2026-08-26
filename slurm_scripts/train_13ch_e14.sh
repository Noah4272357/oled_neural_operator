#!/usr/bin/env bash
#SBATCH --job-name=oled_e14
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E14: b8 微调（步数效率武器）—— 从 E9 best 出发，b8 / lr 2.83e-4，500ep
# 动机：b64 每 epoch 仅 78 步，b8 有 625 步（8×）；微调瓶颈在更新步数。
# LR 2.83e-4 = 1e-4 × sqrt(64/8)（sqrt batch 缩放）。

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

echo "---- E14: b8 finetune from E9 best, lr 2.83e-4, 500ep ----"
python scripts/train.py --output-dir "${OUTPUT_BASE}/e14_b8_finetune_283" \
    --resume "outputs/job_30858/e9_finetune2/best_model.pt" \
    --input-fields force displacement velocity acceleration \
    --target-fields disturbance \
    --batch-size 8 --learning-rate 2.83e-4 --epochs 500 \
    --amp none --no-memory-cache

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort

echo "End Time: $(date)"
echo "Job finished."
