#!/usr/bin/env bash
#SBATCH --job-name=oled_e18
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E18: 第三轮微调 LR 对照（5e-5，长跑）—— 从 E13 best 出发，b64 / lr 5e-5，1200ep
# LR 空间空白点：E9 用 3e-5（-11%）、E13 用 1e-4（-14%）；5e-5 × 1200ep 更长衰减。

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

echo "---- E18: 3rd-stage finetune from E13 best, lr 5e-5, 1200ep ----"
python scripts/train.py --output-dir "${OUTPUT_BASE}/e18_finetune_5e5" \
    --resume "outputs/job_30862/e13_finetune_1e4/best_model.pt" \
    --input-fields force displacement velocity acceleration \
    --target-fields disturbance \
    --learning-rate 5e-5 --epochs 1200 \
    --amp none --no-memory-cache

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort

echo "End Time: $(date)"
echo "Job finished."
