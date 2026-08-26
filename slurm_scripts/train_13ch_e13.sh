#!/usr/bin/env bash
#SBATCH --job-name=oled_e13
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E13: 第 2 轮微调 LR 对照 —— 从 E3' best，lr 1e-4（不降阶），800ep。
# 依据：E3'（1e-4, 800ep）降 45% vs E9（3e-5, 625ep）只降 14% —— LR 太低学习慢；
# 残差仍绝对小量，1e-4 级别微调可能继续有效。

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

echo "---- E13: finetune E3' best, lr 1e-4, 800ep ----"
python scripts/train.py --output-dir "${OUTPUT_BASE}/e13_finetune_1e4" \
    --resume "outputs/job_30853/e3p_finetune/best_model.pt" \
    --input-fields force displacement velocity acceleration \
    --target-fields disturbance \
    --learning-rate 1e-4 --epochs 800 \
    --amp none --no-memory-cache

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort
echo "End Time: $(date)"
echo "Job finished."
