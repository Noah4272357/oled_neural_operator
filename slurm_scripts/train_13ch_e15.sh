#!/usr/bin/env bash
#SBATCH --job-name=oled_e15
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E15: b8 微调（低 LR 对照）—— 从 E9 best 出发，b8 / lr 1e-4，800ep
# 与 E14（lr 2.83e-4）对照：b8 下 LR 对微调收益的影响。
# 800ep × 625 步 = 500k 更新步（E3' 的 8 倍）。

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

echo "---- E15: b8 finetune from E9 best, lr 1e-4, 800ep ----"
python scripts/train.py --output-dir "${OUTPUT_BASE}/e15_b8_finetune_1e4" \
    --resume "outputs/job_30858/e9_finetune2/best_model.pt" \
    --input-fields force displacement velocity acceleration \
    --target-fields disturbance \
    --batch-size 8 --learning-rate 1e-4 --epochs 800 \
    --amp none --no-memory-cache

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort

echo "End Time: $(date)"
echo "Job finished."
