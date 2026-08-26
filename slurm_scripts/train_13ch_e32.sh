#!/usr/bin/env bash
#SBATCH --job-name=oled_e32
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E32: b8 极低 LR 慢爬—— resume E30 best（test 2.18e-4），b8 / lr 2e-5，1500ep
# LR 递减规律检验：2e-4（深预热跳，递减快）→ 5e-5（慢爬，E23 追平）→ 2e-5？
# 目标区间 2.2e-4 → 1e-4 需要高精度精修，极低 LR 无预热浪费直接细磨。

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

echo "---- E32: b8 very-low-LR refine from E30 best, lr 2e-5, 1500ep ----"
python scripts/train.py --output-dir "${OUTPUT_BASE}/b8_2e5_1500" \
    --resume "outputs/job_30886/b8_chain_2e4_h4/best_model.pt" \
    --input-fields force displacement velocity acceleration \
    --target-fields disturbance \
    --batch-size 8 --learning-rate 2e-5 --epochs 1500 \
    --amp none --no-memory-cache

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort

echo "End Time: $(date)"
echo "Job finished."
