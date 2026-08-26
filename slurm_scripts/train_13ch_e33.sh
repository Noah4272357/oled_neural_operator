#!/usr/bin/env bash
#SBATCH --job-name=oled_e33
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E33: b8 慢爬主链第五跳—— resume E31 best（test 2.03e-4），b8 / lr 5e-5，1200ep
# 慢爬链：E23(2.47e-4) → E31(2.03e-4)。5e-5 慢爬任意起点有效且收益持久
# （第四跳 -8% vs 2e-4 深预热 -1.4%）。E33 延续同配方。

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

echo "---- E33: b8 slow-chain hop5 from E31 best, lr 5e-5, 1200ep ----"
python scripts/train.py --output-dir "${OUTPUT_BASE}/b8_5e5_1200_h5" \
    --resume "outputs/job_30887/b8_chain_5e5_1200_h4/best_model.pt" \
    --input-fields force displacement velocity acceleration \
    --target-fields disturbance \
    --batch-size 8 --learning-rate 5e-5 --epochs 1200 \
    --amp none --no-memory-cache

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort

echo "End Time: $(date)"
echo "Job finished."
