#!/usr/bin/env bash
#SBATCH --job-name=oled_e26
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E26: b8 链式第三跳（同配方）—— resume E24 best（test 2.46e-4），b8 / lr 2e-4，800ep
# 链式：E15(2.99e-4) → E24(lr2e-4, 2.46e-4, -18%)。E24 证明 b8+lr2e-4 在
# 600ep 后段爆发（前 400ep val ~1e-3 是预热，不可早判）。E26 同配方再跳。

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

echo "---- E26: b8 chain hop3 from E24 best, lr 2e-4, 800ep ----"
python scripts/train.py --output-dir "${OUTPUT_BASE}/b8_chain_2e4_h3" \
    --resume "outputs/job_30879/b8_chain_2e4/best_model.pt" \
    --input-fields force displacement velocity acceleration \
    --target-fields disturbance \
    --batch-size 8 --learning-rate 2e-4 --epochs 800 \
    --amp none --no-memory-cache

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort

echo "End Time: $(date)"
echo "Job finished."
