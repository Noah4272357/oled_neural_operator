#!/usr/bin/env bash
#SBATCH --job-name=oled_e28
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E28: b8 链式第三跳（源头对比）—— resume E23 best（test 2.47e-4），b8 / lr 2e-4，800ep
# E26 从 E24 best 出发 lr2e-4；E28 从 E23 best（lr5e-5×1200ep 训练而来）出发
# 同 lr2e-4——验证第三跳收益是否依赖起点训练方式（E24 走深预热，E23 走慢爬）。

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

echo "---- E28: b8 chain hop3 from E23 best, lr 2e-4, 800ep ----"
python scripts/train.py --output-dir "${OUTPUT_BASE}/b8_chain_2e4_from_e23" \
    --resume "outputs/job_30878/b8_chain_5e5/best_model.pt" \
    --input-fields force displacement velocity acceleration \
    --target-fields disturbance \
    --batch-size 8 --learning-rate 2e-4 --epochs 800 \
    --amp none --no-memory-cache

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort

echo "End Time: $(date)"
echo "Job finished."
