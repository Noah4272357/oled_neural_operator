#!/usr/bin/env bash
#SBATCH --job-name=oled_e27
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E27: b8 链式第三跳（lr 上探）—— resume E24 best（test 2.46e-4），b8 / lr 3e-4，800ep
# E24 (lr2e-4, -18%) 优于 E22 (lr1e-4, -15%)——LR 越高效用越高？E14 (2.83e-4)
# 当年 100ep 误杀（预热期 val ~1e-3 被误判为失败）。E27 用 3e-4 验证 LR 上界。

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

echo "---- E27: b8 chain hop3 from E24 best, lr 3e-4, 800ep ----"
python scripts/train.py --output-dir "${OUTPUT_BASE}/b8_chain_3e4_h3" \
    --resume "outputs/job_30879/b8_chain_2e4/best_model.pt" \
    --input-fields force displacement velocity acceleration \
    --target-fields disturbance \
    --batch-size 8 --learning-rate 3e-4 --epochs 800 \
    --amp none --no-memory-cache

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort

echo "End Time: $(date)"
echo "Job finished."
