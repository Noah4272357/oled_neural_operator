#!/usr/bin/env bash
#SBATCH --job-name=oled_e35
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E35: b8 中等 LR 慢爬—— resume E31 best（test 2.03e-4），b8 / lr 1e-4，1200ep
# LR 空白点：5e-5（慢爬，-8%）与 2e-4（深预热，-1.4%）之间——1e-4×1200 未试。

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

echo "---- E35: b8 mid-LR refine from E31 best, lr 1e-4, 1200ep ----"
python scripts/train.py --output-dir "${OUTPUT_BASE}/b8_1e4_1200_h5" \
    --resume "outputs/job_30887/b8_chain_5e5_1200_h4/best_model.pt" \
    --input-fields force displacement velocity acceleration \
    --target-fields disturbance \
    --batch-size 8 --learning-rate 1e-4 --epochs 1200 \
    --amp none --no-memory-cache

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort

echo "End Time: $(date)"
echo "Job finished."
