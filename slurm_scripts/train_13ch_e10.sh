#!/usr/bin/env bash
#SBATCH --job-name=oled_e10
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E10: 对照实验 —— 从 E1 best 一次到位微调 lr 3e-5 2000ep（cosine -> 1e-6）。
# 与 E3'（1e-4 800ep）+ E9（3e-5 1200ep）分段方案步数相当（2000 vs 2000），
# 验证"分段降低 LR" vs "一次到位低 LR"哪个更优。

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

echo "---- E10: one-shot finetune E1 best, lr 3e-5, 2000ep ----"
python scripts/train.py --output-dir "${OUTPUT_BASE}/e10_finetune1shot" \
    --resume "outputs/job_30848/e1_13ch_b64_500ep/best_model.pt" \
    --input-fields force displacement velocity acceleration \
    --target-fields disturbance \
    --learning-rate 3e-5 --epochs 2000 \
    --amp none --no-memory-cache

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort
echo "End Time: $(date)"
echo "Job finished."
