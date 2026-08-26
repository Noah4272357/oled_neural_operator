#!/usr/bin/env bash
#SBATCH --job-name=oled_e8
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E8: 微调 + 频谱 loss 组合 —— 从 E1 best 出发，lr 1e-4，800ep，spectral_rl2 loss
# （10-18Hz w=1 / 0-10Hz w=0.3 / 18-250Hz w=2）。
# 针对残差结构：高频伪影占残差能量 55% + 目标频带相位误差。

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

echo "---- E8: finetune E1 best + spectral_rl2 ----"
python scripts/train.py --output-dir "${OUTPUT_BASE}/e8_finetune_spectral" \
    --resume "outputs/job_30848/e1_13ch_b64_500ep/best_model.pt" \
    --input-fields force displacement velocity acceleration \
    --target-fields disturbance \
    --learning-rate 1e-4 --epochs 800 \
    --amp none --no-memory-cache \
    --config configs/config_spectral.yaml

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort

echo "End Time: $(date)"
echo "Job finished."
