#!/usr/bin/env bash
#SBATCH --job-name=oled_e8b
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E8b: 微调 + 频谱 loss（DC 已参与惩罚）—— 从 E1 best，lr 1e-4，800ep。
# 权重：0-10Hz 0.5 / 10-18Hz 1.0 / 18-250Hz 5.0 —— 把梯度预算导向高频相对误差
# （rl2 由高频相对误差主导：残差绝对小量 vs 目标高频绝对能量小）。
# 对照 E3'（全局 rl2 微调，test 4.4e-4）。

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

echo "---- E8b: finetune + spectral loss (high-freq weight 5) ----"
python scripts/train.py --output-dir "${OUTPUT_BASE}/e8b_finetune_spectral" \
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
