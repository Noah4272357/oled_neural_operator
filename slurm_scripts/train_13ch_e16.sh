#!/usr/bin/env bash
#SBATCH --job-name=oled_e16
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E16: 分带相对 L2 loss（高频均衡梯度）—— 从 E13 best 出发，b64 / lr 1e-4，800ep
# 动机：残差分解显示 18-250Hz per-band rl2 = 4.1%（占残差能量 47.7%），
# 但该频带 target 能量仅占 0.004%——全局 relative_l2 下高频梯度极小，
# 结构性欠拟合。BandRelativeL2 按各频带自身 target 能量归一化，均衡梯度分配。
# bands: 0-10Hz w=1 / 10-18Hz w=1 / 18-250Hz w=2（高频略放大）。

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

echo "---- E16: band_rl2 finetune from E13 best, lr 1e-4, 800ep ----"
python scripts/train.py --output-dir "${OUTPUT_BASE}/e16_band_finetune" \
    --resume "outputs/job_30862/e13_finetune_1e4/best_model.pt" \
    --input-fields force displacement velocity acceleration \
    --target-fields disturbance \
    --learning-rate 1e-4 --epochs 800 \
    --amp none --no-memory-cache \
    --config configs/config_band.yaml

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort

echo "End Time: $(date)"
echo "Job finished."
