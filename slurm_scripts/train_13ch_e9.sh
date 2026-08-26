#!/usr/bin/env bash
#SBATCH --job-name=oled_e9
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E9: 主线微调第 2 轮 —— 从 E3' best 出发，lr 3e-5，1200ep（cosine -> 1e-6）。
# E3' 微调 800ep 均匀削减各频带残差 45%（0.00059 -> 0.00044），机制有效，继续。

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

echo "---- E9: finetune E3' best, lr 3e-5, 1200ep ----"
python scripts/train.py --output-dir "${OUTPUT_BASE}/e9_finetune2" \
    --resume "outputs/job_30853/e3p_finetune/best_model.pt" \
    --input-fields force displacement velocity acceleration \
    --target-fields disturbance \
    --learning-rate 3e-5 --epochs 1200 \
    --amp none --no-memory-cache

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort
echo "End Time: $(date)"
echo "Job finished."
