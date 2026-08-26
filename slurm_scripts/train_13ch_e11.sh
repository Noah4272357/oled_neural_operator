#!/usr/bin/env bash
#SBATCH --job-name=oled_e11
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E11: 容量对照 —— w256/b1 从零 500ep（b64/lr2.83e-3）。
# 对照 E1（w128/b1，test 5.9e-4）：更宽 lift 是否改善高频精细映射。

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

echo "---- E11: w256/b1 from scratch ----"
python scripts/train.py --output-dir "${OUTPUT_BASE}/e11_w256" \
    --input-fields force displacement velocity acceleration \
    --target-fields disturbance --amp none --no-memory-cache \
    --batch-size 64 --learning-rate 2.83e-3 --epochs 500 \
    --width 256

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort
echo "End Time: $(date)"
echo "Job finished."
