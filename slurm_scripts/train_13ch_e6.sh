#!/usr/bin/env bash
#SBATCH --job-name=oled_e6
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E6: 容量实验 —— 13ch b64 500ep w128/**b2**（双 FNO block）。
# 对照 E1（w128/b1，test rl2 5.9e-4）：更大容量是否压掉高频伪影/相位误差。

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

echo "---- E6: w128/b2 from scratch ----"
python scripts/train.py --output-dir "${OUTPUT_BASE}/e6_b2" \
    --input-fields force displacement velocity acceleration \
    --target-fields disturbance --amp none --no-memory-cache \
    --batch-size 64 --learning-rate 2.83e-3 --epochs 500 \
    --num-blocks 2

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort

echo "End Time: $(date)"
echo "Job finished."
