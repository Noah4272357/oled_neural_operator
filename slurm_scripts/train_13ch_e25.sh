#!/usr/bin/env bash
#SBATCH --job-name=oled_e25
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E25: b8 链式微调 —— resume outputs/job_30872/e19_finetune4/best_model.pt，b8 / lr 1e-4，800ep
# b8 = 625 步/epoch（b64 的 8 倍）；E15 证明 b8 微调 500k 步突破 b64 瓶颈（test 2.99e-4）。

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

echo "---- E25: b8 finetune from job_30872/e19_finetune4/best_model.pt, lr 1e-4, 800ep ----"
python scripts/train.py --output-dir "${OUTPUT_BASE}/b8_from_e19"     --resume "outputs/job_30872/e19_finetune4/best_model.pt"     --input-fields force displacement velocity acceleration     --target-fields disturbance     --batch-size 8 --learning-rate 1e-4 --epochs 800     --amp none --no-memory-cache

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort

echo "End Time: $(date)"
echo "Job finished."
