#!/usr/bin/env bash
#SBATCH --job-name=oled_cost_probe
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# oled-neural-operator 训练成本探针：全量数据 5 epoch（GPU）。
# 目的：评估每 epoch 训练成本（5000 样本 × 501 时间点，batch 8 → 625 步/epoch），
#       用于外推 500 epoch 完整训练的 GPU 时间与资源预算。
# 数据路径不显式设置：由 src/utils/paths.py 解析链自动定位（$DATA_ROOT > ~/data，回退时 stderr 警告——日志中可见证据）。
#   > ~/data/neural_operator_2，回退时 stderr 警告——日志中可见证据）。

set -euo pipefail

cd "$(dirname "$0")/.."

source "$HOME/envs/base/bin/activate"

export OUTPUT_DIR="outputs/job_${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT_DIR}"

echo "========== Job Info =========="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Node: ${SLURMD_NODENAME}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Python: $(which python)"
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
echo "Start Time: $(date)"
echo "=============================="

nvidia-smi || true

python scripts/train.py \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 5 \
  --device auto

echo "========== Run Artifacts =========="
find "${OUTPUT_DIR}" -maxdepth 2 -type f | sort || true

echo "End Time: $(date)"
echo "Job finished."
