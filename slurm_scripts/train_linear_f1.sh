#!/usr/bin/env bash
#SBATCH --job-name=oled_f1
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# F1（主路径）：13ch 线性 FNO（activation=none）从头训练。
# 依据：算子精确线性（f_dist = M q_ddot - B u，rl2~1e-15）+ 数据集内蕴 LS 地板
#   13ch float32 管线地板 ~6e-5（reg 1e-8）→ 真实无正则训练可至 ~1e-6 级。
# 目标：test relative_l2 ≤ 1e-4（余量 1500×）。

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

echo "---- F1: linear FNO (activation=none), 13ch, b64/lr2.83e-3/500ep, from scratch ----"
python scripts/train.py --config configs/f1_linear_13ch.yaml \
    --output-dir "${OUTPUT_BASE}/linear_13ch_f1"

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort

echo "End Time: $(date)"
echo "Job finished."
