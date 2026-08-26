#!/usr/bin/env bash
#SBATCH --job-name=oled_m3c2
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# M3 判定链第 2 层：链式微调跳 2（慢爬续跳）—— b8 / lr 5e-5 / 1200ep，从跳 1 best
#   （test 2.61e-3）续训。配方同跳 1：E28 链式动力学匹配原则（慢爬权重接慢爬）。
#   跳 1 收益 -57%（6.14e-3 → 2.61e-3）——线性模型对慢爬响应优于非线性 E 系列。

set -euo pipefail

cd "$(dirname "$0")/.."

source "$HOME/envs/base/bin/activate"

export PYTHONUNBUFFERED=1

echo "========== Job Info =========="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
echo "Start Time: $(date)"
echo "=============================="

echo "---- M3 chain hop2: b8 lr5e-5 1200ep from hop1 best (test 2.61e-3) ----"
scripts/train.sh --config configs/m3_linear_13ch_w64_m128_chain1.yaml \
    --resume experiments/fno1d_w64_b1_seed20260810_20260824-162825/best_model.pt \
    --device cuda

echo "========== Run Artifacts =========="
find experiments -maxdepth 2 -name "metrics.csv" -newermt "-3 hours" | sort

echo "End Time: $(date)"
echo "Job finished."
