#!/usr/bin/env bash
#SBATCH --job-name=oled_p1
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# P1 探针：13ch 线性 FNO（activation=none）modes=252 纯学习极限。
# 依据：算子精确线性（f_dist = M q_ddot - B u，rl2~1e-15）；modes=252 = 解析逆算子
#   注入（F5/ls_init_f1）所用全频谱，探针检验纯 SGD 学习在满模态下的极限。
# 判定：test relative_l2 < 1e-4 → 候选 B（线性全频谱纯学习）可行；
#       1e-4~1e-3 → 学习极限不足，依赖 P2/P3 生成侧改造；> 1e-3 → 强依赖生成侧。

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

echo "---- P1: linear FNO (activation=none), 13ch, modes=252, b64/lr2.83e-3/500ep, from scratch ----"
scripts/train.sh --config configs/p1_linear_13ch_m252.yaml --device cuda

echo "========== Run Artifacts =========="
find experiments -maxdepth 2 -name "metrics.csv" -newermt "-3 hours" | sort

echo "End Time: $(date)"
echo "Job finished."
