#!/usr/bin/env bash
#SBATCH --job-name=oled_m3
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# M3 正式代理训练：13ch 线性 FNO（activation=none）width=64 modes=128。
# 依据（p3_decision.md）：形态 A 变体——算子精确线性（f_dist = M q_ddot - B u），
#   P2 子步净化后 modes=128 无截断根因、width 64 表示能力充足；训练数据 = 正式档
#   neural_operator_3（5000/500/200，preroll 0.5s + substeps 5）。
# 判定：test relative_l2 <= 1e-4 → 通过；> 1e-4 → 链式微调（低 lr 从 best 续训）；
#   仍不达标 → 回退 F5 解析注入保底（outputs/ls_init_f1/best_model.pt，2.84e-6）。

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

echo "---- M3: linear FNO (activation=none), 13ch, width=64, modes=128, b64/lr2.83e-3/500ep, data=neural_operator_3 ----"
scripts/train.sh --config configs/m3_linear_13ch_w64_m128.yaml --device cuda

echo "========== Run Artifacts =========="
find experiments -maxdepth 2 -name "metrics.csv" -newermt "-3 hours" | sort

echo "End Time: $(date)"
echo "Job finished."
