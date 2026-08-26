#!/usr/bin/env bash
#SBATCH --job-name=oled_m5d1
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# M5-1 F4 机理诊断微实验（50ep）：LS 初始化 vs 随机初始化，从头训练协议。
# 依据（task-m5-1-brief.md Step 3）：13ch 线性 FNO modes=252 width=128
#   （activation=none，与 lspinv 注入维度一致），b8 lr5e-5 wd1e-4 cosine
#   从头调度，只跑 50ep（E31 慢爬协议切片）。
# 判定：LS 初值 ~1e-5 级（官方 2.83e-6）；若 50ep 内 val 恶化 >10x（>1e-4）
#   → F4 机理复现，P1 死路；否则 P1 可行。
# 注意：--resume 仅加载权重+优化器状态（epoch=0），调度器从头重建。

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

OUTPUT_BASE="outputs/job_${SLURM_JOB_ID}"

echo "---- Run 1: LS init (lspinv reg=1e-10) + train from scratch, 50ep ----"
scripts/train.sh --config configs/m5_linear_13ch_m252_diag.yaml \
    --output-dir "${OUTPUT_BASE}/m5_diag_lsinit" \
    --resume outputs/m5_ls_init/reg_1e-10/best_model.pt \
    --device cuda

echo "---- Run 2: random init control, 50ep ----"
scripts/train.sh --config configs/m5_linear_13ch_m252_diag.yaml \
    --output-dir "${OUTPUT_BASE}/m5_diag_rand" \
    --device cuda

echo "========== Run Artifacts =========="
find experiments -maxdepth 2 -name "history.json" -newermt "-3 hours" | sort
find outputs -maxdepth 3 -name "history.json" -newermt "-3 hours" | sort

echo "End Time: $(date)"
echo "Job finished."
