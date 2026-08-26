#!/usr/bin/env bash
#SBATCH --job-name=oled_e21
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E21: 冻结加块微调（结构性容量扩展，E20 的正确修正版）—— 从 E15 best 扩展
# num_blocks 1→2（e15_b2_scale01.pt：blocks[1] scale 0.1 初始化，optimizer 状态
# 已清理防参数组不匹配），--freeze fc0,blocks.0,fc1 只训新块 blocks.1。
# 动机：线性 LS 证明映射强非线性，1-block FNO 欠拟合尾部（train≈test）；
# E20 无冻结失败（新块随机输出与已训练 fc1 相互追赶），冻结 fc1 后新块只能学
# "fc1 输入流形上的残差修正"，保留已学表示、只增强非线性通道。

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

echo "---- E21: frozen blocks 1->2 finetune from e15_b2_scale01.pt, lr 1e-4, 800ep ----"
python scripts/train.py --output-dir "${OUTPUT_BASE}/e21_b2_frozen" \
    --resume "outputs/job_30864/e15_b2_scale01.pt" \
    --input-fields force displacement velocity acceleration \
    --target-fields disturbance \
    --learning-rate 1e-4 --epochs 800 \
    --amp none --no-memory-cache \
    --config configs/config_b2.yaml \
    --freeze "fc0,blocks.0,fc1"

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort

echo "End Time: $(date)"
echo "Job finished."
