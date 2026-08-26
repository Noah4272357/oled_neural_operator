#!/usr/bin/env bash
#SBATCH --job-name=oled_e20
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E20: 加块微调（非线性容量扩展）—— 从 E13 best 扩展 num_blocks 1→2，lr 1e-4，800ep
# 动机：线性 LS 证明映射强非线性（线性只到 7-19%，FNO 到 0.02%）；1-block FNO
# 欠拟合尾部非线性结构（train≈test 高频 3.8%）。新增 block1（小随机 0.01 scale）
# 增强非线性表达，保留已学 lift/blocks[0]/project。

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

echo "---- E20: blocks 1->2 finetune from E13 best, lr 1e-4, 800ep ----"
python scripts/train.py --output-dir "${OUTPUT_BASE}/e20_b2_finetune" \
    --resume "outputs/job_30862/e13_b2_expanded.pt" \
    --input-fields force displacement velocity acceleration \
    --target-fields disturbance \
    --learning-rate 1e-4 --epochs 800 \
    --amp none --no-memory-cache \
    --config configs/config_b2.yaml

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort

echo "End Time: $(date)"
echo "Job finished."
