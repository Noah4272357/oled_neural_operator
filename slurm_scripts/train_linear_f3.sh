#!/usr/bin/env bash
#SBATCH --job-name=oled_f3
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# F3（理论对照）：8ch 原始输入 + 线性 FNO。
# 理论预测：漂移泄漏屏障（leakage 4.67x）下 per-mode 线性算子最优解收缩为零映射，
#   LS 地板 ~0.57 → 训练应复现 ~0.57 平台——端到端验证数据集内蕴分析。

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

echo "---- F3: linear FNO 8ch raw (control), b64/lr2.83e-3/500ep ----"
python scripts/train.py --config configs/f3_linear_8ch_raw.yaml \
    --output-dir "${OUTPUT_BASE}/linear_8ch_raw_f3"

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort

echo "End Time: $(date)"
echo "Job finished."
