#!/usr/bin/env bash
#SBATCH --job-name=oled_f2
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# F2（证据实验）：8ch canonical + second_derivative 预处理 + 线性 FNO。
# 理论预测：float32 数据量化（s ulp~6e-11）经二次微分放大（/h^2）→
#   LS 地板 ~2.5e-3，不可达 1e-4——验证量化限制分析；同时验证预处理管线端到端。

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

echo "---- F2: linear FNO 8ch + second_derivative preprocessing, b64/lr2.83e-3/500ep ----"
python scripts/train.py --config configs/f2_linear_8ch_diff.yaml \
    --output-dir "${OUTPUT_BASE}/linear_8ch_diff_f2"

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort

echo "End Time: $(date)"
echo "Job finished."
