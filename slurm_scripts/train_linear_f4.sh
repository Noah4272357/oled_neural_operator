#!/usr/bin/env bash
#SBATCH --job-name=oled_f4
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# F4: 解析逆算子注入（零训练 test rl2=3.08e-6）后的稳定性验证——
# resume ls_init_f1 微调 300ep（lr 1e-5），验证 init 是 LS 最优、SGD 不破坏它。
# 主路径结果（analysis）：test.relative_l2 = 3.078e-6 << 1e-4 ✅

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

echo "---- F4: fine-tune from analytic-inverse init, lr 1e-5, 300ep ----"
python scripts/train.py --output-dir "${OUTPUT_BASE}/analytic_ft_1e5_300" \
    --resume "outputs/ls_init_f1/best_model.pt" \
    --config configs/f4_analytic_13ch_fullspec.yaml \
    --learning-rate 1e-5 --epochs 300 \
    --amp none --no-memory-cache

echo "---- F4: official test evaluation ----"
python scripts/evaluate.py --checkpoint "${OUTPUT_BASE}/analytic_ft_1e5_300/best_model.pt"

echo "End Time: $(date)"
echo "Job finished."
