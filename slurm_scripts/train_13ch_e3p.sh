#!/usr/bin/env bash
#SBATCH --job-name=oled_e3p
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E3': 从 E1 best 低 LR 微调（800ep，cosine 1e-4 -> 1e-5）。
# 依据：E3（T_max=1500）高 LR 阶段卡住（ep351 rl2 0.029 vs E1 同 epoch 0.00126）
# —— 扰动频带精细学习需要 LR < ~1e-3；E1 500ep 后 train 未收敛（0.00126），
# 低 LR 微调是针对性手段（残差结构：高频伪影 55% + 相位误差，均为精调对象）。

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

echo "---- E3': resume E1 best, fine-tune lr 1e-4, 800ep ----"
# NOTE: resume 时模型结构由当前 config 决定，必须与 checkpoint 的输入字段一致
# （checkpoint 为 13ch；默认 config 是 8ch recommended_problem）。
python scripts/train.py --output-dir "${OUTPUT_BASE}/e3p_finetune" \
    --resume "outputs/job_30848/e1_13ch_b64_500ep/best_model.pt" \
    --input-fields force displacement velocity acceleration \
    --target-fields disturbance \
    --learning-rate 1e-4 --epochs 800 \
    --amp none --no-memory-cache

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort

echo "End Time: $(date)"
echo "Job finished."
