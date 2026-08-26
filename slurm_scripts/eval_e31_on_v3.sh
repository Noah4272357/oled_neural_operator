#!/usr/bin/env bash
#SBATCH --job-name=oled_m52_e31v3
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# M5-2 Step 1: 档间漂移量化——官方评估 E31 checkpoint 于 neural_operator_3 正式档。
# 判定：新档 test ≈ 2e-4 → 换档 resume；> 5e-4 → 从零。

set -euo pipefail

cd "$(dirname "$0")/.."

source "$HOME/envs/base/bin/activate"

export PYTHON_BIN="$HOME/envs/base/bin/python"
export PYTHONUNBUFFERED=1

echo "========== Job Info =========="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
"${PYTHON_BIN}" -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
echo "Start Time: $(date)"
echo "=============================="

echo "---- M5-2 Step1: E31 checkpoint on neural_operator_3 test ----"
scripts/evaluate.sh \
    --checkpoint outputs/job_30887/b8_chain_5e5_1200_h4/best_model.pt \
    --data-root "$HOME/data/neural_operator_3" \
    --output "outputs/job_${SLURM_JOB_ID}/e31_v3_eval.json"

echo "========== Eval Result =========="
cat "outputs/job_${SLURM_JOB_ID}/e31_v3_eval.json" 2>/dev/null || true

echo "End Time: $(date)"
echo "Job finished."
