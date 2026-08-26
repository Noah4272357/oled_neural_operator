#!/usr/bin/env bash
#SBATCH --job-name=oled_m61_8ch
#SBATCH --partition=GPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# M6 配方迁移到 8ch recommended problem（force 4 + encoder_displacement 4
#   -> disturbance 2）：纯谱共享映射 + per-pair 白化（whiten_8ch.pt）
#   + SGD-momentum 步进衰减（0.1x60 -> 0.01x40 -> 0.001x20，120ep）+ float64。
#   3 seeds 全量训练 + 官方 test 评估（禁 TF32），汇总输出打印。

set -euo pipefail

cd "$(dirname "$0")/.."

source "$HOME/envs/base/bin/activate"

export PYTHONUNBUFFERED=1

echo "========== Job Info =========="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
echo "Start Time: $(date)"
echo "=============================="

for SEED in 20260810 20260811 20260812; do
  echo "==== [$(date)] seed $SEED: train (120ep, cpu) ===="
  scripts/train.sh --config configs/m61_spectral_8ch.yaml \
      --seed "$SEED" --run-name "m61_8ch_s${SEED}" --device cpu
done

echo "==== official evaluation (test split, 200 samples) ===="
for RUN in experiments/m61_8ch_s20260810 experiments/m61_8ch_s20260811 experiments/m61_8ch_s20260812; do
  scripts/evaluate.sh --checkpoint "$RUN/best_model.pt" --output "$RUN/official_test.json"
  cat "$RUN/official_test.json"
  echo
done

echo "========== Run Artifacts =========="
find experiments -maxdepth 2 -name "official_test.json" -newermt "-3 hours" | sort

echo "End Time: $(date)"
echo "Job finished."
