#!/usr/bin/env bash
#SBATCH --job-name=oled_m52p2h2
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# M5-2 Step 2 P2a 主链第 2 跳（链跳 3）：E31 同配方（b8/lr5e-5/wd1e-4 ×1200ep
#   cosine eta_min 3e-6、seed 20260810、loss relative_l2、validate_every 20、
#   amp none），data.root = neural_operator_3，resume P2a 第 1 跳 best_model.pt。
# 跳末官方评估；收益 <5% 且 rl2>1.5e-4 → P2a 停止。

set -euo pipefail

cd "$(dirname "$0")/.."

source "$HOME/envs/base/bin/activate"

export PYTHON_BIN="$HOME/envs/base/bin/python"
export PYTHONUNBUFFERED=1

OUTPUT_BASE="outputs/job_${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_BASE}"

HOP1_CKPT="${HOP1_CKPT:-outputs/job_30929/m52_p2a_hop1/best_model.pt}"

echo "========== Job Info =========="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
"${PYTHON_BIN}" -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
echo "Start Time: $(date)"
echo "=============================="

echo "---- P2a hop2: E31 recipe resume hop1 (v3 data), 1200ep ----"
scripts/train.sh --config configs/m52_p2a.yaml \
    --resume "${HOP1_CKPT}" \
    --output-dir "${OUTPUT_BASE}/m52_p2a_hop2" \
    --device cuda

echo "---- Official eval on neural_operator_3 test ----"
scripts/evaluate.sh \
    --checkpoint "${OUTPUT_BASE}/m52_p2a_hop2/best_model.pt" \
    --data-root "$HOME/data/neural_operator_3" \
    --output "${OUTPUT_BASE}/m52_p2a_hop2/official_test_eval.json"

echo "========== Eval Result =========="
cat "${OUTPUT_BASE}/m52_p2a_hop2/official_test_eval.json"

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" -o -name "history.json" | sort

echo "End Time: $(date)"
echo "Job finished."
