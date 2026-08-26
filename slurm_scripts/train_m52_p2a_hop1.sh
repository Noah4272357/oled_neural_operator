#!/usr/bin/env bash
#SBATCH --job-name=oled_m52p2a
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# M5-2 Step 2 P2a 主链第 1 跳（链跳 2）：
#   E31 同配方（b8/lr5e-5/wd1e-4 ×1200ep cosine eta_min 3e-6、seed 20260810、
#   loss relative_l2、validate_every 20、amp none），data.root = neural_operator_3，
#   换档 resume E31 best_model.pt（Step 1 判定：v3 官方 test 2.0367e-4 ≈ v2 2.0313e-4，档间漂移小）。
# 跳末官方评估（neural_operator_3 test）→ 收益相对基线 2.0367e-4 <5% 且 rl2>1.5e-4 即停 P2a。

set -euo pipefail

cd "$(dirname "$0")/.."

source "$HOME/envs/base/bin/activate"

export PYTHON_BIN="$HOME/envs/base/bin/python"
export PYTHONUNBUFFERED=1

OUTPUT_BASE="outputs/job_${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_BASE}"

echo "========== Job Info =========="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
"${PYTHON_BIN}" -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
echo "Start Time: $(date)"
echo "=============================="

echo "---- P2a hop1: E31 recipe resume (v3 data), 1200ep ----"
scripts/train.sh --config configs/m52_p2a.yaml \
    --resume outputs/job_30887/b8_chain_5e5_1200_h4/best_model.pt \
    --output-dir "${OUTPUT_BASE}/m52_p2a_hop1" \
    --device cuda

echo "---- Official eval on neural_operator_3 test ----"
scripts/evaluate.sh \
    --checkpoint "${OUTPUT_BASE}/m52_p2a_hop1/best_model.pt" \
    --data-root "$HOME/data/neural_operator_3" \
    --output "${OUTPUT_BASE}/m52_p2a_hop1/official_test_eval.json"

echo "========== Eval Result =========="
cat "${OUTPUT_BASE}/m52_p2a_hop1/official_test_eval.json"

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" -o -name "history.json" | sort

echo "End Time: $(date)"
echo "Job finished."
