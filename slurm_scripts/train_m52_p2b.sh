#!/usr/bin/env bash
#SBATCH --job-name=oled_m52p2b
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# M5-2 Step 3 P2b 截断假说：E31 同配方但 modes 252，新档 neural_operator_3 从零，
#   1200ep（b8/lr5e-5/wd1e-4 cosine eta_min 3e-6、seed 20260810、loss relative_l2）。
# 与 P2a（modes 128）对照 → 判定 GELU 结构下 modes 截断是否限制 ~2e-4。
# 若 P2b ≪ P2a（5× 以上差距）→ 截断主因，P2b 续链；否则截断非瓶颈，P2b 关闭。

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

echo "---- P2b: modes 252 from scratch (v3 data), 1200ep ----"
scripts/train.sh --config configs/m52_p2b.yaml \
    --output-dir "${OUTPUT_BASE}/m52_p2b" \
    --device cuda

echo "---- Official eval on neural_operator_3 test ----"
scripts/evaluate.sh \
    --checkpoint "${OUTPUT_BASE}/m52_p2b/best_model.pt" \
    --data-root "$HOME/data/neural_operator_3" \
    --output "${OUTPUT_BASE}/m52_p2b/official_test_eval.json"

echo "========== Eval Result =========="
cat "${OUTPUT_BASE}/m52_p2b/official_test_eval.json"

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" -o -name "history.json" | sort

echo "End Time: $(date)"
echo "Job finished."
