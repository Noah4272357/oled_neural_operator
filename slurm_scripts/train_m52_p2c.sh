#!/usr/bin/env bash
#SBATCH --job-name=oled_m52p2c
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# M5-2 Step 4 P2c 线性化续链（互调假说）：E31 同配方但 activation=none（线性块），
#   resume P2a 第 1 跳 best_model.pt（权重形状相同，GELU→线性仅激活差异；键名
#   fc0.2/fc1.2/fc1.4 → fc0.1/fc1.1/fc1.2 由 remap_gelu_to_linear.py 重排后加载）。
#   1200ep b8/lr5e-5/wd1e-4 cosine eta_min 3e-6、seed 20260810、loss relative_l2。
# 假设：GELU 互调是 2e-4 平台主因 → 切线性后突破；若反升 → 互调非主因，记录关闭。

set -euo pipefail

cd "$(dirname "$0")/.."

source "$HOME/envs/base/bin/activate"

export PYTHON_BIN="$HOME/envs/base/bin/python"
export PYTHONUNBUFFERED=1

OUTPUT_BASE="outputs/job_${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_BASE}"

HOP1_CKPT="${HOP1_CKPT:-outputs/job_30929/m52_p2a_hop1/best_model.pt}"
REMAP_CKPT="${OUTPUT_BASE}/m52_p2c_remap/hop1_best_linear.pt"

echo "========== Job Info =========="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
"${PYTHON_BIN}" -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
echo "Start Time: $(date)"
echo "=============================="

echo "---- P2c: remap GELU hop1 checkpoint to linear key layout ----"
"${PYTHON_BIN}" scripts/analysis/remap_gelu_to_linear.py \
    --checkpoint "${HOP1_CKPT}" \
    --output "${REMAP_CKPT}"

echo "---- P2c: linear block resume (v3 data), 1200ep ----"
scripts/train.sh --config configs/m52_p2c.yaml \
    --resume "${REMAP_CKPT}" \
    --output-dir "${OUTPUT_BASE}/m52_p2c" \
    --device cuda

echo "---- Official eval on neural_operator_3 test ----"
scripts/evaluate.sh \
    --checkpoint "${OUTPUT_BASE}/m52_p2c/best_model.pt" \
    --data-root "$HOME/data/neural_operator_3" \
    --output "${OUTPUT_BASE}/m52_p2c/official_test_eval.json"

echo "========== Eval Result =========="
cat "${OUTPUT_BASE}/m52_p2c/official_test_eval.json"

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" -o -name "history.json" | sort

echo "End Time: $(date)"
echo "Job finished."
