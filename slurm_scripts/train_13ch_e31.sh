#!/usr/bin/env bash
#SBATCH --job-name=oled_e31
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E31: b8 链式第四跳（慢爬精修）—— resume E26 best（test 2.21e-4），b8 / lr 5e-5，1200ep
# E23 证明 5e-5×1200 追平 2e-4×800（E24）：低 LR 无深预热、直接精修效率更高。
# E31 从第三跳最优出发慢爬，检验低 LR 长跑在低误差区间的边际收益。

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

echo "---- E31: b8 chain hop4 slow from E26 best, lr 5e-5, 1200ep ----"
python scripts/train.py --output-dir "${OUTPUT_BASE}/b8_chain_5e5_1200_h4" \
    --resume "outputs/job_30881/b8_chain_2e4_h3/best_model.pt" \
    --input-fields force displacement velocity acceleration \
    --target-fields disturbance \
    --batch-size 8 --learning-rate 5e-5 --epochs 1200 \
    --amp none --no-memory-cache

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort

echo "End Time: $(date)"
echo "Job finished."
