#!/usr/bin/env bash
#SBATCH --job-name=oled_e5
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# E5: 频谱加权 loss 从零训练（500ep，T_max=500，b64/lr2.83e-3）。
# bands: 10-18Hz(目标频带) w=1.0 | 0-10Hz(力频带, 已饱和) w=0.3 | 18-250Hz(伪影带) w=2.0
# 对照 E1（同配置、全局 rl2）：压高频伪影是否带来 test rl2 提升。

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

echo "---- E5: spectral_rl2 from scratch ----"
python scripts/train.py --output-dir "${OUTPUT_BASE}/e5_spectral" \
    --input-fields force displacement velocity acceleration \
    --target-fields disturbance --amp none --no-memory-cache \
    --batch-size 64 --learning-rate 2.83e-3 --epochs 500 \
    --config configs/config_spectral.yaml

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort

echo "End Time: $(date)"
echo "Job finished."
