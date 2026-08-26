#!/usr/bin/env bash
#SBATCH --job-name=oled_13ch
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# oled-neural-operator 最终目标战役：test relative_l2 <= 1e-4。
# 主路径 = 13ch 问题定义（force + displacement + velocity + acceleration -> disturbance）：
#   历史最优（w128/b1，500ep，b8/lr1e-3）test rl2 = 9.6e-4；趋势未收敛。
# E1  13ch b64/lr2.83e-3 500ep  —— 优化战役动力学（sqrt LR）直接应用
# E2  13ch b8/lr1e-3 500ep      —— 历史同语义复现（验证本地数据/代码一致性）

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

run_probe() {
    name=$1
    shift
    echo "---- probe: ${name} ----"
    python scripts/train.py --output-dir "${OUTPUT_BASE}/${name}" \
        --input-fields force displacement velocity acceleration \
        --target-fields disturbance --amp none --no-memory-cache "$@"
}

run_probe e1_13ch_b64_500ep --batch-size 64 --learning-rate 2.83e-3 --epochs 500
run_probe e2_13ch_b8_500ep   --batch-size 8  --learning-rate 1e-3     --epochs 500

echo "========== Run Artifacts =========="
find "${OUTPUT_BASE}" -maxdepth 2 -name "metrics.csv" | sort

echo "End Time: $(date)"
echo "Job finished."
