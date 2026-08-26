#!/usr/bin/env bash
#SBATCH --job-name=oled_m67w
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --output=logs/oled_m67w_%j.out
#SBATCH --error=logs/oled_m67w_%j.err

# m67h WD 系列变体（λ 由命令行注入，见 train_m67h_wd.sh 头注释的尺度推导）

set -euo pipefail
cd /nishome/charliewang/forge-projects/oled-neural-operator
source "$HOME/envs/base/bin/activate"
PY=python

echo "========== Job Info =========="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
$PY -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
echo "Start Time: $(date)"
echo "=============================="

name="m67h_wd_${WD_LAM}"
echo "==== [$(date)] ${name} (LBFGS f64, WD=${WD_LAM}, 2000ep, restart 50) ===="
$PY scripts/analysis/train_m67h_minibatch_sgd.py \
    --epochs 2000 --lr 1.0 --optimizer lbfgs --dtype float64 --device cuda \
    --restart-every 50 --weight-decay "${WD_LAM}" \
    --out "outputs/${name}" 2>&1
echo "---- summary ----"
cat "outputs/${name}/summary.json"
echo
echo "========== End Time: $(date) =========="
