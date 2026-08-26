#!/usr/bin/env bash
#SBATCH --job-name=oled_m67w
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# m67h WD 系列（2026-08-26）：权重衰减 LBFGS——线性家族最后一个可训练
#   正则化。闭式谱族（LS/硬截断/Tikhonov/输入 bin 截断）测试 rl2 全部
#   ≥ 1.52e-3（劣于 head-only 基线 1.52e-3），而 LBFGS 训练轨迹 1.05e-4
#   （31057）远优于一切闭式解——轨迹的正则化由动力学决定，不可被任何
#   谱滤波器复制。WD 通过改变轨迹（吸引子）压低 test 谷值。
#   尺度：训练损失 = mse/n + 0.5*lambda_wd*||W'||^2（W'=s*W，s=23.65，
#   n=5000）→ 对应闭式 Tikhonov lambda_tik = 0.5*lambda_wd*n*s^2。
#   lambda_wd=1e-7 ≈ lambda_tik=0.14（弱）；1e-5 ≈ 14（强）。

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

run_wd() {
    lam=$1
    name="m67h_wd_${lam}"
    echo "==== [$(date)] $name (LBFGS f64, WD=${lam}, 2000ep, restart 50) ===="
    $PY scripts/analysis/train_m67h_minibatch_sgd.py \
        --epochs 2000 --lr 1.0 --optimizer lbfgs --dtype float64 --device cuda \
        --restart-every 50 --weight-decay "$lam" \
        --out "outputs/${name}" 2>&1
    echo "---- summary ----"
    cat "outputs/${name}/summary.json"
    echo
}

run_wd 1e-7
run_wd 1e-6
run_wd 1e-5

echo "========== Job Info =========="
echo "End Time: $(date)"
echo "Job finished."
