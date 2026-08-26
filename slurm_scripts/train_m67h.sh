#!/usr/bin/env bash
#SBATCH --job-name=oled_m67h
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# m67h 系列：mini-batch 全谱线性残差头（修正评估口径，2026-08-26）。
#   根因链（§12 战役记录）：m67c-g full-batch 600 步欠收敛（凸问题
#   3.2M 参数需更长训练）+ 评估 reshape 实虚错位（已修正 per-bin 重建）。
#   配方：raw rfft bins 0-100 列缩放（可逆，凸最优=闭式 9.18e-5）+ 全局
#   目标缩放 + Linear(3232->1004, bias=False)（bias 使闭式恶化 16×）+
#   mini-batch 128（39 步/ep）+ 正确评估。
#   三变体串行（各 2000ep，GPU）：
#     sgd3   SGD-momentum lr 3e-3   （用户指定 SGD 路线，保守）
#     sgd10  SGD-momentum lr 1e-2   （更大步长）
#     adamw  AdamW lr 1e-3          （自适应，病态+噪声鲁棒）

set -euo pipefail

# SLURM 把脚本复制到 /var/spool/.../slurm_script，$0 定位失效——显式绝对路径
cd /nishome/charliewang/forge-projects/oled-neural-operator

# CUDA torch 在 $HOME/envs/base（M6 先例；项目 .venv 为 CPU torch，SLURM 节点无本地 GPU 库）
source "$HOME/envs/base/bin/activate"
PY=python

echo "========== Job Info =========="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
$PY -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
echo "Start Time: $(date)"
echo "=============================="

run_variant() {
    name=$1; opt=$2; lr=$3
    echo "==== [$(date)] $name ($opt lr=$lr, 2000ep) ===="
    $PY scripts/analysis/train_m67h_minibatch_sgd.py \
        --epochs 2000 --lr "$lr" --optimizer "$opt" --device cuda \
        --out "outputs/m67h_${name}" 2>&1
    echo "---- summary ----"
    cat "outputs/m67h_${name}/summary.json"
    echo
}

echo "==== [$(date)] lbfgs float64 + periodic restart (4000ep, history=150, restart 50) ===="
$PY scripts/analysis/train_m67h_minibatch_sgd.py \
    --epochs 4000 --lr 1.0 --optimizer lbfgs --dtype float64 --device cuda \
    --restart-every 50 --out outputs/m67h_lbfgs_f64_r50 2>&1
echo "---- summary ----"
cat outputs/m67h_lbfgs_f64_r50/summary.json
echo

echo "========== Job Info =========="
echo "End Time: $(date)"
echo "Job finished."
