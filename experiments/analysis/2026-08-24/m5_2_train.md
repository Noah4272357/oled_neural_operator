# M5-2 主攻训练记录（P2：E31 续链强化，官方 test rl2 ≤ 1e-4）

> 日期：2026-08-24 | 任务：task-m5-2 | 前置：M5-1 判定 P1 死路（近似 LS 初始化 + 从头训练被 F4 机理 epoch1 内破坏）
> 评估档：`~/data/neural_operator_3` 正式档（5000/500/200，preroll 0.5s + substeps 5），官方 `scripts/evaluate.sh`（禁 TF32）
> 参照配方：E31（`outputs/job_30887/b8_chain_5e5_1200_h4`，13ch GELU FNO1d w128/m128/embed64/lift64/b1，b8/lr5e-5/wd1e-4 ×1200ep cosine eta_min 3e-6、seed 20260810、loss relative_l2、validate_every 20、amp none）

## Step 1: 档间漂移量化（E31 checkpoint 官方评估于 neural_operator_3）

- 命令：`scripts/evaluate.sh --checkpoint outputs/job_30887/b8_chain_5e5_1200_h4/best_model.pt --data-root ~/data/neural_operator_3`（job 30928）
- 结果：**官方 test rl2 = 2.0367e-4**（v2 档 2.0313e-4 → 相对漂移 +0.27%，同量级）
- 判定：档间漂移小 → **P2a 换档 resume**（E31 checkpoint + data.root=neural_operator_3，省 1h 从头）

## Step 2: P2a 主链（E31 同配方，换档 resume；P2a hop1 = 链跳 2）

- 实验：`configs/m52_p2a.yaml` + `slurm_scripts/train_m52_p2a_hop1.sh`（job 30929，node2）
- 轨迹（train rl2 / val rl2 关键点，epoch 从 resume 的 5940 起算）：
  - epoch 5941：train 6.77e-4（新档重适应）
  - epoch 5960：val 3.50e-4 / train 5.99e-4
  - epoch 6560：val 3.54e-4 / train 5.31e-4
  - epoch 6991：train 4.74e-4
  - **best val 2.145e-4 @ epoch 7060**；end train 4.71e-4
- **官方 test rl2（hop1）= 1.9622e-4**
- 判定：收益相对基线 = (2.0367e-4−1.9622e-4)/2.0367e-4 = **+3.6534% < 5%** 且 rl2 = 1.9622e-4 > 1.5e-4 → **续链衰竭确认，P2a 停止（不再发 hop2）**。与 v2 链上历史跳收益（−18%/−10%/−8%）的衰减趋势一致。

## Step 3: P2b 截断假说（modes 252，从零）

- 实验：`configs/m52_p2b.yaml` + `slurm_scripts/train_m52_p2b.sh`（job 30930，node1）
- 轨迹：epoch 1 train 1.000 → epoch 604 train 4.34e-2 → epoch 943 train 2.35e-2 → **best val 1.973e-2 @ epoch 1120**；end train 1.98e-2 @ epoch 1112
- **官方 test rl2 = 2.1845e-2**
- 判定：P2b ≫ P2a（**111.3×** 差，相对 P2a hop1 官方 test 1.9622e-4），**非**"P2b ≪ P2a（5× 以上差距）" → **截断非瓶颈，P2b 关闭**。注：m252 从零在 b8/lr5e-5×1200 配方下远未收敛，与 P2a 的链史（~6000ep）不可直接等比；但"升 252 带来 5× 以上收益"假说被直接否定，无需续链。

## Step 4: P2c 线性化续链（activation none，互调假说）

- 实验：`configs/m52_p2c.yaml` + `slurm_scripts/train_m52_p2c.sh`（job 30932，node1）
- 起点：P2a hop1 best_model.pt（epoch 7060），经 `scripts/analysis/remap_gelu_to_linear.py` 重排键（fc0.2/fc1.2/fc1.4 → fc0.1/fc1.1/fc1.2，形状一致、优化器 state 按索引对齐）后 resume
- 轨迹：切线性首 epoch train rl2 **16.7**（GELU→线性失配尖峰）→ 8ep 内恢复至 0.34 → epoch 515 train 8.4e-3 → epoch 8086 train 2.8e-3 → **best val 1.058e-3 @ epoch 8260（末）**；end train 2.35e-3
- **官方 test rl2 = 1.0881e-3**
- 判定：切线性后 rl2 **反升**（1.088e-3 vs P2a hop1 1.962e-4，5.5×） → **互调非主因（E31 权重已适配激活，切换引入失配）**，P2c 关闭。注：线性 m128 的收敛水平（~1e-3）仍优于 F1（线性 m128 从头 6.03e-3）与 P2b（2.18e-2），但与 GELU m128 平台（~2e-4）有系统性差距——与发现 6（精确线性映射需 modes=252）一致。

## Step 5: 判定门

| 实验 | config 摘要 | 官方 test rl2 | 判定 |
|---|---|---|---|
| Step1 档间漂移 | E31 ckpt @ v3 | 2.0367e-4 | 漂移小 → 换档 resume |
| P2a hop1 | m128 GELU，resume E31，1200ep | **1.9622e-4** | 增益 3.6534%<5% 且 >1.5e-4 → P2a 停止 |
| P2b | m252 GELU，从零，1200ep | **2.1845e-2** | ≫P2a → 截断非瓶颈，P2b 关闭 |
| P2c | m128 线性，resume hop1，1200ep | **1.0881e-3** | 反升 5.5× → 互调非主因，P2c 关闭 |

- 任一实验 ≤ 1e-4？**否**。全部 > 1e-4 且趋势衰竭（续链收益 3.6534%<5%、P2b/P2c 无突破）？
  **是** → **M5-2 FAIL** → 转 M5-3（训练后 joint-LTI 精修，lspinv 已就绪，预期 ~2.83e-6）
- 保留产物：P2a hop1 模型 `outputs/job_30929/m52_p2a_hop1/best_model.pt`（官方 1.9622e-4，学习路线当前最优）

## Concerns

1. **P2b/P2c 训练预算混淆**：P2b 从零 vs P2a 链史（~6000ep）不可等比；"截断非瓶颈"结论主要依赖 P2b 相对 P2a 的大幅劣化方向，未做同链深 m128-vs-m252 对照（GPU 预算内不可行）。
2. **P2c 仍处下降中**：P2c 在 eta_min 结束且 best 在末 epoch（1.058e-3 val），若续跳或可再降，但即使腰斩也远高于 1e-4 门；按规则直接关闭。
3. **P2a hop1 选择指标为 val**：best val 2.145e-4 @7060（val/test 缺口小，无过拟合信号）。
4. **跨档基线漂移**：E31 权重在 v3 上 val 重测 3.5e-4（v2 时 2.23e-4），test 漂移仅 0.27%——val/test 分布差异存在，但不影响各实验内部判定。
