# M5-1: ridge-LS 初始化构造 + F4 机理诊断（代理侧，诊断先行）

> 日期：2026-08-24 | 战役：`2026-08-24-train-to-1e-4`（M5）| 任务：task-m5-1
> 数据：`~/data/neural_operator_3`（正式档 5000/500/200，preroll 0.5s + substeps 5）
> 环境：分析 `.venv`（CPU，torch 2.5.1）；训练 base env（CUDA 12.1，RTX 4090，SLURM）

## 结论先行（P1 判定）

**P1（近似 LS 初始化 + 从头训练）判定：死路。F4 机理复现。**

- LS 初始化质量（官方协议，禁 TF32）：**test rl2 = 2.831e-6**（reg=1e-12/1e-10 相同；reg=1e-8 → 2.886e-6）——优于任务预期（~1e-5 级），与解析注入（F5，官方 2.84e-6）同量级
- F4 微实验（50ep，b8 lr5e-5 wd1e-4 cosine 从头调度）：**epoch 1 内 train rl2 从 2.5e-6 恶化到 0.238**（Adam 一步即摧毁近精确最优）；50ep 后 val rl2 = 9.96e-3（test 9.63e-3），相对初值恶化 **3.5e3×**（判定阈值 >10×）
- 对照随机初始化：50ep 后 val = 3.35e-2（test 3.36e-2）——LS 初始化轨迹仍快 3.5×（初始化盆地残留），但按判定标准（初值恶化 >10× → F4 复现 → P1 死路），P1 关闭
- M5 主攻路径应转向 **P2（E31 续链强化）**；P3（训练后 LS 精修）仍可作保底

## 初始化质量表（reg 网格 × 官方 test rl2 × bins1-2 条件数）

数据：joint frequency-shared 拟合（见下），n_train=5000，注入 modes=252 线性 FNO（w128，13ch）。

| reg | 官方 test rl2 | 每 bin 自评总 rl2（未 pad 基） | bins1-2 每 bin rl2 | bins1-2 Gram cond |
|---|---|---|---|---|
| 1e-12 | **2.831e-6** | 2.26e-11 | 1.76e-11 / 1.41e-11 | 1.396e+28 / 3.208e+24 |
| 1e-10 | **2.831e-6** | 7.37e-11 | 7.21e-11 / 1.53e-11 | 1.396e+28 / 3.208e+24 |
| 1e-8  | 2.886e-6 | 3.06e-10 | 4.2e-11 / 3.0e-10 | 1.396e+28 / 3.208e+24 |

- 全带条件数：median 1.351e+26，max 2.546e+34（未 pad 501 点 rfft 基，float64）
- 三个 reg 在 joint 问题上数值惰性（reg/λmax ~ 1e-19..1e-23），质量 reg 无关——按预期如实记录
- 选中 reg=1e-10 作为微实验初始化（`outputs/m5_ls_init/reg_1e-10/best_model.pt`）

## 条件数诊断（LS 地板疑案，关键发现）

任务预期"reg=1e-10 质量接近 4.83e-6 地板"——诊断结论：**地板成立（4.83e-6），但 per-bin 协议不可辨识，机制以训练子集不稳定性呈现，而非评估子集伪影**（2026-08-24 评审独立重算后修正本节的错误断言）：

1. **地板成立，非评估子集伪影**：`per_mode_ls_floor` 的 4.83e-6 在 50 样本 stride-4 子集与全 200 测试样本上同量级——评审独立重算 **stride-4/50 = 4.8304e-6、stride-1/200 = 4.8638e-6**（floor 拟合 = 200 stride-25 训练样本、每 bin solve reg=1e-10、未 pad 501 点 rfft、float64）。本报告早先版本曾断言"4.83e-6 为评估子集伪影、同一拟合在全 200 测试样本上 TOTAL=9.2e2（bin1=3.2e5）"——**撤回**：9.2e2/3.2e5 无法从任何已提交协议复现（floor 协议下 bin1 每 bin rl2 实为 5.4e-5），系早期**废弃的中间 per-bin 实现**（`/tmp/m5_lspinv_run.log` 显示该迭代输出 1e7-1e11 垃圾、cond 值与最终报告不符）的伪影，被误归因为"评估子集伪影"。另：floor 拟合的 bin-1 映射距 A_true 仅 0.315（Frobenius），其 test rl2 仍达 5.4e-5——映射差异落在测试输入不探测的方向。
2. **bins 1-2 病态根因 = 泄漏目标 + 斜坡方向**：2/4Hz 无扰动力音调（音调在 8-14Hz），bin 1-2 目标能量仅占全带 2.0e-5，且由音调窗泄漏产生；输入以斜坡（ramp）为主导，幅值巨大。每 bin Gram cond ~1e27-1e29（neural_operator_2 与 _3 相同；n=200 实堆叠 Gram cond ~1e31）。**不可辨识性真实存在，但表现为训练子集不稳定性**：floor 的 stride-25 训练子集拟合音调 bin 每 bin rl2 = 6.7e-10，而连续 0..199 训练子集拟合 = 1.1e-2-1.6e-2；"|M|~1e3 随机矩阵"表述误导——max|M|=1137.55 实为质量系数（bin-1 映射 0.315 距 A_true 但 test rl2 5.4e-5）。复数 Hermitian 解（zgesv）在同样 Gram 上直接爆掉（rl2 ~1e10）——这是先前的 zgesv 修复的诊断依据。
3. **音调 bin 4-7（占 99.98% 目标能量）的每 bin 拟合系统性偏置 ~1e-3，且随 n 增长**（bin 4 Frobenius 相对偏置：n=200 → **1.015e-3**，n=1000 → **1.845e-3**，n=5000 → **3.472e-3**；早先"n=200/1000/5000 完全一致、恒为 1.011e-3"为假，1.011e-3 仅对应 n=200 量级）——每 bin 输入流形仅 ~7-11 维有效（音调 bin Gram 相对奇异谱 >1e-10 阈 ~11 维、~15 近零方向；配方受限），映射沿近零方向的行动**信息论上不可辨识**。每 bin test rl2 因此训练子集依赖（见 item 2），"每 bin rl2 ~1.1e-2"作为普适断言为假。
4. **正确模型 = joint frequency-shared map（LTI）**：精确算子 d = M q̈ − B u 逐点线性 → 每 bin 映射频率无关 → 单一共享 2×13 复映射在 bins 3-250 联合拟合（bins 0-2 排除：斜坡近共线结构会主导联合条件数；共享映射对所有 bin 求值）。joint 拟合恢复精确映射至 **5.5e-13**（n=5000，评审独立重算；早先报告 3.4e-12 同量级），每 bin test rl2 见上表——权威值为 `ls_init_report.json` 的 `per_bin_test_rl2`：**2.264e-11（1e-12）/ 7.370e-11（1e-10）/ 3.059e-10（1e-8）**（早先的"总 rl2 9.6e-11"为一次 ad-hoc 验证（joint 无 ridge 拟合）数值，不在任何 ls_init_report.json 中，撤回）；精确物理映射 A_true 本身为 4.7e-15。注入网络后（padded 基）官方 2.83e-6——与解析注入 2.84e-6 相同，即 ~2.8e-6 为 padded 基网络表示层，非映射质量。
5. `fit_ridge_ls_maps` 已按 joint 拟合重写（每 bin Gram 条件数报告保留）；brief 的"每 bin 解"规格被诊断证伪后替换，偏差已记录在代码 docstring。
6. `per_mode_ls_floor` 既有缺陷（评审发现 3，已修复）：`range(0,200,4)[:n_test]` 使 n_test 无论多大只取 50 样本——现改为 n_test>50 时 stride-1 全采样；两协议 floor 值同量级（4.83e-6 / 4.86e-6），结论不受影响。

## F4 机理诊断微实验（50ep）

配置：`configs/m5_linear_13ch_m252_diag.yaml`（13ch 线性 FNO，modes 252，w128，activation=none，b1；b8 lr5e-5 wd1e-4，cosine eta_min 3e-6 从头调度，50ep，validate_every 5）。SLURM：`slurm_scripts/m5_diag_50ep.sh`（job 30926，GPU node1，RTX 4090，torch 2.5.1+cu121）。

| epoch | LS 初始化 train rl2 | LS 初始化 val rl2 | 随机初始化 train rl2 | 随机初始化 val rl2 |
|---|---|---|---|---|
| 1 | 2.376e-01 | — | 1.000e+00 | — |
| 5 | 2.196e-01 | 9.557e-02 | 9.984e-01 | 9.976e-01 |
| 10 | 1.755e-01 | 2.468e-01 | 9.250e-01 | 8.939e-01 |
| 15 | 1.485e-01 | 8.371e-02 | 7.020e-01 | 6.941e-01 |
| 20 | 1.465e-01 | 8.472e-02 | 5.990e-01 | 5.673e-01 |
| 25 | 1.038e-01 | 1.222e-01 | 2.448e-01 | 3.024e-01 |
| 30 | 8.341e-02 | 7.339e-02 | 8.802e-02 | 8.628e-02 |
| 35 | 5.795e-02 | 1.656e-02 | 6.709e-02 | 4.934e-02 |
| 40 | 3.292e-02 | 2.213e-02 | 4.802e-02 | 3.911e-02 |
| 45 | 1.856e-02 | 9.414e-03 | 4.189e-02 | 3.888e-02 |
| 50 | 1.255e-02 | 9.962e-03 | 4.027e-02 | 3.350e-02 |
| test（50ep 后） | — | **9.63e-03** | — | **3.36e-02** |

- 初值：官方 test 2.831e-6 → 50ep 后 val 9.96e-3：**恶化 3.5e3×（>10×）→ F4 机理复现**
- 机理（代码路径验证）：`restore_training_state` 严格加载（权重含注入映射，max|weights1|=1137.55=质量系数）；Adam 的无差别步进（梯度归一化，步长 ~lr 与梯度幅值无关）在第一个 epoch（625 步）内把 1e-6 级最优的残差梯度尖点放大——epoch 1 平均 train rl2 即 0.238（CPU 1-epoch 复现 0.218 一致）
- 对照：LS 初始化 50ep 后仍比随机快 3.5×（盆地残留），但按判定标准 P1 关闭

## 命令与环境

```bash
# 1) lspinv 扫描（分析 .venv，CPU）——3 reg × 5000 样本 joint 拟合 + 注入 checkpoint + 自评
.venv/bin/python scripts/analysis/ls_init_fno.py \
    --config configs/m5_linear_13ch_m252_diag.yaml \
    --data-root ~/data/neural_operator_3 \
    --output-dir outputs/m5_ls_init --mode lspinv \
    --reg 1e-12 1e-10 1e-8 --n-train 5000

# 2) 官方评估（禁 TF32；CPU 无 TF32）
PYTHON_BIN=.venv/bin/python scripts/evaluate.sh \
    --checkpoint outputs/m5_ls_init/reg_1e-10/best_model.pt \
    --data-root ~/data/neural_operator_3

# 3) F4 微实验（SLURM GPU；base env ~/envs/base/bin/activate）
sbatch slurm_scripts/m5_diag_50ep.sh   # job 30926；轨迹 outputs/job_30926/m5_diag_{lsinit,rand}/history.json

# 4) CPU 1-epoch 验证（权重加载 + Adam 摧毁复现）
.venv/bin/python scripts/train.py --config configs/m5_linear_13ch_m252_diag.yaml \
    --output-dir /tmp/m5_cpu_check --resume outputs/m5_ls_init/reg_1e-10/best_model.pt \
    --device cpu --epochs 1 --overwrite
```

- 环境：分析 `.venv`（CPU torch 2.5.1，numpy/h5py）；训练 `~/envs/base/bin/activate`（torch 2.5.1+cu121，RTX 4090）
- 资产：`outputs/m5_ls_init/reg_{1e-12,1e-10,1e-08}/best_model.pt` + `ls_init_report.json`（run 产物，gitignored）；轨迹 `outputs/job_30926/...`（run 产物，gitignored）

## 提交

- 层 1（代码）：`scripts/analysis/ls_init_fno.py`（lspinv joint 拟合重写 + 诊断 docstring）、`configs/m5_linear_13ch_m252_diag.yaml`（新建）、`slurm_scripts/m5_diag_50ep.sh`（新建）
- 层 2（结果）：`experiments/analysis/2026-08-24/m5_1_diag.md`（本文件）
- run 产物不入库；无 AI 署名；不 push
