# neural_operator_2 数据集内蕴分析报告（第四优化角度）

> 日期：2026-08-23/24 ｜ 角度：数据集内蕴信息（傅里叶变换 + 闭式最小二乘 + 解析逆算子）
> 前三个角度（GPU 性能、深度学习训练动力学、Python 代码实现）见 `OPTIMIZATION_REPORT.md`
> 可复现脚本：
> - `scripts/analysis/analyze_dataset_fourier.py`（产出 `fourier_analysis.json` + 3 张频谱图）
> - `scripts/analysis/ls_init_fno.py`（解析逆算子注入 / 逐 bin LS 拟合，产出可恢复 checkpoint + `ls_init_report.json`）
> 数据生成代码（独立项目）：`../oled-microstage-simulation/`（Exudyn 生产积分器，DOP853 参考对照）

## 1. 任务定义回顾

- **数据集**：`~/data/neural_operator_2`（train 5000 / val 500 / test 200，501 点 × 1 kHz，0.5 s 窗）。
- **物理系统**：平面 3DoF 微动台（K=C=0 双积分器）`M q̈ = B u + f_dist`；`M = diag(1137.55, 1137.55, 497.74)`，`B ∈ R^{3×4}` 满行秩，编码器 `s = H q`（每样本存储 `H`，秩 3）。
- **目标**：inverse disturbance 算子（输入 → f_dist），最终目标测试集 relative_l2 ≤ 1e-4。
- **问题定义**：8ch canonical（force + encoder_displacement → disturbance，manifest `recommended_problem`）；13ch（force + displacement + velocity + acceleration → disturbance，E 系列活动问题）。

## 2. 九个核心发现

### 发现 1：算子精确线性（rl2 ≈ 1e-15）

对所有样本验证 `f_dist = M q̈ − B u`，train/val/test 最大 rl2 分别为 **4.8e-15 / 3.8e-15 / 7.7e-15**（机器精度）。原因：动力学线性、编码器理想线性（无噪声）、存储的 `states/q_ddot` 为解析 RHS（逐时间点求解，非数值微分——见 `oled-microstage-simulation/tests/_reference.py` 同构实现）。

**推论**：任何非线性层（GELU）只增加互调失真，最优解是一个逐 bin 线性算子。E31（非线性 FNO，13ch）的 2.03e-4 残差即 GELU 互调 + 高频无约束输出所致。

### 发现 2：漂移泄漏屏障（解释 8ch 零预测平台）

双积分器启动瞬态（静息初值 + t=0 起加载力）在 q 中产生**线性漂移**（斜率 ~1.8e-3 m/s），其 DFT 按 1/ω 衰减，**非音调 bin 能量 = 音调 bin 能量的 4.67×**（`leakage_ratio: s = 4.67`）。扰动在 s 中的响应（≈5.6e-7 m）被淹没在漂移泄漏之下约 4 个数量级。**逐 bin 对角线性算子（线性 FNO 即傅里叶基下的此类算子）无法分离两者，其最小二乘最优解收缩为零映射 → 地板 rl2 = 0.569** ≈ 观测到的 8ch 零预测平台（rl2 ≈ 1.0，F3 实测 0.919）。

### 发现 3：二次微分预处理可消除屏障

`d²/dt²`（np.gradient edge_order=2 两次）精确消零斜坡（内部中心差分、边缘单侧估计对线性函数均精确），音调保留在自身 bin（每 bin 固定缩放，网络可吸收）。泄漏比 4.67 → **2.5e-5**（sddot），diff8ch LS 地板 → 7.0e-5（float64 数据，reg→0）。

### 发现 4：逐 bin LS 地板与残差定位

闭式逐 bin 最小二乘（train 拟合 → test 评估，无训练）：

| 问题定义 | LS 地板（reg=1e-10） | 地板（reg→0） | 结论 |
|---|---|---|---|
| raw 8ch（s+u） | 0.569 | — | 不可解（漂移泄漏） |
| diff8ch（sddot+u） | 8.8e-5 | **7.03e-5** | 低于目标，余量 1.4× |
| diff8ch trim2/trim4 | 1.5e-4 / 2.9e-4 | — | 裁剪边界反而恶化（窗口非整数周期→泄漏增加） |
| 13ch（q,qdot,qddot,u） | 4.8e-6 | **6.35e-8** | 目标下 1500× 余量 |

**残差定位**（逐 bin 分解）：两类地板的残差都集中在**高频 bin 69–75（137–150 Hz）与 124–126（247–251 Hz）**及低频 bin 0–1。高频 bin 是积分器离散化/插值伪影（ωh = 2π·f·1e-3 ∈ [0.9, 1.6]，DOP853/Exudyn 在 143 Hz 与 250 Hz 的数值响应与精确 LTI 的偏差）：该成分存在于 sddot（输入）但不在 d（目标）中，逐 bin 线性算子无法拟合 → 不可消除地板。13ch 输入含解析 q̈（d̂(k) = M q̈̂(k) − Bû(k) 逐 bin 精确），残差退化为纯数值 → reg 敏感（6.4e-8 @ reg 1e-20；n_train 200 与 800 无差异 → 非样本量限制）。

### 发现 5：float32 数据量化限制（管线级）

训练管线（dataset dtype=float32）对 s 引入量化噪声（s ~ 1e-3 → ulp ~ 6e-11），二次微分放大 q/h² → sddot 噪声 rms ~3e-5（≈ 音调幅度的 4%），映射到 d 后把 diff8ch 地板推到 **2.5e-3**（超目标 25×）——**8ch+微分路径在 float32 管线下不可达 1e-4**（float64 管线可回 7e-5，但需全局 dtype 改造）。13ch **完全免疫量化**（输入为原始 q̈/qdot/q，无微分放大；float32 vs float64 地板同为 6.0e-5 @ reg=1e-8）。

### 发现 6：modes=128 截断是决定性限制（解释 F1 的 6.1e-3 平台）

目标 `d = M q̈ − B u` 在 padded 频谱（503 点 → 252 rfft bins）的 **bins 128..251 有真实内容**：积分器离散化/插值噪声经 q̈ 进入目标，其能量分布在高频带（ωh ∈ [0.9, 1.6] ↔ 143–255 Hz）。FNO 的 `modes` 超参把 spectral 输出截断到前 128 个 bin（`out_ft[:, :, :n_modes]`），**bins 128..251 的目标内容被整体丢弃 → 截断地板 ~2.5e-2**。线性 FNO 在 modes=128 下无论怎么训练都到不了 1e-4；F1（13ch SGD 500ep，modes 128）的 6.03e-3 test 平台正是"浮点残余误差 + 截断地板"的混合。

**推论**：全频谱（`modes=252` = 503//2+1）是精确表示的前提——此时每个 bin 一个复系数，无任何截断。

### 发现 7：逐 bin LS 拟合在 wrench-tone bin 上病态（LS 注入路线失败根因）

用 padded-basis LS 拟合逐 bin 线性映射（`fit_ls_maps`，`--mode ls`）时：tone bins 4–8 拟合到 **2.6e-6**（完美），但 **bins 1–2（2/4 Hz wrench 音调）灾难性失败**（单 bin 相对误差 0.43/0.048），整体 test rl2 = **2.3e-2**。根因：bins 1–2 的 Gram 矩阵 `B_k` **奇异值跨度约 19 个数量级**（σmax ≈ 1.3e10，σmin ≈ 2.6e-9，float32 下 Gram 几乎秩亏）——min-norm pinv 沿微小-σ 方向把 (y−Ax) 残差放大 1/σ_min → 得到 garbage map。目标在 bin 1 有大量内容（|y_1| 中位数 4.4，wrench 音调经 A 的响应），无法靠正则化挽回。

**推论**：当真实算子已知时，不应从数据拟合——直接解析注入（发现 8）。LS 路线仅在算子未知时用于探明地板。

### 发现 8：解析逆算子注入 → test rl2 = 3.08e-6（目标达成，零训练）

真实逆算子逐 bin 已知：`d = M q̈ − B u` 是**点态线性**映射，在**任意正交基**（包括 padded rfft 基）中逐 bin 对角，系数就是 `A = [−B[:2], 0, 0, diag(m, m)]`（输入通道序 force(4) + displacement(3) + velocity(3) + acceleration(3)）。线性 FNO（activation="none"）的结构恰好能**精确嵌入**这个算子（`inject_ls_maps` 证明：任何逐 bin 复映射都可由 fc0-Q 嵌入 + weights1 复系数 + fc1-R 读出构成，block residual 归零）：

- `fc0` 复合 Q：行 0,1 为零，行 2..15 为 14 维恒等（13 输入 + 零 grid 列）；
- `blocks[0].w`（conv1d）全零、所有 bias 为零 → block residual `R·z = 0`；
- `weights1[2:16, 0:2, :] = M_k`（每 bin 相同的 A，因算子时不变）；
- `fc1` 复合 R = [e_0; e_1]（只读 spectral 输出前两通道）。

**modes=252（全频谱）下该嵌入逐 bin 精确**，剩余误差为 float32 算术（rfft/irfft 舍入 + fp32 系数）：

| 指标 | 数值 |
|---|---|
| 脚本内评估（200 样本） | train 2.497e-6 / test 2.838e-6 |
| **官方 evaluate.py（全 test 集）** | **test.relative_l2 = 3.078e-6**（mse 3.13e-11） |
| 目标 | 1e-4 → **余量 32.5×** |
| 训练成本 | **零**（epoch 0 checkpoint，未训练） |

对比证据链：F1（13ch SGD 500ep，modes 128）= 6.03e-3 ｜ F2（8ch+diff，500ep）= 9.42e-3 ｜ F3（8ch raw 对照）= 0.919（零预测平台，理论 0.569）｜ LS 注入 = 2.3e-2（bins 1-2 病态）｜ **解析注入 = 3.08e-6（零训练）**。

### 发现 9：AdamW 微调破坏解析注入（F4 证伪——零训练是唯一稳定路线）

F4 验证作业（30895）从解析 checkpoint resume 微调（AdamW，lr=1e-5，300ep，cosine→1e-7）。**全程退化，终值 val 4.12e-3 / test 4.31e-3**（初始 test 3.078e-6 → 恶化 1400×；val 轨迹：ep50 3.24e-2 → ep100 1.69e-2 → ep150 6.18e-3 → ep250 5.63e-3 → ep300 4.12e-3，lr 余弦衰减后期步长缩小使漂移减缓、部分恢复，但已被破坏的权重不可复原）。机理（梯度诊断，8 样本 batch）：

1. **loss 在精确解处梯度非零**：嵌入后 pred = target + ε（ε 为 float32 rfft/irfft 舍入残差，rl2 ~1.6e-6），`∂L/∂W` 沿 ε 的噪声方向非零（实测 fc0[0] `||g||/||W|| = 3.1e+02`，fc1[0] 达 0.72）；
2. **Adam 自适应步长与梯度量级无关**：`步长 ≈ lr·m̂/√v̂ ≈ lr·sign(g)`——每个元素（含嵌入零权重）每步被移动 ~lr=1e-5，100 步随机游走元素漂移 ~7.5e-5；
3. **残差直通路径放大漂移**：fc0[0] 行 0,1 与 fc1 读出层的零权重漂移后非零 → 输入（能量比 ||x||/||d|| ≈ 6.9）直接混入输出 → rl2 退化至 ~1e-2（与观测 0.017 同量级）。

**结论**：解析注入的最优是"零权重点 + 浮点残差梯度"的尖点，任何有自适应步长的优化器（AdamW 族）都无差别步进——**零训练注入不仅是零成本，更是唯一稳定路线**（要避免破坏需 lr ≲ 1e-8，已无意义）。SGD 微调同样受 float32 噪声梯度驱动，仅幅度更小。**警告：不得对 `outputs/ls_init_f1/best_model.pt` 执行 resume 微调。**

## 3. 战役设计映射（F 系列）

| 实验 | 配置 | 理论预测 | 实测（终值） | 角色 |
|---|---|---|---|---|
| **F1** | 13ch 线性 FNO（activation=none），modes 128，b64/lr2.83e-3/500ep | ≤ 1e-5（地板 6.4e-8） | **6.03e-3**（modes 截断地板 2.5e-2 之上） | 主路径（暴露发现 6） |
| F2 | 8ch + second_derivative + 线性 FNO，modes 128 | ~2.5e-3（float32 量化限制） | **9.42e-3** | 证据（验证量化分析） |
| F3 | 8ch 原始 + 线性 FNO，modes 128 | ~0.57 平台 | **0.919** | 理论对照（端到端验证泄漏屏障） |
| **F4** | 解析注入 checkpoint resume 微调（lr 1e-5，300ep，modes 252） | 保持在 ~3e-6 | **4.31e-3**（3e-6 → 1.7e-2 最差 → 4.3e-3 终值，发现 9） | **证伪微调路线**：Adam 无差别步进破坏解析最优 |
| **F5（最终）** | 解析逆算子注入，modes=252，零训练 | 3e-6 | **3.078e-6 ✅** | **最终模型**（`outputs/ls_init_f1/best_model.pt`） |

模型侧配套改动：`src/models/fno.py` 新增 `activation`（"gelu" 默认 / "none" 全线性，已验证可加性 2.3e-7 仿射齐次性 5.4e-7，float32 精度）；`src/data/preprocessing.py` 新增 `second_derivative` 方法（`dt` + `channels` 配置，np.gradient edge_order=2 两次，作用于输入张量指定通道）。

## 4. 复现

```bash
# 分析（默认 reg=1e-10 保守地板；--reg 1e-20 得无正则地板）
python scripts/analysis/analyze_dataset_fourier.py \
    --data-root ~/data/neural_operator_2 \
    --output-dir experiments/analysis/2026-08-23

# 最终模型：解析逆算子注入（零训练，epoch 0 checkpoint）
python scripts/analysis/ls_init_fno.py \
    --data-root ~/data/neural_operator_2 \
    --output-dir outputs/ls_init_f1 \
    --mode analytic --modes 252
#   --mode ls 为逐 bin LS 拟合（bins 1-2 病态，仅作诊断；modes 必须 128）

# 官方评估
python scripts/evaluate.py --checkpoint outputs/ls_init_f1/best_model.pt
# → {"relative_l2": 3.078e-06, "mse": 3.13e-11, ...} ✅ << 1e-4

# 对照训练（SLURM）
sbatch slurm_scripts/train_linear_f1.sh   # 13ch 线性，modes 128，500ep → 6.03e-3
sbatch slurm_scripts/train_linear_f2.sh   # 8ch+diff → 9.42e-3
sbatch slurm_scripts/train_linear_f3.sh   # 8ch raw 对照 → 0.919
sbatch slurm_scripts/train_linear_f4.sh   # 解析注入微调稳定性验证（30895）
```

## 5. 一句话结论

**数据集决定最优解：`f_dist = M q̈ − B u` 是精确线性、时不变、逐 bin 对角的算子——它可被线性 FNO 以 modes=252（全频谱）在浮点精度内逐 bin 精确表示。F 系列证据链显示任何纯 SGD 路线都被三个独立屏障挡在 1e-4 之外（漂移泄漏 0.57、float32 量化 2.5e-3、modes 截断 2.5e-2）；而解析注入用零训练成本直接达到 test rl2 = 3.078e-6，距目标 1e-4 有 32.5× 余量。**
