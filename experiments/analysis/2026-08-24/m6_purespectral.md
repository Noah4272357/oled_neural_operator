# M6 战役：纯谱模型 SGD 训练达官方 test rl2 ≤ 1e-4（2026-08-24/25）

> 硬门判定：**PASS**——3/3 seeds 官方 test.relative_l2 全部 ≤ 1e-4（2.8379e-06 / 3.4073e-06 / 4.8738e-06），最差余量 20.5×。本战役为**纯训练达成（零注入）**。
> 前置：M5 战役（2026-08-24）FNO 空间学习极限 1.85885e-4 全链关闭后，用户授权跳出 FNO 空间；spike 实证（2026-08-24）先行，M6-1 官方管线集成（2026-08-25），M6-2 官方训练 3 seeds（2026-08-25 02:16–02:33），M6-3 本记录落盘 + 提交。

## 1. 战役定义与硬门

- **硬门**：官方 `scripts/evaluate.py` test.relative_l2 ≤ 1e-4（13ch 输入 force(4)+q(3)+qdot(3)+qddot(3) → disturbance(2)，禁 TF32；CPU float64 天然满足）；速度不报告
- **训练语义**：必须 SGD 训练达成（零训练解析注入保底不再算数——M5 用户定义延续）；L-BFGS 已证失效（R3b 9.44e-1）、AdamW 固定步长地板 7e-4（R3d）→ SGD+momentum 为唯一已验证一阶路径
- 数据档：正式档 `~/data/neural_operator_3`（5000/500/200，13ch，seed 20260810）
- 模型：纯谱共享复数线性映射（rfft → 固定白化 W buffer → 共享复数 M(2×13) → irfft，26 训练参数），经 `src/models/factory.py` 注册（name="spectral"），官方 evaluate.py 严格 load_state_dict（含 persistent buffer W）

## 2. Spike 已证事实（2026-08-24，m6_spike_eigen.py / m6_spike_spectral.py）

> 数值以本文件与 `OPTIMIZATION_REPORT.md` §11 为准；部分 round-1 定性事实引自 spike 脚本 docstring。指标为官方路径（src.training.validate.validate，与 scripts/evaluate.py 同代码路径）。

### round-1：结构/预处理/优化器初扫（S1/S1n/S2/S2n + REF 闭式天花板）

| run | 配置 | 结果（官方 test rl2） | 结论 |
|---|---|---|---|
| REF-A | 闭式 LS 共享映射 fit bins 3-250（numpy lstsq，float64） | 1.353e-5（bins 0-2 贡献 1.35e-5） | **bins 0-2 必须参与拟合** |
| REF-B | 闭式 LS 共享映射 fit 全 bins 0-250（ridge 1e-10） | 4.593e-7 | 217× 低于门；共享映射类可达 |
| S1 | bare 共享映射，f64 AdamW lr=1e-3 | 几乎不动（train rl2 0.9995，10ep） | 裸 Gram 条件数受限（损失相关方向特征值 1.7e3-2.4e4，跨 ~5 量级） |
| S1n | 通道归一化（固定 per-channel scale），f64 AdamW 1e-2，50ep | 8.13e-3 平台 | 条件数受限（L-BFGS 同平台 7.7e-3，**非 F4 地板**） |
| S1n | 通道归一化，f64 SGD-m 1e-1 | 发散（3.4） | 一阶步长与条件数不匹配 |
| S1n f32 vs f64 | 同上，float32 对照 | 8.52e-3 vs 8.13e-3 | 本误差级 F4 不构成威胁；仍用 float64 |
| S2/S2n | per-bin 251×2×13 独立映射设计（~6.5k 复参） | M5-1 证据：per-bin 拟合对训练子集敏感（test rl2 1e-10..1e-2 跨子集） | round-1 不优先，最终未入选（共享映射即精确正确类） |

### round-2：全局白化 bug（关键教训）

| run | 结果 | 机理 |
|---|---|---|
| S1w 全局白化（W 缺 sqrt(N_pairs) 归一化） | 训练"冻结"：train rl2 0.995（30ep） | G 聚合 5000×251 对；不除 N_pairs 时白化后 z 协方差 = I/1.25e6 → 最优映射 M* 放大 ~1120× → AdamW 固定步长（F4 机理）需 ~50k 步遍历参数空间 → 冻结非收敛失败 |

修复：`train_whiten` 除以 N_pairs = 5000×251 = 1,255,000 → 每 (样本,bin) 协方差 = I（round-3 起）。

### round-3：per-pair 白化 × 优化器矩阵（R3a-e）

| run | 配置 | 结果 | 判定 |
|---|---|---|---|
| R3a | 共享 scale 归一化 + L-BFGS | 7.668e-3 | 条件数对照（仅 scale 归一化，非白化）；L-BFGS 仍可行但远未达门 |
| R3b | 每对白化 + L-BFGS | 9.44e-1 | L-BFGS 失效（关闭） |
| R3c | 每对白化 + SGD-m lr0.1 ×100ep | **1.81e-5** | **首个训练达门（5.5×）**，一阶 SGD 家族；动量噪声平台 1.5-2.4e-5 |
| R3d | 每对白化 + AdamW 固定 1e-2 | 6.99e-4 | AdamW F4 地板在 7e-4 |
| R3e | 每对白化 + AdamW cosine 1e-2 | 6.46e-2 | cosine 过早退火，不可用 |

（R3a 数值 7.668e-3 取自 spike 日志 m6_spike_round3.log：shared scale 归一化 + L-BFGS 100ep。R3a-f 全表见下节日志证据。）

### round-4：per-bin 白化诊断（R4a-f）

- 诊断：全局 per-pair 白化使**聚合**协方差为 I，但 bin 混合（ramp bins 0-2 / tone bins 4-7 / noise bins 8-250）下 batch 协方差 spread ~6e4（diag 5e-3..315）
- 方案：per-bin 白化 W_k（251 个 13×13，各 bin 自己的 per-pair Gram）→ 每 bin z 协方差 = I，批 Hessian ~I（秩亏 tone bin 塌缩，映射行保持初值，无害）
- R4a-f 网格（shared/perbin × lbfgs/sgd-m/adamw-cos）运行；最终配方未采用 per-bin 白化——全局 per-pair 白化 + 共享映射（S5a）即达门

### round-5：SGD-m 步进衰减（S5a/b/c）

| run | 调度 | 结果 | 判定 |
|---|---|---|---|
| **S5a** | step 0.1→0.01→0.001 ×120ep（60/40/20） | **2.886e-6** | **最终配方，34.7× 余量，近闭式地板（REF-B 4.59e-7）** |
| S5b | drop 0.1→0.001@60 ×100ep | 3.747e-6 | 单跳衰减变体（100ep，155.6s），近 S5a 但略差；未入选 |
| S5c | 固定 0.1 ×120ep | 2.039e-5 | 固定 lr 停在动量噪声平台（**衰减的必要性**） |

## 3. 官方训练结果（3 seeds，M6-2 2026-08-25 02:16–02:33）

白化复用 M6-1 产物 `outputs/spectral_whiten/whiten.pt`（与 spike `train_whiten` bitwise identical，M6-1 评审独立复验 np.array_equal=True）。训练命令 `scripts/train.py --config configs/m61_spectral.yaml`（seed 1 默认；seed 2/3 加 `--seed 20260811/20260812 --run-name m61_s*`）。120ep/seed ≈ 3.2min，CPU float64。

| seed | run 目录 | test.relative_l2 | test.mse | 余量（1e-4/rl2） | wall time | 判定 |
|---|---|---|---|---|---|---|
| 20260810（config 默认） | experiments/spectral_w128_b1_seed20260810_20260825-021611 | **2.8379e-06** | 2.6611e-11 | 35.2× | 206.99 s（~1.72 s/ep） | PASS |
| 20260811 | experiments/m61_s20260811 | **3.4073e-06** | 3.8362e-11 | 29.3× | 197.95 s（~1.65 s/ep） | PASS |
| 20260812 | experiments/m61_s20260812 | **4.8738e-06** | 7.8490e-11 | 20.5× | 192.39 s（~1.60 s/ep） | PASS |

- **3/3 seeds 全部 ≤ 1e-4 → 战役硬门 PASS**；均值 3.71e-6 ≤ 5e-6 → 未触发"配方鲁棒性风险"附加披露（阈值 >5e-6）；checkpoint_epoch 均 120（best = last ep，multistep ep60/100 两次 ×0.1 后 0.001 收敛平台）
- 官方评估输出原文（scripts/evaluate.py 原样，验收口径不加工）：
  - seed 20260810：`{"metrics": {"mse": 2.661144736374193e-11, "relative_l2": 2.8378861881815567e-06, "rmse": 5.158628438232583e-06}, "checkpoint_epoch": 120, "samples": 200, "split": "test"}`
  - seed 20260811：`{"metrics": {"mse": 3.8362454752005106e-11, "relative_l2": 3.4073278340511116e-06, "rmse": 6.193743193901819e-06}, "checkpoint_epoch": 120, "samples": 200, "split": "test"}`
  - seed 20260812：`{"metrics": {"mse": 7.848976491153437e-11, "relative_l2": 4.873794803004062e-06, "rmse": 8.859444955048504e-06}, "checkpoint_epoch": 120, "samples": 200, "split": "test"}`

### val relative_l2 轨迹（每 validate_every=10 ep，history.json）

| ep | seed 20260810 | seed 20260811 | seed 20260812 |
|---|---|---|---|
| 10 | 7.842684e-01 | 7.839850e-01 | 7.842279e-01 |
| 20 | 5.670453e-01 | 5.667633e-01 | 5.670059e-01 |
| 30 | 3.498266e-01 | 3.495438e-01 | 3.497850e-01 |
| 40 | 1.326118e-01 | 1.323141e-01 | 1.325728e-01 |
| 50 | 3.647073e-05 | 2.112689e-05 | 1.464925e-05 |
| 60 | 9.192501e-06 | 3.618833e-05 | 1.727265e-05 |
| 70 | 3.632234e-06 | 4.210708e-06 | 5.619502e-06 |
| 80 | 3.421886e-06 | 3.928789e-06 | 5.368094e-06 |
| 90 | 3.281537e-06 | 3.663426e-06 | 5.097302e-06 |
| 100 | 2.892430e-06 | 3.426137e-06 | 4.888756e-06 |
| 110 | 2.781388e-06 | 3.325398e-06 | 4.789567e-06 |
| 120 | 2.758734e-06 | 3.306079e-06 | 4.763359e-06 |

三轨迹形态一致：ep10-40 慢降（~0.13）→ ep50 突变 ~1e-5（multistep 首个里程碑前 lr=0.1 阶段高阶收敛）→ ep60/100 两次退火 → 1e-6 平台；best=ep120。

## 4. 与 M5 关系

M5（FNO 空间，54 万参数）SGD 学习极限 1.85885e-4 全链关闭；M6 跳出 FNO 空间，以纯谱共享映射（数据内蕴精确 LTI 类，26 参数）SGD 训练最差 seed 4.87e-6 达门——**硬门由纯训练达成，零注入**，并超越 F5 解析注入保底（3.078e-6，OPTIMIZATION_REPORT.md §8.3）同量级。

## 5. 复现命令

```bash
# 1) 白化（确定性，train 5000 样本全量，~6.7s；与 spike train_whiten bitwise identical）
.venv/bin/python scripts/analysis/spectral_whiten.py --data-root ~/data/neural_operator_3 --output outputs/spectral_whiten/whiten.pt

# 2) 训练（S5a 配方：SGD-m 0.9 + multistep [60,100] gamma 0.1 + float64；120ep ≈ 3.2 min/seed）
.venv/bin/python scripts/train.py --config configs/m61_spectral.yaml            # seed 默认 20260810
.venv/bin/python scripts/train.py --config configs/m61_spectral.yaml --seed 20260811 --run-name m61_s20260811

# 3) 官方评估（唯一验收，原样输出）
.venv/bin/python scripts/evaluate.py --checkpoint <run>/best_model.pt
```

## 5.5 证据日志（全管线原始输出，随本记录入库）

`m6_logs/`（本目录下）保存 M6 全流程原始运行输出（原始证据，逐位可查）：

| 文件 | 内容 |
|---|---|
| m6_spike_full.log | round-1 结构/预处理/优化器初扫（S1/S1n/S2/S2n）+ REF-A/B 闭式地板 |
| m6_spike_round3.log | round-3 R3a-e 优化器矩阵（R3a 7.668e-3 → R3e 6.463e-2 全表 + 每 10ep 轨迹） |
| m6_spike_round5.log | round-5 S5a/b/c 步进衰减（S5a 2.886e-6 / S5b 3.747e-6 / S5c 2.039e-5，每 10ep 轨迹） |
| m6_spike_w.log | 白化矩阵诊断（W 谱/数值，round-2 全局白化 bug 证据） |
| m6_debug_whiten.py | 白化冻结 bug 复现脚本（缺 sqrt(N) 归一化 → M* 放大 ~1120×） |
| m6_seed{1,2,3}_train.log | 官方训练逐 epoch 输出（epoch 轨迹 + summary，含 in-train test） |
| m6_seed{1,2,3}_eval.log | 官方 evaluate.py 原样输出（验收指标，原始证据） |

## 6. 记录落盘与提交

- 本记录 + OPTIMIZATION_REPORT.md §11（M6 战役汇总）+ memory（m6-purespectral-campaign.md）由 M6-3 落盘
- 提交序列（M6 全战役）：4c2bac5（模型+factory）→ 5af3b0e（sgd+multistep+trainer+测试）→ c1f326d（whiten 脚本+config+gitignore）→ 071b3aa（M6-2 gitignore 补漏）→ M6-3（spike 脚本 + 计划 + §11 + 本记录）→ 849a869（证据落盘：m6_logs 11 文件 + spike 数值补全 + §5.5 表）→ d2e33fa（README/README_zh/CLAUDE.md 达门状态同步）；分层提交、消息无 AI 署名、不 push；run 产物（experiments/spectral_*/、experiments/m61_s*/、outputs/spectral_whiten/）不入库
