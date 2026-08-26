# oled-neural-operator

> 项目状态：目标已达成 — 硬门（纯 SGD 训练 test.relative_l2 ≤ 1e-4）由 M6 纯谱模型达成：3/3 seeds ≤ 4.87e-6（最差余量 20.5×，零注入，2026-08-25）；F5 解析注入 3.078e-6 为对照记录
> 当前阶段：性能战役收官 — M6 纯谱共享映射（26 复参，rfft→白化→M→irfft）SGD 训练达门；FNO 空间学习极限 1.86e-4（M5）与 E31 2.03e-4 已被谱空间超越；最终谱模型禁止 resume 微调（F4 证伪）
> 文档约定：README.md（英文规范源）/ README_zh.md（中文镜像），本文件为 AI 助手指令

## 最终目标（THE GOAL）

- **测试集相对 L2 误差 ≤ 1e-4**（`test.relative_l2`，`scripts/evaluate.py` 输出）。
- **全权授权**：可任意更改项目内神经算子代码实现（模型/训练/数据管线/loss/配置），目标是唯一约束；不破坏项目结构合规（type-2、doctor）与提交纪律。
- **当前状态（已达成）**：硬门由 **M6 纯谱模型**达成——纯训练（SGD+momentum 0.9，float64，零注入）3/3 seeds 官方 test.relative_l2 = 2.838e-6 / 3.407e-6 / 4.874e-6（最差余量 20.5×，均值 3.71e-6）。M5 关闭 FNO 空间（学习极限 1.85885e-4）后跳出 FNO：以数据内蕴精确 LTI 类（rfft → 固定白化 W buffer → 共享复数 M 2×13 → irfft，26 复参）训练达成，120ep/seed ≈ 3.2min CPU。F5 解析注入（零训练，`outputs/ls_init_f1/best_model.pt`）test rl2 = 3.078e-6 为对照记录（同量级、零训练成本），仍禁止 resume 微调（F4 证伪）。
- 战役档案：`OPTIMIZATION_REPORT.md` §11（M6 汇总）/§8（性能战役 E 系列 + F 系列）+ `experiments/analysis/2026-08-24/m6_purespectral.md`（全管线记录，含 `m6_logs/` 11 个原始运行日志）。8ch 非线性战役（✅ SGD 训练收官）：`OPTIMIZATION_REPORT.md` §12 + `experiments/analysis/2026-08-26/8ch-nonlinear-campaign.md`（关键结论：r 的宽带尾为输入全谱的跨频段线性函数；8ch 线性表示无信息极限——投影（全秩）train R²=0.9994，f64 真数值秩 2339/3232；⚠️ 双重假象修正——原闭式上限 9.18e-5 为直接 einsum 收缩顺序伪影，闭式家族"全死"（≥1.52e-3）与"8ch 信息极限 0.638"为诊断脚本多次迭代 shuffle 的**行错位假象**；**行对齐重测：Tikhonov λ=1e-3 闭式解 9.42e-5 达标（零训练）但用户裁决不接受闭式收官，坚持必须 SGD 训练**；结构扫描（per-bin 1.526e-3 / Toeplitz 共享核全败 / 输出侧不可低秩 / **输入侧 PCA d=1280 最小可表达**）→ **✅ 纯 SGD 训练达标：固定 PCA 投影 d=1280 + SGD+momentum 0.9 + WD 1e-3 + 固定对角预条件 P=1/(S²+λ)（train 谱统计，M6 白化精神，正定预条件不改变不动点）+ float64，3/3 seeds test rl2 = 9.480689e-05 ≤ 1e-4（`outputs/sgd_pca_1280/`，GPU ~1min/seed）**；LBFGS 1.055e-4 未达标根因 = 目标错误（无正则过拟合）而非优化器；m67c-g 报告值曾因评估 reshape 实虚混叠虚高 2.4-2.5×；欠收敛根因 = full-batch 600 步不足 + 条件数灾难）。
- 战役纪律：每个假设必须有对照实验证据；指标以测试集为准，训练集/验证集为诊断信号；进展与失败都要落盘（`OPTIMIZATION_REPORT.md` 追加 + `history/` 或 experiments 记录）。

## 开发环境

- **项目级 `.venv/` 纪律**：`requirements.txt` 是依赖规范（随仓库开源），`.venv/` 是本地实例化（gitignored）。创建：`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`。严禁安装到系统/user site-packages。
- **`PYTHON_BIN` 环境变量**：`scripts/train.sh` / `scripts/evaluate.sh` 通过它切换解释器（如 `PYTHON_BIN="$HOME/miniconda3/envs/d2l/bin/python"`）；未声明时用 `python`。
- 本项目依赖 torch/h5py/numpy；测试套件（tests/）：test_config/test_paths 纯标准库（任意解释器可跑），test_cache/test_metrics 需 torch（项目 .venv，`PYTHON_BIN=.venv/bin/python`）。

## 项目概览（Project Overview）

- 用 **1D 傅里叶神经算子（FNO1d）** 学习 OLED microstage（微动台）的 **inverse disturbance** 算子：从执行器力 + 编码器位移轨迹预测扰动力轨迹，作为仿真替代模型。
- 默认问题定义（数据集 manifest `recommended_problem`）：输入 8 通道 = `force`(4) + `encoder_displacement`(4)，输出 2 通道 = `disturbance`，时间网格 501 点（dt=1ms，0.5s）。
- 训练循环：500 epoch 默认、AdamW + cosine 调度、relative_l2 损失、每 `validate_every` 评估、best/last checkpoint；评估不重建 trainer（checkpoint-only）。
- 自驱动假设-验证循环项目（无上游论文复现），数据为合成仿真基准。

## 核心设计原则

- **配置驱动**：实验选择全在 `configs/config.yaml`（JSON 兼容，无 PyYAML 也可解析），Python 模块只负责构造与运行行为；CLI 标志覆盖配置。
- **懒加载 HDF5**：每个仿真一个 `.h5`，`__getitem__` 内开文件——worker 安全、内存与数据集大小无关。
- **确定性**：seed 注入 DataLoader generator + `seed_worker`，训练/验证/测试顺序可复现。
- **checkpoint 自包含**：v2 schema 存解析后 config + legacy args + RNG 状态（python/numpy/torch/cuda），`--resume` 续训可换 LR/重建调度器。

## 目录结构

```
oled-neural-operator/
├── CLAUDE.md               # AI 助手指令（本文件）
├── README.md / README_zh.md# 项目文档（英文规范源 + 中文镜像）
├── requirements.txt        # 依赖规范（版本区间 + 用途注释）
├── configs/config.yaml     # 实验配置（默认 recommended_problem 8ch→2ch）
├── OPTIMIZATION_REPORT.md  # 训练成本优化（5.8×）+ 性能战役（E1-E36 + F 系列）全档案
├── src/
│   ├── data/               # dataset（HDF5 懒加载）/ dataloader / preprocessing
│   ├── models/             # fno.py（FNO1d 实现）+ factory.py
│   ├── training/           # 组件工厂 / epoch 循环 / 验证 / trainer
│   └── utils/              # config / checkpoint / logging / device / seed / paths
├── scripts/
│   ├── train.py            # 训练 CLI 与应用装配（--data-root 可覆盖数据路径）
│   ├── evaluate.py         # checkpoint-only 评估
│   ├── train.sh / evaluate.sh  # shell 启动器（PYTHON_BIN）
│   └── analysis/           # 数据集内蕴分析（analyze_dataset_fourier.py）+ 解析逆算子注入（ls_init_fno.py）
├── slurm_scripts/          # SLURM 作业脚本（train_cost_probe / train_opt_ablation / train_linear_f1-f4）
├── tests/                  # unittest：config（5）+ paths（7），纯标准库
├── experiments/            # 运行产物目录（config 快照/history/metrics/log + 被忽略的 .pt；analysis/ 含 2026-08-23 内蕴分析）
├── outputs/                # SLURM 作业产物（job_*）+ 最终解析注入模型（ls_init_f1/，gitignored）
├── checkpoints/  logs/     # 运行产物目录（结构占位，仅 .gitkeep）
└── .gitignore
```

`experiments/` 承载实际运行产物（含已入库的 4 组历史实验与 F 系列分析）；`outputs/` 承载 SLURM 作业产物与最终模型（gitignored，零成本可复现）；`checkpoints/`、`logs/` 为运行产物占位目录，新产物类别（如汇总 JSON、managed runs）写入对应目录。

## 模型接口规范

- `build_model(config["model"], input_channels, target_channels)` 动态建网（`src/models/factory.py`）；`input_channels` 从数据集按 `input_fields` 声明顺序沿最后一维拼接计算。
- FNO1d 关键超参：`modes`（保留傅里叶模态数）、`width`（隐层宽）、`embed_dim`/`lift_dim`（lift/project MLP 维度）、`num_blocks`（Fourier block 数）。默认 128/128/64/64/1（约 214 万参数）。
- 输入拼接 `grid`（时间轴）作为额外通道；非周期域 padding=2。

## 数据 Split 规范

- 每样本一个 `.h5`；`dataset_manifest.json` 提供 `fields`（字段宽度）/`splits`/`recommended_problem`/`complete`。
- 规模：train 5000 / val 500 / test 200；时间 501 点。
- 字段宽度：`force`=4、`encoder_displacement`=4、`displacement/velocity/acceleration`=3、`disturbance`=2；HDF5 路径映射见 `src/data/dataset.py` `FIELD_PATHS`。
- 训练 shuffle、验证/测试保持 manifest 顺序；`max_*_samples` 限制样本数（烟测用小值）。

## 数据体系（Data System Reference）

- **数据根解析优先级**（`src/utils/paths.py` `resolve_data_root`）：
  1. 显式路径（`--data-root` CLI 或 config `data.root` 非空值）——直接使用；
  2. `$DATA_ROOT/<neural_operator_2>`（项目级环境变量）；
  3. `~/data/neural_operator_2`——最终默认，回退时 stderr 显式警告（每进程一次）。
- `config.data.root: null` 表示运行时解析（train/evaluate 启动时经 `apply_data_root` 写回绝对路径，保存进 run 快照与 checkpoint）。
- 数据集只读：数据集不可修改，实验产物写项目内 `experiments/`。
- 数据集位置：`~/data/neural_operator_2`（即默认回退路径）。
- 战役档 `neural_operator_3`（M3/M5/M6 正式档）：manifest 3.0，同方程（`M q̈ = B u + f_dist`，B/H 不动），新增 **preroll 0.5s + integration substeps 5**；`recommended_problem` 仍为 inverse 8→2。M5/M6 以 13ch 状态输入（force+displacement+velocity+acceleration → disturbance）经 `configs/m5*.yaml` / `configs/m61_spectral.yaml` 显式指向 `~/data/neural_operator_3`。

## 评估协议

- 主指标 **relative_l2**（另支持 mse/rmse，全局聚合语义，见 `src/training/metrics.py`）。
- 训练期 val 每 `validate_every` epoch 评估一次；选优指标 = 损失函数名（默认 relative_l2），最优模型存 `best_model.pt`。
- 评估不重建 trainer：`scripts/evaluate.sh --checkpoint <run>/best_model.pt`；config 从 checkpoint 恢复，`--data-root` 可覆盖数据路径；split 默认 test。

## 实验执行

- `scripts/train.sh --config configs/config.yaml`；CLI 标志覆盖 config（`--epochs` 在 `--resume` 时为增量 epoch）。
- run 目录自动生成：`experiments/fno1d_w{width}_b{blocks}_seed{seed}_{timestamp}`，防碰撞；每 run 含解析后 `config.yaml`、`train.log`、`metrics.csv`、`history.json`（含 summary）。
- `--no-save` 不落盘；`--output-dir` 显式指定（非空目录需 `--overwrite`）。
- 13ch 状态输入战役档（M3/M5/M6）复现方式记录于 README（Data/Default problem 段）。

## 技能体系（Skills System Reference）

- 本项目暂无项目专属技能；本文件（CLAUDE.md）为 AI 助手指令的唯一项目内规范源，README 双语文档为人类可读规范源。

## 工程闭环（Engineering Closed Loop）

- 每轮循环：train → evaluate → 结果落盘（metrics/history/checkpoint）→ 实验清单与文档更新。
- 完成门以**真实验证**替代仪式化自检：unittest 套件（`python -m unittest discover -s tests -v`）+ git diff review + 退出码。
- 涉及科研声明（数值结论）须附对照实验证据与可复现命令，结论写入 README 或分析文档。
- git 纪律：提交人工执行、分层提交、消息不含 AI 共同作者署名。

## 数据源

- OLED microstage 合成仿真数据集 `neural_operator_2`（生成侧 manifest `complete=true`，本地已完整上传）；战役档 `neural_operator_3`（manifest 3.0，preroll 0.5s + substeps 5）为 M3/M5/M6 官方数据档。
- 工程假设（扰动 RMS 0.5-2% 连续力、扰动音调频率 {8,10,12,14}Hz、电机利用率 5-30% 等）以 manifest `engineering_assumptions` 为唯一规范源；`model` 段记录 CAD 推导的执行器映射 B 与编码器测量矩阵 H。
- 数据只读：任何处理不修改数据集文件（见"数据体系"）。

## 已知差异

1. **旧问题定义**：已入库的 4 组 500-epoch 实验使用 13 通道状态输入（force + displacement + velocity + acceleration → disturbance），与当前默认 8 通道 encoder 输入不同；完整配置与复现命令见 README。
2. **旧 checkpoint 路径**：历史 checkpoint 存旧机器绝对路径 `/data/charliewang_data/...`，换机评估需 `--data-root` 指向新位置（或 `$DATA_ROOT`）。
3. **数据集变更**：`neural_operator_2` 相对早期数据集的字段差异（新增 encoder_displacement/measurement_matrix_H）以 manifest `fields` 为准，`dataset.py` 已兼容。
4. **默认训练语义变更（2026-08-23 优化战役）**：默认 batch 8/lr 1e-3 → **64/2.83e-3**（sqrt LR 缩放，100-epoch 动力学验证等价，详见 OPTIMIZATION_REPORT.md）；新默认下训练的模型不与历史 500-epoch 实验（batch 8）直接可比。旧语义复现：`--batch-size 8 --learning-rate 1e-3 --amp none --no-memory-cache`。
5. **两套达标模型并存**：F5 解析注入（`outputs/ls_init_f1/best_model.pt`，零训练 epoch 0，test rl2 = 3.078e-6，复现见下）与 M6 纯谱模型（SGD 训练，test rl2 2.84e-6–4.87e-6）均为达标产物；按用户硬门口径（必须 SGD 训练达成、零注入）以 **M6 为准**。两者均**禁止 resume 微调**（F4 证伪：AdamW lr1e-5 微调 300ep 恶化 1400×，见 ANALYSIS_REPORT.md 发现 9）。F5 复现：`python scripts/analysis/ls_init_fno.py --data-root <ROOT> --output-dir outputs/ls_init_f1 --mode analytic --modes 252`（确定性，零训练成本）。M6 复现：`experiments/analysis/2026-08-24/m6_purespectral.md` §5。
6. **FNO 空间上限已被推翻**："纯学习路线最优 E31 = 2.03e-4（13ch 非线性 FNO，b8 lr5e-5×1200ep 链式微调）"与 M5 学习极限 1.85885e-4（OPTIMIZATION_REPORT.md §10）仅适用于 **FNO 空间**（表示能力限制：GELU 互调 + modes 截断）；M6 跳出 FNO 空间，以谱空间 26 复参 SGD 训练达 4.87e-6，"纯 SGD 不可达 1e-4"结论不成立。

## 与其他项目的关系

- 独立开源仓库：同工作区项目 stressno-bench / optics-surrogate 仅作结构与纪律参考，无代码依赖。
- 数据：`~/data/neural_operator_2`（只读，见"数据体系"）。
