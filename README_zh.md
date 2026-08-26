# OLED 神经算子（中文镜像）

> **规范源声明**：本文件为 [README.md](./README.md) 的中文镜像，英文版为准，如有出入以英文版为准。

FNO1d 代理模型（及 M6 纯谱模型），用于 **OLED 微动台扰动力估计**：从执行器力 + 编码器位移轨迹预测扰动力轨迹（inverse disturbance 问题），训练于合成仿真 HDF5 基准 `neural_operator_2` / `neural_operator_3`（train 5000 / val 500 / test 200，每样本 501 个时间点，dt = 1ms）。

> **状态**：探索期——训练/评估闭环完全可用；`experiments/` 下已入库 4 组 500-epoch 历史实验。数据集已本地完整上传，2026-08-23 完成全链路烟测。
>
> **性能目标已达成**：硬门（纯 SGD 训练 `test.relative_l2 ≤ 1e-4`）由 **M6 纯谱模型**达成（2026-08-25）——3/3 seeds `2.838e-6` / `3.407e-6` / `4.874e-6`（最差余量 20.5×，零注入）。F5 解析注入（零训练，modes=252 全频谱）`3.078e-6` 为对照记录。详见 [`OPTIMIZATION_REPORT.md`](./OPTIMIZATION_REPORT.md) §11 / §8 与 [`experiments/analysis/2026-08-24/m6_purespectral.md`](./experiments/analysis/2026-08-24/m6_purespectral.md)。

## 快速链接

- [English README](./README.md) — 规范源（英文）
- [`CLAUDE.md`](./CLAUDE.md) — AI 助手指令
- [`configs/config.yaml`](./configs/config.yaml) — 默认实验配置
- [数据](#数据) — `~/data/neural_operator_2`（默认数据根，只读）
- [`src/utils/paths.py`](./src/utils/paths.py) — 数据根解析逻辑

## 项目结构

```
oled-neural-operator/
├── CLAUDE.md  README.md  README_zh.md  requirements.txt
├── configs/config.yaml       # 默认问题：8 通道（force + encoder_displacement）→ 2 通道
├── src/                      # data / models / training / utils（paths、config 等）
├── scripts/                  # train.py / evaluate.py + train.sh / evaluate.sh
│   └── analysis/             # 数据集内蕴分析 + 解析逆算子注入
├── tests/                    # unittest，纯标准库（config 5 + paths 7）
├── experiments/              # 运行产物；历史实验 + 2026-08-23 分析已入库
├── outputs/                  # SLURM 作业产物（job_*）+ 最终解析注入模型（ls_init_f1/，gitignored）
├── checkpoints/  logs/       # 运行产物占位目录（仅 .gitkeep）
└── .gitignore
```

运行产物位于 `experiments/`（config 快照、`train.log`、`metrics.csv`、`history.json`、checkpoint）；`outputs/`、`checkpoints/`、`logs/` 为运行产物占位目录，供未来产物类别使用。

## 数据

数据集：`neural_operator_2`，每个仿真一个 `.h5`。数据根解析（见 `src/utils/paths.py`），优先级从高到低：

1. **显式路径**——CLI `--data-root`，或 config `data.root` 非空值（如历史 checkpoint 中存储的绝对路径）；
2. `$DATA_ROOT/neural_operator_2`——项目级环境变量；
3. `~/data/neural_operator_2`——最终默认回退，回退时 stderr 显式警告（每进程一次）。

`configs/config.yaml` 声明 `data.root: null`，路径在运行时解析；`apply_data_root` 把解析后的绝对路径写回 run 的 config 快照与 checkpoint，保证每个 run 自包含。

战役档 `neural_operator_3`（manifest 3.0，同方程 `M q̈ = B u + f_dist`，B/H 不动，新增 preroll 0.5 s + integration substeps 5）为 M3/M5/M6 官方数据档；M5/M6 以 13ch 状态输入经 `configs/m5*.yaml` / `configs/m61_spectral.yaml` 显式指向 `~/data/neural_operator_3`。

## 默认问题配置

默认问题遵循数据集 manifest 的 `recommended_problem`：**8 输入通道**（`force` ×4 + `encoder_displacement` ×4）→ **2 目标通道**（`disturbance`），501 个时间点。

```bash
scripts/train.sh --config configs/config.yaml
```

已入库的 4 组历史实验（各 500 epoch）及 M3/M5/M6 战役使用的是 13 通道状态输入定义（`force` + `displacement` + `velocity` + `acceleration` → `disturbance`，战役在 `neural_operator_3`）。复现命令：

```bash
scripts/train.sh \
  --input-fields force displacement velocity acceleration \
  --target-fields disturbance
```

## 可复现性

- **种子**：config 中的 `seed` 驱动 PyTorch generator 与 worker 播种（`src/utils/seed.py`）；run 目录名内嵌 seed。
- **确定性**：DataLoader 顺序经种子 generator 可复现；训练/验证/测试顺序固定。
- **Checkpoint**：v2 schema 存解析后 config、legacy args 与 RNG 状态（python/numpy/torch/cuda）；`--resume` 完整恢复，可换 LR/调度器。
- **测试**：`python -m unittest discover -s tests -v`（纯标准库，无需 torch/h5py）。
- **M6 纯谱配方**（SGD 训练达门，`OPTIMIZATION_REPORT.md` §11）：
  ```bash
  .venv/bin/python scripts/analysis/spectral_whiten.py --data-root ~/data/neural_operator_3 --output outputs/spectral_whiten/whiten.pt
  .venv/bin/python scripts/train.py --config configs/m61_spectral.yaml   # 120 ep，~3.2 min CPU
  .venv/bin/python scripts/evaluate.py --checkpoint <run>/best_model.pt
  ```

## 状态

| 阶段 | 说明 | 当前 |
| --- | --- | --- |
| 探索期 | 训练/评估闭环可用，4 组历史实验已入库 | ✅ |
| 性能目标 | `neural_operator_3` 上 `test.relative_l2 ≤ 1e-4` | ✅ **3.078e-6**（F5，零训练解析注入，§8.3） |
| 纯训练硬门（M6） | `neural_operator_3` 上 `test.relative_l2 ≤ 1e-4`，SGD 训练 | ✅ **4.87e-6** 最差 seed（余量 20.5×）——纯谱共享映射，26 训练参数，零注入 |
| 8ch 推荐问题达标（SGD） | 官方 8ch 问题（force + encoder_displacement → disturbance）上 `test.relative_l2 ≤ 1e-4`，SGD 训练 | ✅ **9.480689e-05**（3/3 seeds，余量 5.4%，2026-08-26）——PCA-1280 投影 + 固定对角预条件 SGD+momentum，零注入（§12.5） |
| 最佳学习型模型 | 纯谱共享映射，SGD 120ep（M6 S5a 配方） | `2.84e-6`–`4.87e-6`（3 seeds）——取代 E31 FNO 上限 `2.03e-4` |
| 成熟期 | 全战役可复现文档 + NOTES.md | — |
| 论文期 | 论文级实验与图表 | — |

---

## 使用文档

用默认配置训练：

```bash
scripts/train.sh --config configs/config.yaml
```

目标解释器未激活时设置 `PYTHON_BIN`：

```bash
PYTHON_BIN="$HOME/miniconda3/envs/d2l/bin/python" \
  scripts/train.sh --config configs/config.yaml --device cuda
```

命令行标志覆盖已加载的 config；`--resume` 时 `--epochs` 表示追加的 epoch 数。

通过配置或 CLI 选择 Adam 而非默认 AdamW：

```bash
scripts/train.sh --optimizer adam --learning-rate 1e-3
```

### 从 checkpoint 续训

向 `--resume` 传 `last_model.pt`，`--epochs` 设为追加的 epoch 数。要从指定学习率重新开始，传 `--learning-rate` 且不传 `--keep-resume-learning-rate`：

```bash
PYTHON_BIN="$HOME/miniconda3/envs/d2l/bin/python" \
  scripts/train.sh \
  --resume experiments/PREVIOUS_RUN/last_model.pt \
  --epochs 500 \
  --learning-rate 1e-5 \
  --output-dir experiments/continued_lr1e-5 \
  --device cuda
```

模型与优化器状态恢复、优化器学习率重置为 `1e-5`、为追加的 500 epoch 新建调度器。使用新输出目录以免破坏原 run。

若改用 checkpoint 中保存的学习率：加 `--keep-resume-learning-rate` 并省略 `--learning-rate`（除非你也要改配置调度器的参考学习率）。

每个已保存的 run 在 `experiments/` 下创建独立目录（除非显式指定 `--output-dir`）。run 内含解析后 `config.yaml`、`train.log`、`metrics.csv`、`history.json` 与启用的 `best_model.pt`、`last_model.pt`。显式非空输出目录受保护，除非传 `--overwrite`。

不重训直接评估：

```bash
scripts/evaluate.sh --checkpoint experiments/RUN/best_model.pt
```

v2 checkpoint 含解析后 config 与可复现状态，同时保留历史的 `args`、`model_state_dict`、`optimizer_state_dict`、`scheduler_state_dict` 键。

## 包布局

- `src/data`：数据集索引、预处理选择、DataLoader 装配
- `src/models`：FNO 实现与模型工厂
- `src/training`：组件工厂、epoch 循环、验证、trainer
- `src/utils`：config、checkpoint、日志、设备选择、播种
- `scripts/train.py`：训练 CLI 与应用装配
- `scripts/evaluate.py`：checkpoint-only 评估
- `scripts/train.sh` 与 `scripts/evaluate.sh`：shell 启动器
