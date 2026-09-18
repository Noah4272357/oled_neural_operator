# OLED 微动台扰动力算子代理模型 / OLED Microstage Disturbance Operator Surrogate

本仓库提供一个 OLED 喷墨打印微动台的**外部扰动力反演**代理模型：由执行器力与
光栅位移读数时序，预测作用在运动刚体上的外部扰动力时序。

模型在配套仓库 [`oled-microstage-simulation`](../oled-microstage-simulation)
生成的冻结合成仿真数据集上训练。测试执行步骤见 [TESTING.md](TESTING.md)。

<details>
<summary>English</summary>

This repository provides a surrogate model for **external disturbance-force inversion** on
the OLED inkjet-printer microstage: from the actuator-force and encoder/grating displacement
reading time series, it predicts the external disturbance-force time series acting on the
moving body.

The model is trained on the frozen synthetic simulation dataset produced by the companion
repository [`oled-microstage-simulation`](../oled-microstage-simulation). The test procedure
is [TESTING.md](TESTING.md).

</details>

---

## 1. 项目简介 / Overview

- **输入**：执行器等效推力（4 路）与光栅位移读数（4 路）的时序，共 8 通道。
- **输出**：外部扰动力（X / Y）时序，共 2 通道。
- **数据**：冻结的合成仿真数据集，0.5 s @ 1 kHz，5000 / 500 / 200 划分。
- **模型**：基于谱域表示的算子代理模型，257,024 个可训练参数。
- **训练**：SGD，100 个 epoch，CPU 上约 60 秒。
- **结果**：测试集全局相对 L2 误差 `5.940277290056465e-08`（阈值 `1e-4`）。

数据集来自数值仿真，**不是真实设备的测量数据**。

<details>
<summary>English</summary>

- **Input**: 8 channels — actuator equivalent force (4) and encoder/grating displacement
  reading (4) time series.
- **Output**: 2 channels — the external disturbance force (X / Y) time series.
- **Data**: a frozen synthetic simulation dataset, 0.5 s at 1 kHz, split 5000 / 500 / 200.
- **Model**: a spectral operator surrogate with 257,024 trainable parameters.
- **Training**: SGD, 100 epochs, about 60 seconds on CPU.
- **Result**: global relative L2 error on the test split `5.940277290056465e-08`
  (threshold `1e-4`).

The dataset comes from numerical simulation. It is **not** measurement data from a real
device.

</details>

## 2. 任务定义 / Task Definition

固定 1 kHz 时间栅格上的扰动力反演问题：

```
raw input : 执行器力 (4) + 光栅位移读数 (4)  ->  8 通道
target    : 外部扰动力 (X, Y)                ->  2 通道
raw task  : [501, 8]  ->  [501, 2]            (0.5 s @ 1 kHz)
```

<details>
<summary>English</summary>

A disturbance-force inversion problem on a fixed 1 kHz time grid:

```
raw input : actuator force (4) + encoder/grating displacement reading (4)  ->  8 channels
target    : external disturbance force (X, Y)                              ->  2 channels
raw task  : [501, 8]  ->  [501, 2]                              (0.5 s at 1 kHz)
```

</details>

## 3. 数据集 / Dataset

canonical 数据集路径：

```
/nishome/charliewang/data/oled_microstage_inverse_disturbance
```

| 项 / Item | 值 / Value |
|---|---|
| manifest schema | `neural_operator_dataset_manifest/3.0` |
| HDF5 schema | `5.0` |
| 时长 · 采样 / duration · sampling | 0.5 s @ 1 kHz，501 时间点 / 501 time points |
| 划分 / split | train 5000 / validation 500 / test 200（manifest 定义，互不重叠 / manifest-defined, no overlap） |
| 求解器 / solver | Exudyn GeneralizedAlpha，`preroll_s = 0.5`，`integration_substeps = 5` |
| 推荐问题 / recommended problem | `inverse_disturbance` |

版本信息由数据集 manifest 与每个样本的 `/metadata` 组承担（生成器 git commit、
求解器、依赖版本、config hash）；目录名不带版本号。

数据集位置可用 `--data-root` 或 `$DATA_ROOT` 覆盖；测试配置记录了 canonical 路径。

<details>
<summary>English</summary>

Version information lives in the dataset manifest and in each sample's `/metadata` group
(generator git commit, solver, dependency versions, config hash); the directory name carries
no version number.

Override the location with `--data-root` or `$DATA_ROOT`; the test configuration documents
the canonical path.

</details>

## 4. 输入输出契约 / Input–Output Contract

**raw 任务是 `[501, 8] -> [501, 2]`。** 模型内部的特征张量宽 16 通道；多出的 8 个
通道由 8 路原始输入**确定性导出**（见 §5），不是额外的数据。

| 阶段 / stage | 宽度 / width | 内容 / contents |
|---|---|---|
| raw 输入 / raw input | **8** | `force` (4) + `encoder_displacement` (4)，从 HDF5 读入 |
| 模型内部特征 / internal features | 16 | raw 8 + 一阶差分 (4) + 二阶差分 (4) |
| 输出 / output | **2** | `disturbance/force` (X, Y) |

模型的输入只由 `data.input_fields` 选定的 `force` 与 `encoder_displacement` 构成；
状态量（`q`、`q_dot`、`q_ddot`）与扰动力目标不参与输入构造。

<details>
<summary>English</summary>

**The raw task is `[501, 8] -> [501, 2]`.** The model's internal feature tensor is 16
channels wide; the extra eight are **derived deterministically** from the eight raw inputs
(see §5) and are not additional data.

The model input is built only from `force` and `encoder_displacement`, selected by
`data.input_fields`. State variables (`q`, `q_dot`, `q_ddot`) and the disturbance target take
no part in constructing the input.

</details>

## 5. 特征预处理 / Feature Preprocessing

`diff_features`（`src/data/preprocessing.py`）：对四路光栅位移读数通道追加 3 点一阶
与二阶中心差分，把 8 通道扩展为 16 通道。

该变换在 **float64** 下运行（`data.transform_dtype: float64`）。HDF5 中的值是 float64，
而二阶差分需要除以 `dt²`；若先降精度再差分，会注入一个 `eps32 · |x| / dt²` 量级的
量化误差，此后任何上转型都无法消除。

追加差分行提供的是有限差分加速度估计，用于覆盖光栅通道的起步漂移斜坡。

<details>
<summary>English</summary>

`diff_features` (`src/data/preprocessing.py`) appends the 3-point first and second central
differences of the four encoder channels, growing 8 channels to 16.

The transform runs in **float64** (`data.transform_dtype: float64`). The HDF5 values are
float64 and the second difference divides by `dt²`; downgrading before differentiating would
inject quantisation error of order `eps32 · |x| / dt²` that no later upcast can undo.

The appended difference rows provide finite-difference acceleration estimates, which cover
the start-up drift ramp of the encoder channels.

</details>

## 6. 算子代理模型 / Operator Surrogate Model

本项目采用**基于谱域表示的算子代理模型（spectral operator surrogate）**。

模型首先基于训练数据拟合一个**固定的谱域特征基**（白化统计量 `sd` / `Vd` / `Sd` /
`sy`，只由 train split 计算），随后由一个**可训练的稠密映射** `W` 学习输入时序到
外部扰动力时序的映射。

- 实现：`SpectralDenseMap`（`src/models/spectral_dense.py`）——rfft → 固定白化 →
  可训练稠密映射 → irfft，全程 float64。
- 可训练参数：唯一参数 `W`，形状 `256 × 1004`，共 **257,024** 个。
- 固定部分：谱域特征基以非训练 buffer 形式随模型保存。

<details>
<summary>English</summary>

This project uses a **spectral operator surrogate**.

A **fixed spectral feature basis** is fitted from the training data (the whitening
statistics `sd` / `Vd` / `Sd` / `sy`, computed from the train split only). A **trainable
dense map** `W` then learns the mapping from the input time series to the external
disturbance-force time series.

- Implementation: `SpectralDenseMap` (`src/models/spectral_dense.py`) — rfft → fixed
  whitening → trainable dense map → irfft, float64 throughout.
- Trainable parameters: a single parameter `W` of shape `256 × 1004`, **257,024** in total.
- Fixed part: the spectral basis is stored with the model as non-trainable buffers.

</details>

## 7. 拟合训练 / Training

| 项 / Item | 值 / Value |
|---|---|
| 优化器 / optimizer | **SGD**，lr `2.684e5`，momentum 0.0，weight decay 0.0 |
| 调度器 / scheduler | cosine annealing to 0 |
| epoch 数 / epochs | 100 |
| 损失 / loss | MSE |
| 批大小 / batch | 全训练批 / full train batch (5000)，评估批 / eval batch 128 |
| 验证 / validation | 每 5 个 epoch / every 5 epochs |
| 最优 checkpoint / best checkpoint | 按验证 MSE 最低 / lowest **validation MSE** |
| 设备 / device | **CPU** |
| 墙钟时间 / wall clock | ~60 s |

训练始终从**随机初始化**开始，`loss.backward()` 与 `optimizer.step()` 每个 epoch
各执行一次。

两条操作约束：

- **学习率按全批 5000 样本标定。** MSE 对 batch 取均值，因此损失 Hessian 按
  `1/numel` 缩放；在明显更小的子集上使用 canonical 学习率不会收敛。
  `--max-train-samples` 是接线检查模式，不是测试模式。
- **训练流程不含 resume 路径**，每次运行都是一次完整的从零训练。

关于硬件：本机上存在 2 × NVIDIA RTX 4090（24 GB），但本代理模型的 canonical 执行
路径是 **CPU**，上述时间即为 CPU 实测值。

<details>
<summary>English</summary>

Training always starts from a **random initialization**; `loss.backward()` and
`optimizer.step()` run once per epoch.

Two operational constraints:

- **The learning rate is calibrated for the full 5000-sample batch.** MSE averages over the
  batch, so the loss Hessian scales as `1/numel`; the canonical learning rate will not
  converge on a much smaller subset. `--max-train-samples` is a wiring-check mode, not a
  test mode.
- **The training pipeline has no resume path**; every run is a complete from-scratch
  training.

About the hardware: this machine has 2 × NVIDIA RTX 4090 (24 GB), but the canonical
execution path for this surrogate is **CPU**, and the timing above is measured on CPU.

</details>

## 8. 测试指标 / Test Metric

覆盖**整个测试集一次性聚合**的全局相对 L2 误差：

```
sqrt( Σ 全部测试样本、全部时间点、两个通道 (prediction - target)² )
--------------------------------------------------------------------
sqrt( Σ 全部测试样本、全部时间点、两个通道  target² )
```

它覆盖全部 200 个测试样本 × 501 个时间点 × 2 个通道 = 200,400 个元素。

**测试阈值：`1e-4`。**

会记录两个值：

- `relative_l2` —— 与 `src/training/metrics.py` 一致的计算方式，累加前提升到
  float32。
- `relative_l2_float64` —— 同一预测在 float64 下累加，即真实模型误差。

float32 累加大约比真值高 1.27 倍，两者都作为测试结果保存。

<details>
<summary>English</summary>

**Global** relative L2 over the whole test split at once, covering all 200 test samples ×
501 time points × 2 channels = 200,400 elements.

**Test threshold: `1e-4`.**

Two values are recorded: `relative_l2`, computed as `src/training/metrics.py` does with a
float32 promotion before accumulating; and `relative_l2_float64`, the same prediction
accumulated in float64, which is the true model error. The float32 accumulation is about
1.27× the true value. Both are saved as the test result.

</details>

## 9. 已验证结果 / Verified Result

| 口径 / path | 值 / value |
|---|---|
| `relative_l2`（float32 累加）/ float32 accumulation | **`5.940277290056465e-08`** |
| `relative_l2_float64`（float64 复算）/ float64 recomputation | **`4.672084296433188e-08`** |
| 测试阈值 / test threshold | `1e-4` |
| 测试结果 / test result | **PASS** |
| 余量 / margin | 1,683× / 2,140× |

每次测试运行都会把自身结果写入 `<run-dir>/test_metrics.json`。

<details>
<summary>English</summary>

Each test run writes its own result to `<run-dir>/test_metrics.json`.

</details>

## 10. 快速开始 / Quick Start

```bash
cd /nishome/charliewang/forge-projects/oled-neural-operator
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests      # 47 tests
```

完整的测试命令序列见 [TESTING.md](TESTING.md)——该文件是唯一可复制粘贴的执行流程，
每条命令在其中只出现一次。

<details>
<summary>English</summary>

The block above creates the project-local environment, installs the dependencies and runs
the 47-test suite.

The full test command sequence is **not** repeated here: it lives in [TESTING.md](TESTING.md),
which is the single copy-paste procedure and in which every command appears exactly once.

</details>

## 11. 仓库结构 / Repository Structure

```
configs/test.yaml              测试配置（唯一）/ the test configuration (the only one)
requirements.txt               torch / numpy / h5py / matplotlib
scripts/
  show_environment.py          STEP 0  环境记录 -> environment.json/.txt
  inspect_dataset.py           STEP 1  数据集摘要 -> dataset_summary.json
  fit_basis.py                 STEP 3  训练集专用谱基 -> spectral_basis.pt
  train.py                     STEP 4  SGD 训练 -> best_model.pt, history.json
  evaluate.py                  STEP 5  测试集推理与指标 -> test_metrics.json, predictions.npz
  report.py                    STEP 6  图表 -> figures/*.png
src/
  data/                        HDF5 数据集、特征预处理、DataLoader、内存缓存
  models/                      SpectralDenseMap / GridAdapter / model factory
  training/                    trainer、epoch 循环、验证、指标、优化器构造
  utils/                       配置、checkpoint、运行日志、设备、路径、随机种子
tests/                         unittest（标准库），含端到端管线冒烟测试
runs/                          生成的测试运行产物（不入 git，保留 .gitkeep）
```

<details>
<summary>English</summary>

The tree above is the whole repository: one test configuration, the dependency spec, six
scripts (one per step of the procedure), the four source subpackages, the test suite (stdlib
`unittest`, including an end-to-end pipeline smoke test), and `runs/` for generated test-run
artifacts (gitignored except for its sentinel file).

</details>

## 12. 测试流程 / Test Procedure

完整的、可直接复制粘贴的测试流程，含每一步的预期输出，见 [TESTING.md](TESTING.md)。

<details>
<summary>English</summary>

The complete, copy-paste test procedure — with the expected output of every step — is
[TESTING.md](TESTING.md).

</details>
