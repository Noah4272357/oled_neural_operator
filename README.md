# OLED 微动台外部扰动力代理模型 / OLED Microstage Disturbance Surrogate

面向 OLED 喷墨打印微动台的**外部扰动力反演**代理模型：由执行器力与光栅位移读数
轨迹，预测作用在运动刚体上的外部扰动力轨迹。

模型在配套仓库 [`oled-microstage-simulation`](../oled-microstage-simulation) 生成的
冻结合成基准上训练。本仓库是正式验收的两个 repository 之一，现场执行步骤见
[ACCEPTANCE.md](ACCEPTANCE.md)。

<details>
<summary>English</summary>

A surrogate model for **external disturbance force inversion** on the OLED inkjet-printer
microstage: from the actuator force and encoder/grating displacement reading trajectories,
predict the external disturbance force trajectory acting on the moving body.

The model is trained on the frozen synthetic benchmark produced by the companion repository
[`oled-microstage-simulation`](../oled-microstage-simulation). This is one of the two
repositories delivered at formal acceptance; the on-site procedure is
[ACCEPTANCE.md](ACCEPTANCE.md).

</details>

---

## 1. 任务定义 / Task Definition

固定 1 kHz 时间栅格上的反问题（扰动力反演）：

```
raw input  ：执行器力 (4)  +  光栅位移读数 (4)     ->  8 通道
target     ：外部扰动力 (X, Y)                    ->  2 通道
raw task   ：[501, 8]  ->  [501, 2]         （0.5 s，1 kHz）
```

扰动信号是**合成基准信号，不是设备实测**。

<details>
<summary>English</summary>

Inverse (disturbance inversion) problem on a fixed 1 kHz time grid:

```
raw input : actuator force (4) + encoder/grating displacement reading (4)  ->  8 channels
target    : external disturbance force (X, Y)                              ->  2 channels
raw task  : [501, 8]  ->  [501, 2]                       (0.5 s at 1 kHz)
```

The disturbance is a **synthetic benchmark signal, never a device measurement**.

</details>

## 2. 数据集 / Dataset

canonical 验收数据集：

```
/nishome/charliewang/data/oled_microstage_inverse_disturbance
```

| 项 / Item | 值 / Value |
|---|---|
| manifest schema | `neural_operator_dataset_manifest/3.0` |
| HDF5 schema | `5.0` |
| 时长 · 采样 / duration · sampling | 0.5 s @ 1 kHz，501 时间点 / 501 time points |
| 划分 / split | train 5000 / val 500 / test 200（manifest 定义，互不重叠 / manifest-defined, no overlap） |
| 求解器 / solver | Exudyn GeneralizedAlpha，`preroll_s = 0.5`，`integration_substeps = 5` |
| 推荐问题 / recommended problem | `inverse_disturbance` |

版本信息由数据集 manifest 与每个样本的 `/metadata` 组承担（生成器 git commit、
求解器、依赖版本、config hash）。目录名刻意不带版本号。

用 `--data-root` 或 `$DATA_ROOT` 覆盖数据集位置；配置文件记录了 canonical 路径。

<details>
<summary>English</summary>

The canonical acceptance dataset is at the path above. Its version information lives in the
dataset manifest and in each sample's `/metadata` group (generator git commit, solver,
dependency versions, config hash); the directory name deliberately carries no version
number.

Override the location with `--data-root` or `$DATA_ROOT`; the configuration documents the
canonical path.

</details>

## 3. 输入 / 输出契约 / Input / Output Contract

**raw 任务是 `[501, 8] -> [501, 2]`。** 模型的内部输入张量宽 16 通道，多出的 8 个
通道是**由 8 路原始输入确定性导出**的（见 §4），不是额外的数据。

| 阶段 / stage | 宽度 / width | 内容 / contents |
|---|---|---|
| raw 验收输入 / raw acceptance input | **8** | `force` (4) + `encoder_displacement` (4)，从 HDF5 读入 |
| 模型内部特征 / model internal features | 16 | raw 8 + 一阶差分 (4) + 二阶差分 (4) |
| 目标 / target | **2** | `disturbance/force` (X, Y) |

任何状态量（`q`、`q_dot`、`q_ddot`）与目标都不会进入模型输入：`data.input_fields`
只选择 `force` 与 `encoder_displacement`，预处理变换不再读取其他内容。

<details>
<summary>English</summary>

**The raw task is `[501, 8] -> [501, 2]`.** The model's internal input tensor is 16
channels wide; the extra eight are **derived deterministically from the eight raw inputs**
(see §4) and are not additional data.

No state variable (`q`, `q_dot`, `q_ddot`) and no target ever enters the model input:
`data.input_fields` selects `force` and `encoder_displacement` only, and the preprocessing
transform reads nothing else.

</details>

## 4. 最终预处理 / Final Preprocessing

`diff_features`（`src/data/preprocessing.py`）：对四路光栅位移读数通道追加 3 点
一阶与二阶中心差分，把 8 通道扩展为 16 通道。

它在 **float64** 下运行（`data.transform_dtype: float64`）。这是必需的而非装饰性的：
HDF5 中的值是 float64，而二阶差分要除以 `dt²`，若先降精度再差分，会注入一个
`eps32 · |x| / dt²` 量级的量化地板，此后任何上转型都无法消除。

派生通道存在的原因：光栅通道带有起步漂移斜坡，追加的差分行是有限差分加速度估计。

<details>
<summary>English</summary>

`diff_features` appends the 3-point first and second central differences of the four
encoder channels, growing 8 channels to 16.

It runs in **float64** (`data.transform_dtype: float64`). This is required, not cosmetic:
the HDF5 values are float64 and the second difference divides by `dt²`, so downgrading
before differentiating injects a quantisation floor of order `eps32 · |x| / dt²` that no
later upcast can undo.

The derived channels exist because the encoder channels carry a start-up drift ramp; the
appended difference rows are the finite-difference acceleration estimates.

</details>

## 5. 最终模型 / Final Model

`SpectralDenseMap`（`src/models/spectral_dense.py`）：rfft → 固定白化 → 可训练稠密
映射 → irfft，全程 float64。

- **唯一可训练参数** `W`，形状 `256 × 1004` = **257,024 参数**。
- 固定的非训练 buffer `sd`、`Vd`、`Sd`、`sy` 来自训练集专用的谱基。
- 它**不是**傅里叶神经算子（FNO）。它是作用在展平逐 bin 频谱上的稠密线性映射，
  没有非线性激活。

<details>
<summary>English</summary>

`SpectralDenseMap` (`src/models/spectral_dense.py`): rfft → fixed whitening → trainable
dense map → irfft, float64 end to end.

- **One trainable parameter**, `W`, of shape `256 × 1004` = **257,024 parameters**.
- The fixed, non-trainable buffers `sd`, `Vd`, `Sd`, `sy` come from the train-only spectral
  basis.
- It is **not** a Fourier Neural Operator (FNO). It is a dense linear map on the flattened
  per-bin spectrum and has no nonlinear activation.

</details>

## 6. 梯度训练 / Gradient Training

| 项 / Item | 值 / Value |
|---|---|
| 优化器 / optimizer | **SGD**，lr `2.684e5`，momentum 0.0，weight decay 0.0 |
| 调度器 / scheduler | cosine annealing to 0 |
| epoch 数 / epochs | 100 |
| 损失 / loss | MSE |
| 批大小 / batch | 全训练批 / full train batch (5000)，评估批 / eval batch 128 |
| 验证 / validation | 每 5 个 epoch / every 5 epochs |
| 最优 checkpoint / best checkpoint | 按验证 MSE 最低 / lowest **validation MSE** |
| 设备 / device | **CPU**（canonical 验收路径 / the canonical acceptance path） |
| 墙钟时间 / wall clock | ~60 s |

`loss.backward()` 与 `optimizer.step()` 每个 epoch 各执行一次；训练从随机初始化
开始，且从不 resume。

两条操作事实：

- **不要从 checkpoint 微调。** 交付模型是单次从零训练的产物；该类模型的 resume
  微调已知无益，且管线按设计没有 resume 路径。
- **学习率是按全批 5000 样本标定的。** MSE 对 batch 取均值，因此损失 Hessian 按
  `1/numel` 缩放；在小规模子集上用 canonical 学习率会发散。`--max-train-samples`
  是接线检查模式，不是验收模式。

<details>
<summary>English</summary>

`loss.backward()` and `optimizer.step()` run once per epoch; the run starts from a fresh
random initialization and never resumes.

Two operational facts:

- **Do not fine-tune from a checkpoint.** The delivered model is the product of a single
  from-scratch run; resume fine-tuning of this model class is known to not help, and the
  pipeline has no resume path by design.
- **The learning rate is calibrated for the full 5000-sample batch.** MSE averages over the
  batch, so the loss Hessian scales as `1/numel`; training on a small subset with the
  canonical lr diverges. `--max-train-samples` is a wiring-check mode, not an acceptance
  mode.

</details>

## 7. 验收指标 / Acceptance Metric

覆盖**整个测试集一次性聚合**的全局相对 L2 误差：

```
sqrt( Σ 全部测试样本、全部时间点、两个通道 (prediction - target)² )
--------------------------------------------------------------------
sqrt( Σ 全部测试样本、全部时间点、两个通道  target² )
```

不是先算每个样本的相对 L2 再平均。它覆盖全部 200 个测试样本 × 501 个时间点 ×
2 个通道 = 200,400 个元素。

**验收阈值：`1e-4`。**

会记录两个值，因为它们不同且差异已被理解：

- `relative_l2` —— 与 `src/training/metrics.py` 完全一致的计算方式，累加前提升到
  float32。**这是官方数值。**
- `relative_l2_float64` —— 同一预测在 float64 下累加。**这是真实模型误差。**
  float32 累加大约高报 1.27 倍。

`pass` 要求**两者都**低于阈值。

<details>
<summary>English</summary>

**Global** relative L2 over the whole test split at once — not a per-sample relative L2
averaged afterwards. It covers all 200 test samples × 501 time points × 2 channels =
200,400 elements.

**Threshold: `1e-4`.**

Two values are recorded, because they differ and the difference is understood:
`relative_l2` is computed exactly as `src/training/metrics.py` does (promoting to float32
before accumulating) and is the official number; `relative_l2_float64` is the same
prediction accumulated in float64 and is the true model error. The float32 accumulation
over-reports by ~1.27×. `pass` requires **both** to be under the threshold.

</details>

## 8. 当前验证结果 / Current Verified Result

| 口径 / path | 值 / value |
|---|---|
| 官方（`relative_l2`，float32 累加）/ official (float32 accumulation) | **`5.940277290056465e-08`** |
| float64 直接复算 / float64 direct recomputation | **`4.672084e-08`** |
| 阈值 / threshold | `1e-4` |
| 余量 / margin | 官方 1,683× / 真值 2,140× |

每次验收运行都会把自身结果写入 `<run-dir>/test_metrics.json`。上表是已记录的参考值；
重新排练会复现同一结果（见 ACCEPTANCE.md）。

<details>
<summary>English</summary>

Each acceptance run writes its own result to `<run-dir>/test_metrics.json`. The figures
above are the recorded reference; a fresh rehearsal reproduces the same result (see
ACCEPTANCE.md).

</details>

## 9. 快速开始 / Quick Start

```bash
cd /nishome/charliewang/forge-projects/oled-neural-operator
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests      # 47 tests
```

正式的验收命令序列不在本文件重复，见 [ACCEPTANCE.md](ACCEPTANCE.md)——该文件是
唯一可复制粘贴的现场执行手册，每条命令在其中只出现一次。

<details>
<summary>English</summary>

The block above creates the project-local environment, installs the dependencies and runs
the 47-test suite.

The formal acceptance command sequence is **not** repeated here: it lives in
[ACCEPTANCE.md](ACCEPTANCE.md), which is the single copy-paste runbook and in which every
command appears exactly once.

</details>

## 10. 仓库结构 / Repository Structure

```
configs/acceptance.yaml        最终配置（唯一）/ the final configuration (the only one)
requirements.txt               torch / numpy / h5py / matplotlib
scripts/
  show_environment.py          STEP 0  环境记录 -> environment.json/.txt
  inspect_dataset.py           STEP 1  数据集摘要 -> dataset_summary.json
  fit_basis.py                 STEP 3  训练集专用谱基 -> spectral_basis.pt
  train.py                     STEP 4  SGD 训练 -> best_model.pt, history.json
  evaluate.py                  STEP 5  全测试集评估 -> test_metrics.json, predictions.npz
  report.py                    STEP 6  图表 -> figures/*.png
src/
  data/                        HDF5 数据集、预处理、DataLoader、内存缓存
  models/                      SpectralDenseMap + GridAdapter + factory
  training/                    trainer、epoch 循环、验证、指标、优化器工厂
  utils/                       配置、checkpoint、run 日志、设备、路径、随机种子
tests/                         unittest（标准库）；含一个端到端验收冒烟测试
runs/                          生成的验收产物（不入 git）
```

<details>
<summary>English</summary>

The tree above is the whole repository: one final configuration, the dependency spec, six
acceptance-facing scripts (one per step of the runbook), the four source subpackages, the
test suite (stdlib `unittest`, including one end-to-end acceptance smoke test), and `runs/`
for generated acceptance artifacts. `runs/` is gitignored except for its sentinel file.

</details>

## 11. 正式验收流程 / Formal Acceptance Workflow

完整的、可直接复制粘贴的现场终端流程，含每一步的预期输出，见
[ACCEPTANCE.md](ACCEPTANCE.md)。

<details>
<summary>English</summary>

The complete, copy-paste, on-site terminal procedure — with the expected output of every
step — is [ACCEPTANCE.md](ACCEPTANCE.md).

</details>
