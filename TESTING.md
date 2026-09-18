# OLED 微动台扰动力代理模型测试流程 / OLED Microstage Disturbance Surrogate Test Procedure

本文件是**可逐行执行**的测试流程。命令已针对本服务器验证，可直接复制粘贴。

```text
第一部分（数据集生成）：/nishome/charliewang/forge-projects/oled-microstage-simulation
                        → 见该仓库的 TESTING.md
第二部分（本仓库）    ：扰动力代理模型测试
```

**canonical 数据集 / canonical dataset**：`/nishome/charliewang/data/oled_microstage_inverse_disturbance`
**canonical 设备 / canonical device**：`cpu`

<details>
<summary>English</summary>

This is the line-by-line executable test procedure. The commands have been verified against
this server and can be copied and pasted directly.

Part one (dataset generation) is in the `oled-microstage-simulation` repository and has its
own `TESTING.md`. Part two (this repository) is the disturbance surrogate test.

</details>

---

> 数据说明 / Data note
>
> 本仓库不含项目物理模型参数。`configs/test.yaml` 指向的冻结数据集中，
> `dataset_manifest.json` 记录了项目物理模型参数，一般测试流程无需直接查看这些文件；
> 推荐使用项目提供的数据集摘要命令获取结构与统计信息。
>
> <details>
> <summary>English</summary>
>
> This repository contains no project physical-model parameters. Inside the frozen dataset
> referenced by `configs/test.yaml`, the `dataset_manifest.json` records the project
> physical-model parameters. They are not required for the standard test workflow; use the
> provided dataset-summary command for structural and statistical information.
>
> </details>

---

## STEP 0. 记录软硬件环境 / Record the Software and Hardware Environment

```bash
cd /nishome/charliewang/forge-projects/oled-neural-operator
RUN=runs/test_$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN"
echo "RUN=$RUN"

.venv/bin/python scripts/show_environment.py \
    --config configs/test.yaml --run-dir "$RUN"
```

输出包含 SYSTEM / GPU / SURROGATE ENVIRONMENT / SIMULATION ENVIRONMENT / CODE 五段，
并保存：

```
$RUN/environment.json   机器可读 / machine-readable
$RUN/environment.txt    人可读 / human-readable
```

硬件说明：本机存在 **2 × NVIDIA RTX 4090 (24 GB)**，本代理模型管线的 canonical 执行
设备是 **CPU**；venv 中的 torch 是 CPU-only 构建，`torch.cuda.is_available()` 为
`False`。完整训练约 60 秒。

数据生成侧（另一仓库）使用独立的 venv，含 Exudyn 1.10.0 与 SciPy。

<details>
<summary>English</summary>

The output has five blocks (SYSTEM / GPU / SURROGATE ENVIRONMENT / SIMULATION
ENVIRONMENT / CODE) and writes `environment.json` and `environment.txt` into the run
directory.

Hardware note: this machine has **2 × NVIDIA RTX 4090 (24 GB)**; the canonical execution
device for this surrogate pipeline is **CPU**, the venv carries a CPU-only torch build, and
`torch.cuda.is_available()` is `False`. Full training takes about 60 seconds.

The data generation side (the other repository) uses its own venv containing Exudyn 1.10.0
and SciPy.

</details>

## STEP 1. 检查冻结数据集 / Inspect the Frozen Dataset

```bash
.venv/bin/python scripts/inspect_dataset.py \
    --config configs/test.yaml --run-dir "$RUN"
```

期望输出：

```text
Frozen test dataset
  Dataset             : oled_microstage_inverse_disturbance
  Path                : /nishome/charliewang/data/oled_microstage_inverse_disturbance
  Manifest schema     : neural_operator_dataset_manifest/3.0
  HDF5 schema         : 5.0
  Complete            : True
  Train               : 5000
  Validation          : 500
  Test                : 200
  Duration            : 0.5 s
  Sampling rate       : 1000 Hz
  Time points         : 501
  Raw input channels  : 8  (force 4 + encoder_displacement 4)
  Target channels     : 2  (disturbance 2)
  Raw task shape      : [501, 8] -> [501, 2]
  Preroll             : 0.5 s
  Integration substeps: 5
  Recommended problem : inverse_disturbance
```

同时保存 `$RUN/dataset_summary.json`。

<details>
<summary>English</summary>

The command prints the reduced dataset summary shown above and writes the same content to
`$RUN/dataset_summary.json`. It reads only the dataset manifest, so no sample is opened.

</details>

## STEP 2. 核验 train / validation / test / Verify the Split

划分由数据集 manifest 定义（按全局索引连续切分，互不重叠），上一步已显示
`5000 / 500 / 200`。STEP 3、STEP 4、STEP 5 会分别打印实际加载的样本数，三者必须一致。

<details>
<summary>English</summary>

The split is defined by the dataset manifest (contiguous global indices, non-overlapping)
and the previous step already showed `5000 / 500 / 200`. Steps 3, 4 and 5 each print the
number of samples they actually load; the three must agree.

</details>

## STEP 3. 使用训练集拟合谱基 / Fit the Spectral Basis from the Train Split

```bash
time .venv/bin/python scripts/fit_basis.py \
    --config configs/test.yaml --run-dir "$RUN"
```

期望输出（约 11 秒 / roughly 11 s）：

```text
Fitting basis on the train split only: 5000 samples

Train-only spectral basis fitted
  Train samples       : 5000
  Feature width       : 3232
  Target width        : 1004
  Retained directions : 256
  Input channels      : 16
  Output channels     : 2
  Elapsed             : 10.3 s
  Wrote               : runs/test_.../spectral_basis.pt
```

本步骤**只读取 train split**，产出固定的谱域特征基（白化统计量 `sd` / `Vd` / `Sd` /
`sy`），不涉及对目标的拟合。可训练映射 `W` 由 STEP 4 的梯度训练得到。

<details>
<summary>English</summary>

This step reads the **train split only** and produces the fixed spectral feature basis (the
whitening statistics `sd` / `Vd` / `Sd` / `sy`). It performs no fitting against the target;
the trainable map `W` comes from the gradient training in STEP 4.

</details>

## STEP 4. 从随机初始化执行 SGD 训练与验证 / Train from a Random Initialization with SGD

```bash
time .venv/bin/python scripts/train.py \
    --config configs/test.yaml --run-dir "$RUN" --device cpu
```

终端每个 epoch 打印一段：

```text
Epoch 020/100
  Train MSE         : 5.6449537e-08
  Train relative L2 : 1.2458411e-04
  Learning rate     : 2.427701e+05
  Val   MSE         : 2.4321565e-08
  Val   relative L2 : 8.2471659e-05
  Best epoch        : 20 (val MSE 2.4321565e-08)
```

结束时打印：

```text
Training complete
  Epochs            : 100
  Best epoch        : 100
  Best val MSE      : ...
  Best val relative L2: ...
  Wall clock        : ~61 s
```

训练从随机初始化开始，使用 SGD 迭代 100 个 epoch，每 5 个 epoch 验证一次，并按验证
MSE 选择最优 checkpoint。

产物：`config.yaml`、`history.json`、`metrics.csv`、`train.log`、`best_model.pt`、
`last_model.pt`。

<details>
<summary>English</summary>

Training starts from a random initialization, runs SGD for 100 epochs, validates every 5
epochs, and selects the best checkpoint by validation MSE.

Artifacts: `config.yaml`, `history.json`, `metrics.csv`, `train.log`, `best_model.pt`,
`last_model.pt`.

</details>

## STEP 5. 测试集推理 / Test-Set Inference

```bash
.venv/bin/python scripts/evaluate.py --run-dir "$RUN" --device cpu
```

该命令加载 `best_model.pt`，对**全部 200 个测试样本**、全部 501 个时间点、两个输出
通道执行推理，并保存逐样本预测：

```
$RUN/predictions.npz     predictions / targets，形状 (200, 501, 2)
```

<details>
<summary>English</summary>

The command loads `best_model.pt` and runs inference over **all 200 test samples**, all 501
time points and both output channels, saving the per-sample predictions to
`$RUN/predictions.npz` with shape `(200, 501, 2)`.

</details>

## STEP 6. 计算并保存 global relative L2 / Compute and Save the Global Relative L2

同一命令在推理之后计算并保存测试指标：

```
$RUN/test_metrics.json
```

终端输出：

```text
Test evaluation
  Splits              : test
  Samples             : 200
  Time points         : 501
  Target channels     : 2
  Elements            : 200400
  MSE                 : ...
  RMSE                : ...

  GLOBAL RELATIVE L2  : 5.9...e-08   (official, float32 accumulation)
  GLOBAL RELATIVE L2  : 4.6...e-08   (float64 direct recomputation)
  TEST THRESHOLD      : 1.000000e-04
  RESULT              : PASS
```

指标口径：`relative_l2` 与 `src/training/metrics.py` 一致，累加前提升到 float32；
`relative_l2_float64` 为同一预测在 float64 下的复算，即真实模型误差。两者都远低于
`1e-4` 的测试阈值。

<details>
<summary>English</summary>

The same command computes and saves the test metric after inference.

Metric semantics: `relative_l2` matches `src/training/metrics.py` and promotes to float32
before accumulating; `relative_l2_float64` is the same prediction recomputed in float64,
which is the true model error. Both are far below the `1e-4` test threshold.

</details>

## STEP 7. 生成训练曲线、预测曲线和误差曲线 / Generate Curves

```bash
.venv/bin/python scripts/report.py --run-dir "$RUN"
```

写出 `$RUN/figures/`：

```
training_curve.png             训练 MSE / 验证 MSE 对 epoch
validation_curve.png           验证 relative L2 对 epoch（含 1e-4 参考线）
prediction_sample_001..003.png Fx / Fy 真值与预测对比
error_sample_001..003.png      Fx / Fy 残差时序
```

三个示例样本固定为测试集的**第一个、中间、最后一个**（索引 0 / 100 / 199），按位置
选取。

<details>
<summary>English</summary>

The three example samples are the **first, middle and last** of the test split
(indices 0 / 100 / 199), fixed by position.

</details>

---

## 产物位置 / Artifact Locations

```text
$RUN/
├── environment.json / environment.txt      STEP 0
├── dataset_summary.json                    STEP 1
├── config.yaml                             STEP 4（解析后的配置 / resolved configuration）
├── spectral_basis.pt                       STEP 3（train-only）
├── best_model.pt / last_model.pt           STEP 4
├── history.json / metrics.csv / train.log  STEP 4
├── test_metrics.json / predictions.npz     STEP 5 / STEP 6
└── figures/*.png                           STEP 7
```

<details>
<summary>English</summary>

The tree above is the complete test-run directory contract; every item is produced by the
step named beside it.

</details>

## 单元测试 / Unit Tests

```bash
.venv/bin/python -m unittest discover -s tests    # 47 项，含端到端管线冒烟测试
```

<details>
<summary>English</summary>

Runs the 47-test suite, which includes an end-to-end pipeline smoke test on a synthetic
miniature dataset.

</details>

## 步骤索引 / Step Index

每条命令在本文件中只出现一次，位于下表的对应小节。

| 步骤 / Step | 小节 / Section | 命令 / Command |
|---|---|---|
| STEP 0 环境记录 | STEP 0 | `scripts/show_environment.py` |
| STEP 1 数据集检查 | STEP 1 | `scripts/inspect_dataset.py` |
| STEP 2 划分核验 | STEP 2 | （沿用 STEP 1 输出 / uses the STEP 1 output） |
| STEP 3 训练集专用谱基 | STEP 3 | `scripts/fit_basis.py` |
| STEP 4 从零训练 + 验证 | STEP 4 | `scripts/train.py --device cpu` |
| STEP 5 测试集推理 | STEP 5 | `scripts/evaluate.py` |
| STEP 6 全局相对 L2 | STEP 6 | （同一命令的输出 / output of the same command） |
| STEP 7 曲线 | STEP 7 | `scripts/report.py` |

<details>
<summary>English</summary>

Each command appears exactly once in this document, in the section listed above. This index
deliberately repeats no command block; follow the section links instead.

</details>
