# OLED 微动台验收流程 — 代理模型拟合与测试

本文件是**现场可逐行执行**的终端手册。命令已针对本服务器验证，可直接复制粘贴。

```text
第一部分（数据生成）  ：/nishome/charliewang/forge-projects/oled-microstage-simulation
                        → 见该仓库的 ACCEPTANCE.md
第二部分（本仓库）    ：代理模型拟合与测试
```

**canonical 数据集**：`/nishome/charliewang/data/oled_microstage_inverse_disturbance`
**canonical 设备**：`cpu`

> ⚠️ 演示纪律
>
> - 本仓库**不含**企业物理参数；数据集的 `dataset_manifest.json` 含这些数值，
>   但本手册的全部命令都不会打印它们，也不要求打开该文件。
> - 不要打开 `configs/acceptance.yaml` 之外的历史目录。
> - 首次运行前请确认 `runs/` 下没有同名 run 目录。

---

## 0. 激活环境

```bash
cd /nishome/charliewang/forge-projects/oled-neural-operator
.venv/bin/python --version          # 期望 Python 3.10.12
```

本仓库使用项目本地 `.venv`（torch 2.5.1+cpu，CPU-only），不修改系统 /
user site-packages。

## 1. 展示环境（STEP 0）

```bash
RUN=runs/acceptance_$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN"
echo "RUN=$RUN"

.venv/bin/python scripts/show_environment.py \
    --config configs/acceptance.yaml --run-dir "$RUN"
```

现场输出包含 SYSTEM / GPU / SURROGATE ENVIRONMENT / SIMULATION ENVIRONMENT / CODE
五段，并保存：

```
$RUN/environment.json   机器可读
$RUN/environment.txt    人可读
```

需要说明的两点（不要回避）：

- 本机确有 **2 × NVIDIA RTX 4090 (24 GB)**，但代理模型管线**按设计在 CPU 上运行**，
  venv 中的 torch 是 CPU-only 构建，`torch.cuda.is_available()` 为 `False`。
  完整训练约 60 秒，不需要 GPU。
- 数据生成侧（另一仓库）使用独立的 venv，含 Exudyn 1.10.0 与 SciPy。

## 2. （可选）数据生成演示

数据生成的现场演示在**另一个仓库**执行，见
`../oled-microstage-simulation/ACCEPTANCE.md` 第 2 节。

> 该演示只执行 `generate-dataset` 命令并展示其终端输出（模型类型、输入/输出维数、
> 进度、最终数据字段与样本信息）。**不要在屏幕共享中打开 `configs/model.yaml`
> 或 `dataset_manifest.json`。**

## 3. 检查冻结数据集（STEP 1）

```bash
.venv/bin/python scripts/inspect_dataset.py \
    --config configs/acceptance.yaml --run-dir "$RUN"
```

期望输出：

```text
Frozen acceptance dataset
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

## 4. 确认 train / val / test 划分（STEP 2）

划分由数据集 manifest 定义（按全局索引连续切分，互不重叠），上一步已显示
`5000 / 500 / 200`。第 6、7、8 步会分别打印实际加载的样本数，三者必须一致。

## 5. 验收 run 目录（STEP 5）

第 1 步已创建 `$RUN`。本次验收的全部产物都写入该目录，不再落在
`experiments/`、`outputs/`、`logs/`、`checkpoints/`。

## 6. 拟合 train-only 谱基（STEP 3）

```bash
time .venv/bin/python scripts/fit_basis.py \
    --config configs/acceptance.yaml --run-dir "$RUN"
```

期望输出（约 11 秒）：

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
  Wrote               : runs/acceptance_.../spectral_basis.pt
```

> 展示要点：这一步**只读取 train split**，产出的是白化统计量（`sd` / `Vd` /
> `Sd` / `sy`），**不是**对目标做闭式回归——模型参数 `W` 由第 7 步的梯度训练得到。

## 7. 从零初始化训练 + 验证（STEP 4）

```bash
time .venv/bin/python scripts/train.py \
    --config configs/acceptance.yaml --run-dir "$RUN" --device cpu
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

> 展示要点：从**随机初始化**开始、**SGD**、**100 个 epoch**、每 5 个 epoch 验证、
> 按验证集选最优 checkpoint。训练日志可见 loss 单调下降约 6 个数量级。

产物：`config.yaml`、`history.json`、`metrics.csv`、`train.log`、
`best_model.pt`、`last_model.pt`。

## 8. 全测试集评估（STEP 5）

```bash
.venv/bin/python scripts/evaluate.py --run-dir "$RUN" --device cpu
```

期望输出：

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
  ACCEPTANCE LIMIT    : 1.000000e-04
  RESULT              : PASS
```

> 需要主动说明的口径差异：官方的 `relative_l2` 由 `src/training/metrics.py`
> 计算，累加前会提升到 float32，因此略高于真值；float64 直接复算给出真实误差，
> 两者都远低于 1e-4，且 `pass` 要求两者同时达标。

产物：`test_metrics.json`、`predictions.npz`。

## 9. 生成图表（STEP 6/7）

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

> 三个示例样本固定为测试集的**第一个、中间、最后一个**（索引 0 / 100 / 199），
> 按位置选取，**不是**挑最容易的样本。

## 10. 汇总指标

```bash
cat "$RUN/test_metrics.json"
```

关键字段：`relative_l2`（官方）、`relative_l2_float64`（真值）、`threshold`、
`pass`。

## 11. 产物位置

```text
$RUN/
├── environment.json / environment.txt      STEP 0
├── dataset_summary.json                    STEP 1
├── config.yaml                             STEP 4（解析后的最终配置）
├── spectral_basis.pt                       STEP 3（train-only）
├── best_model.pt / last_model.pt           STEP 4
├── history.json / metrics.csv / train.log  STEP 4
├── test_metrics.json / predictions.npz     STEP 5
└── figures/*.png                           STEP 6/7
```

## 12. 测试

```bash
.venv/bin/python -m unittest discover -s tests    # 47 项，含端到端验收冒烟测试
```

---

## 附：一页速查

```bash
cd /nishome/charliewang/forge-projects/oled-neural-operator
PY=.venv/bin/python
RUN=runs/acceptance_$(date +%Y%m%d-%H%M%S); mkdir -p "$RUN"

$PY scripts/show_environment.py --config configs/acceptance.yaml --run-dir "$RUN"
$PY scripts/inspect_dataset.py  --config configs/acceptance.yaml --run-dir "$RUN"
$PY scripts/fit_basis.py        --config configs/acceptance.yaml --run-dir "$RUN"
$PY scripts/train.py            --config configs/acceptance.yaml --run-dir "$RUN" --device cpu
$PY scripts/evaluate.py         --run-dir "$RUN" --device cpu
$PY scripts/report.py           --run-dir "$RUN"
```
