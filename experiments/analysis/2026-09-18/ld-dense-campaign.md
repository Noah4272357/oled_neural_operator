# 8ch 代理模型：float32 量化缺陷修复 + 纯训练达标（2026-09-18）

> 状态：**收官**。用户裁决口径「代理模型必须是训练得到的，不是直接的解析解」已满足：
> 交付物 `experiments/ld_dense_final/best_model.pt` 由 `scripts/train.py` 从随机初始化 SGD 训练产生，
> 官方路径 `scripts/evaluate.py` 出数 **test relative_l2 = 5.940e-08 ≤ 1e-4**（余量 1683×）。
> 唯一可训练参数 = `model.W (256×1004)`；无闭式 head、无解析注入、无 pinv/lstsq。
>
> ⚠️ **该 5.940e-08 是指标自身的 float32 地板，不是模型的真实误差。** `src/training/metrics.py:29-30`
> 做 `prediction.detach().float() - target.float()`，指标在 float32 下计算。同一份预测：
> float64 计算得 **4.672084e-08**（真实误差），float32 计算得 **5.940277e-08**（= 官方报告值本身；
> 口径差异见 §9，**不是**"重算逐位复现"）。
> 真实值约 **4.67e-08**。这不威胁门槛（余量仍 1683×），但**这批数字的有效位数到此为止**。

## 1. 任务与验收口径

- **问题**：8ch 逆扰动算子 `Gθ : {u(t), s(t)} → f̂_dist(t)`，4 路音圈电机力 + 4 路光栅位移 → 2 维扰动力，501 点、dt=1ms。
- **判据**（`季华实验室项目档案/archive/02` §3.2.2）：`E_rel = √(ΣᵢΣₖ‖f̂ᵢ(tₖ)−fᵢ(tₖ)‖₂²) / √(ΣᵢΣₖ‖fᵢ(tₖ)‖₂²) < 10⁻⁴`，与 `RegressionMetrics.relative_l2` 符号同一。
- **用户追加口径**：(a) 模型必须**训练**得到，不是解析解；(b) 线性/非线性**两条都跑，按 val 选优**，test 只在最后触碰一次；(c) 按需允许 float64。
- **数据**：`~/data/neural_operator_3`（manifest 3.0），train 5000 / val 500 / test 200。**全程只读**（收尾以 `find -newermt` 验证零改动）。

## 2. float32 量化缺陷（本战役的起点）

### 2.1 机制

`src/data/dataset.py::_read_time_fields` 把 HDF5 的 float64 **先降为样本 dtype（float32）再交给 `transform`**。
`diff_features` 的第二差分除以 `dt² = 1e-6`，把输入侧的 float32 舍入误差**放大 10⁶ 倍**。
`second_derivative` 同理会放大。这是一个**顺序错误**，不是精度不足。

### 2.2 双向验证（同一模型、同一 split，只改 cast 顺序）

| split | 缺陷在场（默认路径） | 修复后 | 倍数 |
|---|---|---|---|
| train | 5.658941e-05 | 1.954627e-05 | 2.90× |
| **val** | **1.638786e-04（不过门）** | **1.977140e-05（过门）** | **8.29×** |
| test | 9.480696e-05 | 2.043966e-05 | 4.64× |

缺陷值在本次复现中**逐位一致**，修复值同样逐位一致 → 缺陷可复现、修复可验证。
中间档 `dtype=float32, transform_dtype=float64` 得 test 2.535117e-05（受 transform 输出仍存 float32 限制）。

### 2.3 改法（opt-in，默认行为逐字节不变）

- `OLEDNeuralOperatorDataset` / `create_datasets` / `create_dataloaders`：新增 `transform_dtype`。
- `_read_time_fields` 按 `transform_dtype or self.dtype` 读取；`__getitem__` 在 `transform` **之后**把返回 dict 中每个 tensor 降回 `self.dtype`（含 `grid`）。
- `src/data/dataloader.py`：转发 `transform_dtype`，并新增 `data.dtype` 出口（缺省回退 float32，**不得回退 None**——`_resolve_dtype(None)` 返回 None 会覆盖 `create_datasets` 自身的默认值）。
- 回归测试 `tests/test_dataset_precision.py`（4 例，hermetic：stub `h5py`，不依赖真实 HDF5）。

## 3. 模型类选择：per-bin 可达标，但被 dense 跨频超出 473×

> ⚠️ **本节已于 2026-09-18 更正。** 原表混用了不同输入表示下的测量值，彼此不可比：
> 「per-bin 5.633e-04」来自 8ch `second_derivative` + float32 `fno1d`；
> 「dense 1.5e-06」来自 float32 污染特征 + d=1280（比训练实际达到的值还差 32×）。
> 更严重的是，原表据此得出的**「per-bin 整体出局（高于门槛）」结论被更正后的数据推翻**。
> 下表是在**同一条件**下重新测量，取代原表全部数字与结论。

**统一条件**：16ch `diff_features`、float64 特征与 float64 变换、train 5000 / test 200、时间域 `E_rel`。

| 模型类 | total | 内部点 | 边缘 8 点 |
|---|---|---|---|
| 共享（频率无关，单矩阵） | 1.0260e-02 | 2.3290e-03 | 9.9919e-03 |
| per-bin（输入仅 101 bin） | 1.4460e-03 | 4.1892e-04 | 1.3840e-03 |
| per-bin（输入全 251 bin） | 2.4727e-05 | 2.4032e-05 | 5.8241e-06 |
| **dense 跨频闭式（d=256，类上限）** | **5.2295e-08** | 5.0654e-08 | 1.2996e-08 |
| **dense 跨频训练模型（交付）** | **4.6721e-08** | 4.6347e-08 | 5.8978e-09 |
| *旧复合体（对照，管线不同）* | *2.043966e-05* | — | — |

**更正后的判读**：

1. **per-bin 类并未出局。** 给足全谱输入时它达到 **2.4727e-05，低于 1e-4 门槛 4×**，本身是可达标的。
   只有当输入被限制在低频 101 bin 时才塌到 1.4460e-03（见第 3 点）。
2. **旧复合体的 2.043966e-05 与 per-bin 上限 2.4727e-05 吻合到 17%**——旧管线实际是**坐在 per-bin 类的墙上**，
   残差分支仅轻微越界。这比原表「per-bin 5.633e-04 而复合体 2.04e-05」的叙事更准确。
3. **dense 跨频仍比 per-bin 好 473×**（2.4727e-05 → 5.2295e-08）。交付采用 `SpectralDenseMap` 的理由
   不是「per-bin 不达标」，而是「dense 高出两个数量级」。

**dense 这额外的 473× 从哪来（本节的关键发现）**：

| 目标分解（test 200） | 占比 |
|---|---|
| `‖y_low‖/‖y‖`（bin 0..100） | 9.999990e-01 |
| **`‖y_high‖/‖y‖`（bin 101..250）** | **1.445812e-03** |

**高频段只占目标能量 0.0002%，却占目标振幅的 1.4458e-03——比 1e-4 门槛高 14.5×。**
故任何只能产生低频输出的模型，误差地板就是这 1.4458e-03。实测印证：
per-bin 把高位输入置零后得 **1.4460e-03**，与 `‖y_high‖/‖y‖ = 1.445812e-03` **吻合到三位有效数字**。

> **旁证（独立测量）**：2026-08-26 战役的 `diag_tikhonov_structure.py` 在**不同设置**下
> （8ch `second_derivative`、f64、残差口径）测出 per-bin 独立 = **1.526e-3**，与本节的 1.4460e-03
> 相差 5%。两次测量用的表示、通道数、口径都不同却落在同一量级，说明该地板是**目标自身的性质**，
> 不是某一种特征表示的产物。（该旧数字本身是其战役的真实记录，不因本节更正而失效。）



要过 1e-4，必须把这段高频内容还原到约 7% 的相对精度。两条路线：

- **per-bin**：靠输入自身的高频 bin。消融验证——把 train 的高频 bin **跨样本打乱**后重新拟合，
  test 从 2.4727e-05 退化到 **1.4372e-03**，说明该信息**样本特异且真实**，不是拟合噪声。
  （高频 bin 仅占输入总能量 **0.000024%**，是 per-bin 映射把它放大后用于输出的。）
- **dense 跨频**：从**低频输入合成**高频输出。这是 per-bin 结构原理上做不到的，
  也正是 473× 差距的来源。

**根因**：编码器→`q̈` 的反演**不是循环算子**——有限差分是 Toeplitz 而非 circulant，
`diff_features` 的边缘填充 `out[0]=out[2]` 更明确非平稳。DFT 只对角化 circulant 算子，
故非 circulant 的线性算子**必然耦合频率**，而 per-bin 映射在数学上就是一个 circulant 算子。

## 4. 数值秩发现与 d 的选择（本战役第二个非平凡发现）

改用 float64 特征后，特征矩阵 `(5000×3232)` 的奇异谱**衰减到 1.9e-14 相对量级**——
即该矩阵**数值上近乎秩亏**（float64 噪声地板 ≈ `eps·S[0]` = 1.64e-12）。
而旧 basis 是在 float32 特征上算的，被 float32 量化地板（1.5e-07）截住，掩盖了这一点。

用 1/Sd 白化噪声方向 → `AᵀA = I` 数值上不再成立 → **同一 lr 直接发散**（首次训练 run 从第 2 epoch 起爆到 inf）。

`d` 扫描（float64 特征，闭式最小二乘上限）：

| d | S[d−1]/S₀ | ‖AᵀA−I‖ | train | val | **test** |
|---|---|---|---|---|---|
| 128 | 3.27e-10 | 4.68e-07 | 1.71e-08 | 1.80e-08 | 1.88e-08 |
| **256** | **1.36e-10** | **1.63e-06** | 4.83e-08 | 5.13e-08 | **5.23e-08** |
| 384 | 5.55e-11 | 1.63e-06 | 1.33e-07 | 1.47e-07 | 1.58e-07 |
| 512 | 2.55e-11 | 1.20e-05 | 2.75e-07 | 3.25e-07 | 3.43e-07 |
| 768 | 3.59e-13 | 1.15e-04 | 1.07e-05 | 1.32e-05 | 1.38e-05 |
| 1280 | 1.88e-14 | 4.02e-01 | 4.10e-04 | 7.41e-04 | 6.58e-04 |

**取 d = 256**：`S[d−1]` 高于 float64 噪声地板 83×，`‖AᵀA−I‖ = 1.6e-06`（Hessian 各向同性到 1.6e-6，
下述 lr 因而有效），上限 5.23e-08 —— 距门槛 1912×。

## 5. 训练方案

- **Hessian 结构**：白化后 `AᵀA = I`，损失 `(1/n)‖A W Rᵀ − Y‖²` 的 Hessian 为 `(2/n)(RᵀR ⊗ AᵀA)`，
  `R` 为 irfft。实测 `κ(RᵀR) = 2.000000`（DC/Nyquist 权重 1，其余 2）。
  因 `AᵀA ∝ m` 而 `n ∝ m`，**Hessian 与批大小无关** → lr 不随 batch 缩放（初版按 batch 缩放是错的）。
- **lr**：一步损失在 lr 上是严格抛物线，三次求值给出精确 Newton 步。
  **注意**：δ 必须够大，否则曲率项被损失的分辨率淹没（δ=1 时 `L(2δ)−2L(δ)+L(0)` 打印精度下为 0，曲率纯噪声）。
  取 δ=1e5 后得 `a = 3.9496e-11`、`b = −2.1202e-05` → **lr\* = 2.6841e5**；1.5× 即发散。
- **损失口径**：用 `mse` 而非 `relative_l2`。二者对固定 split 只差常数 `‖y‖`，**选优序不变**；
  但 `relative_l2` 的梯度带 `1/(‖e‖‖y‖)` 因子，使 `|W*|`（max 449、rms 4.42）需要 ~4.5e4 步。
  官方判据仍由 `scripts/evaluate.py` 独立计算 `relative_l2`，不受训练损失口径影响。
- **配置**（`configs/ld_spectral_dense.yaml`）：`dtype/transform_dtype = float64`、`preprocessing = diff_features(16ch)`、
  full batch 5000、SGD `lr=2.684e5` momentum 0 wd 0、cosine → 0、100 epoch、CPU ≈ 59 s。
- **选择协议**：`best_model.pt` 按 `best_validation_*` 选优；test 在定稿后由 `scripts/evaluate.py` 触碰一次。

## 6. 结果

### 路线 L — 线性（交付物）

```
experiments/ld_dense_final/official_test.json
  split: test, samples: 200
  relative_l2 = 5.940277290056465e-08      （判据 < 1e-4，余量 1683×）
              = float32 指标地板；同一预测 float64 计算为 4.672084e-08（见 §9）
  mse         = 1.1659823694267896e-14
  rmse        = 1.0798066352022429e-07
best_validation_relative_l2 = 6.074205768110958e-08    （100 epoch）
```

训练全程 train/val/test 三者同步下降、无过拟合间隙（例如第 60 步 train 4.67e-09 / val 4.65e-09 / test 4.52e-09）。

### checkpoint 成分（无解析分量）

```
model.W    (256, 1004)  float64  ← 唯一可训练参数（SGD 产出）
model.sd   (3232,)      float64  ┐
model.Vd   (256, 3232)  float64  ├ 固定预处理统计量，仅由 train split 计算
model.Sd   (256,)       float64  │
model.sy   ()           float64  ┘
state_dict 中 lti/analytic/pinv/solve/head/refine 标记：NONE
```

`sd`/`Vd`/`Sd` 为**特征**侧统计量（列标准差 + SVD 基），对目标**未做任何最小二乘或伪逆**。
`sy` 是 train 目标谱的**全局标准差**（单标量 30.1536，单位归一化常数，非回归解）；
此处如实披露：它是唯一取自目标的量，且只取自 train。

### 路线 N — 非线性（用户要求的对照）

`configs/lnn_fno16ch_gelu.yaml`：`fno1d` + `gelu`（`embed_dim 64 / modes 252 / width 128 / lift_dim 64 / num_blocks 1`，
**4,171,586 参数**，float32/complex64），同 16ch `diff_features` 输入、同 `transform_dtype: float64`（缺陷已修），
损失 `relative_l2`、AdamW lr 1e-3 wd 1e-4、cosine → 3e-6、**300 epoch 完整跑完**（≈72.5 min CPU）。

```
experiments/lnn_fno16ch_gelu_final/official_test.json
  split: test, samples: 200
  relative_l2 = 2.8964909609337864e-03     ← 判据 < 1e-4，差 29×  → 不达标
best_validation_relative_l2 = 2.7463050727497976e-03
```

训练轨迹（未收敛，仍在缓降）：

| epoch | 1 | 10 | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|---|---|
| train | 1.0003 | 0.99839 | 4.97e-02 | 1.64e-02 | 1.27e-02 | 5.27e-03 | 2.83e-03 | **2.38e-03** |
| val | — | 0.99731 | 5.74e-02 | 4.02e-02 | 1.12e-02 | 1.05e-02 | 3.17e-03 | **2.75e-03** |

**判读**：这是一次**公平的**完整运行，不是被中断的运行（对比 `m62`/`m63` 停在 ep 4/9 无 `history.json`）。
它把「FNO 在 8ch 学不动」从**未检验的说法**变成了**受控证据**：300 epoch 后 train 仍有 2.38e-3，
不存在「多训就能过门」的迹象（cosine 已把 lr 压到 3e-6，后 50 epoch 仅降 16%）。
与项目既有 FNO 学习极限 1.86e-4（M5，13ch）同向；本run 因 16ch 输入 + 门控更严，停在同一量级偏上。

## 6b. val 选优（用户裁决「按 val 选优」）

| 路线 | best val relative_l2 | 官方 test relative_l2 | 判定 |
|---|---|---|---|
| **L（dense 跨频，交付）** | **6.0742e-08** | **5.9403e-08** | ✅ **过门（余量 1683×）** |
| N（FNO+gelu，对照） | 2.7463e-03 | 2.8965e-03 | ❌ 差 29× |

**选优结论：路线 L 胜出，优势 45,222×（val 口径）。** 交付模型 = `experiments/ld_dense_final/best_model.pt`。
路线 N 作为对照**按用户要求保留**，不删除、不改动，其 checkpoint 与 `official_test.json` 留在原位备查。

## 7. 失效产物声明（不得再引用）

以下产物建立在 float32 量化缺陷的特征上（和/或含闭式分量），自本战役起**作废**：

| 产物 | 失效原因 |
|---|---|
| `outputs/lti_head_16ch.pt` | 在 float32 污染特征上 `pinv` 拟合；且实测单独仅 1.19e-03 |
| `experiments/m63_residual_targets` | 由上述 head 派生的残差训练集 |
| `outputs/sgd_pca_1280/` | 复合体（闭式 head + 训练残差分支），非纯训练产物 |
| `OPTIMIZATION_REPORT.md` §7/§8/§12 相关数字 | 均建立于上述两者之上 |
| 旧 `basis_16ch_f64_d1280.pt` | 在 float32 特征上算的 basis；d=1280 在 float64 下 `‖AᵀA−I‖ = 0.40`，已废弃（改用 d=256） |

`experiments/m62_fno16ch_s20260810` / `m63_fno16ch_refined_q1` 系**被中断的运行**（无 `history.json`），
不能作为「FNO 学不动」的证据；本次路线 N 是首次完整跑完的 16ch FNO。

## 8. 复现命令

```bash
cd /nishome/charliewang/forge-projects/oled-neural-operator
PY=.venv/bin/python

$PY -m unittest discover -s tests -v                 # 24 tests OK（含 4 例精度回归）

# basis（仅用 train；与训练同表示，float64）
$PY scripts/analysis/spectral_basis.py --config configs/ld_spectral_dense.yaml \
    --d 256 --output outputs/spectral_basis/basis_16ch_f64_d256.pt

# 路线 L：训练 → 官方出数
$PY scripts/train.py --config configs/ld_spectral_dense.yaml --run-name ld_dense_final
$PY scripts/evaluate.py --checkpoint experiments/ld_dense_final/best_model.pt \
    --output experiments/ld_dense_final/official_test.json

# 指标 float32 地板：同一预测的 float64 真值 vs float32 官方值（见 §9）
PYTHONPATH=. $PY scripts/analysis/f64_metric_floor.py --run experiments/ld_dense_final
```

`f64_metric_floor.py` 的期望输出（2026-09-18 实测）：

```
float64 (true)          : 4.6720842964331878e-08
float32 (metrics.py)    : 5.9402772900564648e-08   <- 官方值，与 official_test.json 逐位相同
float32 (float64 sums)  : 5.9402770788493857e-08   <- 独立复算，与上一行一致到 7.4 位
floor / true            : 1.2714x
margin vs 1e-4 (true)   : 2140x
```

## 9. 诚实条款

- 路线 L 的 test 数字**唯一来源**是 §8 的 `official_test.json`；未使用任何解析/闭式回退。
  该文件经 `scripts/evaluate.py` **重跑复现，逐位一致**（5.9402772900564648e-08）。
- **指标精度地板（须知悉）**：官方指标在 **float32** 下计算（`src/training/metrics.py:29-30` 强制
  `prediction.detach().float() - target.float()`），故 **5.940e-08 是测量地板，不是模型误差**。
  同一份预测，三种算法给出三个值：

  | 算法 | 值 | 说明 |
  |---|---|---|
  | float64（`double()` 后求误差） | **4.6720842964e-08** | **真实误差**（余量 2140×） |
  | float32，`metrics.py` 口径 | **5.9402772901e-08** | 官方报告值 |
  | float32，独立复算（float64 累加） | **5.9402770788e-08** | 与官方值一致到 **7.4 位有效数字** |

  后两者绝对差 `2.112e-15`（相对 `3.556e-8`）——**不是**"逐位复现"——差异源于 `metrics.py` 用 **float32 张量**
  逐 batch `+=` 累加（`_squared_error_tensor`，L31-36），而独立复算把每 batch 的 float32 和
  提升为 float64 后再累加。**两者都是同一 float32 差值的不同求和顺序，都不等于真值。**
  连带后果：§4 扫描中 `d=128` 的 1.88e-08 与 `d=256` 的 5.23e-08 **都低于或贴近该地板**，
  官方口径**无法区分二者**——`d` 的选择对**官方报告值**实际不敏感，真实差异只在 float64 下可见。
  复现命令见 §8。
- **§3 已更正（自纠记录）**：本节表格原有的两行混用了不同输入表示下的测量值，
  并据此得出「per-bin 整体出局」的**错误结论**。2026-09-18 在同条件下重测后更正为：
  per-bin 可达标（2.4727e-05，低于门槛 4×），dense 超出其 473×。更正详情见 §3 提示框。
- **选择协议披露（须知悉）**：`d` 的选择发生在 §4 的扫描中，而该扫描**同时打印了 test**。
  据实说明其选择依据与后果：
  - 选择依据是**条件数**（`S[d−1]` 高于 float64 噪声地板的倍数、`‖AᵀA−I‖`），不是 test 误差。
  - 但**该依据本身也非「取最优」**——`d=128` 在**每一个已测轴上都支配 `d=256`**：
    误差更低（test 1.88e-08 vs 5.23e-08）、`‖AᵀA−I‖` 更小（4.68e-07 vs 1.63e-06）、
    噪声地板余量更大（199× vs 83×）。故 `d=256` 是**保守余量取值**，**不是**任何意义下的最优点。
  - **因此本报告的数字不是该协议下能达到的最好成绩**：`d=128` 会更好。
    取 `d=256` 使得 test 未被用于选择（若按 test 选，必取 128）——**该结果因而是保守的**。
  - 事后改选 `d=128` 会构成**在看过 test 之后的选择**，违反步骤 5 协议，故不改。
- **同时披露**：`d` 扫描脚本（`/tmp/d_sweep.py`）与 `lr` 线性搜索脚本（`/tmp/lr_linesearch.py`）
  均为一次性探针，**未入库**；其结论已由本节与 §4/§5 的表格完整承接。`lr` 的选择只用了**训练损失**的一步抛物线拟合（不经 val，不经 test）。
- **`experiments/ld_probe/` 与 `experiments/lnn_gelu_probe/` 是冒烟 run，不是选择依据**：
  分别为 5 epoch（adamw lr 0.01）与 20 epoch，`best_val` 0.9894 / 0.7430——用途仅为确认管线端到端可跑。
  它们**先于** §5 的 lr 发现，配置与最终交付 run 不同，**未参与任何选优**。保留在原地备查（gitignored，共 ~314 MB）。
- **口径未决项（结转）**：`archive/02` §3.2.2 的判据写 `Gθ : {u(t), s(t)}`，本交付的输入是 16 通道
  `[u(4), s(4), ṡ(4), s̈(4)]`（后 8 通道为 `s` 的确定性有限差分派生）。若验收方要求 `s` 必须**原样**输入，
  需事先澄清——本战役不解决。回填验收表时已按 ✅ 标注并显式挂 ⚠️ **输入口径待验收方澄清**，
  **不得**按「已无条件通过」解读。
- 数据只读：`find ~/data/neural_operator_3 -newermt 2026-09-16` 无输出。
