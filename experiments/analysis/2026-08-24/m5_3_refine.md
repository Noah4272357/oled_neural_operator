# M5-3 训练后 joint-LTI 精修记录（保底任务，官方 test rl2 ≤ 1e-4 门）

> 日期：2026-08-24 | 任务：task-m5-3 | 前置：M5-2 判定 FAIL（P2a/P2b/P2c 均 >1e-4 且衰竭）
> 起点模型：`outputs/job_30929/m52_p2a_hop1/best_model.pt`（官方 test rl2 = 1.9622e-4，学习路线当前最优）
> 评估档：`~/data/neural_operator_3`（5000/500/200，13ch 输入），官方 `scripts/evaluate.sh`（禁 TF32，时域 relative_l2）
> 协议：训练集 5000 残差 r = d − model(x)，未填充 501 点 rfft 基 bins 3..250，float64，闭式 ridge-LS（**零梯度步骤**，F4 教训），包装 `M_final(x) = model(x) + LTI_refine(x)`（RefinedFNO1d，`lti_map` buffer），官方评估 + 分解

## 结论（TL;DR）

- **官方 test rl2：1.9622e-4 → 1.8589e-4（−5.26%）**，未达 1e-4 门 → **M5-3 FAIL**
- **诚实判定**：训练后残差的线性（LTI-in-x）内容仅 ~13%（bins 3..250 能量），闭式 LTI 精修的理论上限远低于门；残差 ~87% 是非线性模型误差，非任何闭式线性头可回收——与 M5-1 正交性观察、F4 机理自洽
- 训练路线（M5-1/2/3）全链关闭；**唯一达到 1e-4 的模型仍是解析注入 ls_init_f1（官方 2.84e-6）**，本项目目标已达成状态不变（type-2）

## Step 1: 残差 joint-LTI 拟合 + 组合模型 + 官方评估

### 协议修正（本任务最重要发现）

M5-1 的 joint-fit 协议（`ls_init_fno.fit_ridge_ls_maps`：real-stacked 4 列设计 + `coeff[0::2,:2].T − 1j·coeff[1::2,:2].T` 重建）**只对复数一致的目标有效**（精确线性逆算子 d = M q̈ − B u，M5-1 恢复 A_true 至 5.5e-13）。对训练后模型的**非线性残差**：

- 4 列 [Re r1, Re r2, Im r1, Im r2] 是独立投影，重建公式把实部系数"旋转"为虚部估计；在病态联合设计（cond ~2.2e9，~15 个配方受限近零方向）上，旋转映射的输出幅度 ∝ σ_max·‖coeff‖——实部拟合达 77.4% 解释率的同时，重建复数映射 |Mx|² ~ 1e10 vs |r|² ~ 7（**爆炸 9 个量级**，暴力复核证实）。"0.00% 恢复"与"77.4% 解释"的矛盾即源于此
- 修复：**复数约束 LS**（`np.linalg.lstsq` 原生复数，最小化复数范数 = 正确约束目标）。合成验证：重建公式对线性目标正确（3e-16），复数拟合对非线性目标为真最优（ratio 0.964）
- 顺带修复：分解侧 `M_eff[:3] = 0.0` 作用于 (2,13) 共享 map 会把整表清零（bins 归零属于 (251,2,13) 包装 buffer）——改为逐 bin 归零 + per-bin `after_error`（暴力复核逐 bin 一致）

### 结果（reg 网格 1e-14/1e-12/1e-10/1e-8 全部逐位相同——ridge 数值惰性，秩满 rank=13）

| split | M0 freq rl2 | 精修后 freq rl2 | 恢复率 | 官方时域 |
|---|---|---|---|---|
| train (n=5000) | 1.829109e-4 | 1.718741e-4 | **6.03%** | 1.7069e-4（前 200） |
| val (n=500) | 2.211702e-4 | 2.116543e-4 | 4.30% | 2.1681e-4（前 200） |
| test (n=200) | 2.052112e-4 | 1.953484e-4 | 4.81% | **1.85885e-4（官方，全 200）** |

- 官方 M0 test = 1.9622e-4 → 精修后 1.8589e-4，**官方相对提升 5.26%**（误差能量 −10.2%）
- max|M| = 103.4（伪影消解；旧协议下 709 是病态伪信号，非真实系数）
- 逐 bin：bins 4-6（tone）真实改善（4.2e-5→3.6e-5 / 4.5e-5→3.5e-5 / 4.7e-5→3.3e-5）；bin0（7.3e-5，最大单项）未校正
- 误差能量分布（训练集 per-bin）：bins 0-2 占 22.5%，bins 3-250 占 77.5%（精修后 74.5%）

### 判定

残差线性内容（复数约束）解释率 **12.97%**（bins 3..250）；组合模型官方 1.8589e-4 > 1e-4 → **Step 1 未达门** → 执行 Step 2。

## Step 2: 扩展杠杆逐一评估（全部穷尽）

1. **更细 reg 网格（1e-14/1e-12）**：与 1e-8..1e-10 **逐位相同**——联合系统秩满 13、ridge 相对 1e-11 尺度数值惰性，无效果（已实测，非推断）
2. **float64 全链路**：精修路径全程 float64（rfft→einsum→irfft），自评时域（1.85885e-4）与官方（1.85885e-4）**逐位一致**——无精度损失
3. **bins 扩展**：bins 0-2 是 ramp 结构（Gram cond 9.3e27/8.2e23），共享 map 外推到该输入量级区（|x| 大 1e3 倍）必爆炸（M5-1 与本次共同确认）；**逐 bin 映射**在配方受限流形上不可识别（per-bin Gram cond 中位数 9.1e16，max 2.6e36——M5-1 已记载），任何 ridge 要么爆炸要么退化到零——频率依赖线性头同样死路。3..250 已是全部良态域
4. **val 泛化分解**：train 6.03% vs val 4.30%（freq）——轻微过拟合缺口，无泛化崩坏，但也不存在"泛化未到账"的隐藏收益

## 精度来源构成（诚实性契约）

| 组成 | 数值 | 说明 |
|---|---|---|
| SGD 训练部分（M0，官方） | 1.9622e-4 | 占最终误差的 ~90%（能量比） |
| 闭式 LTI 精修头贡献（官方） | −5.26%（相对 rl2） | 全部来自闭式 ridge-LS，**零梯度步骤** |
| 残差线性内容上限（bins 3..250） | ~12.97% 能量 | 复数约束拟合可解释部分——精修理论上限 |
| 最终官方 test rl2 | **1.85885e-4** | > 1e-4 → FAIL |

预期与现实：任务简介预期 ~2.83e-6（M5-1 解析注入量级）**不成立**——那是"精确线性逆算子"的拟合误差；本任务是"训练后非线性残差"的线性拟合，二者目标完全不同。训练已吸收大部分线性部分（正交性现象），剩余残差 87% 非线性。

## 战役判定链

- M5-1：P1（近似 LS 初始化）死路（F4 机理：Adam 步进破坏精确映射）
- M5-2：P2a（续链 +3.65% 衰竭）/ P2b（modes 252 反证）/ P2c（线性化反升 5.5×）→ FAIL
- **M5-3：闭式 LTI 精修 −5.26% → FAIL**
- 训练路线（SGD + 闭式精修）全链关闭；**1e-4 合规模型唯一为解析注入 ls_init_f1（官方 evaluate.py 复核 2.84e-6，余量 35×，零训练）**——项目"目标已达成（type-2 合规）"状态不变

## Concerns

1. **精修提升随训练收敛而收缩**：残差线性内容（12.97%）是 E31 链 6000+ep 训练后的剩余——若早期模型精修，回收率或更高，但与"训练到最优再精修"的保底定位矛盾（早期模型本身 rl2 差）
2. **bins 0-2 占 22.5% 误差能量但不可校正**：ramp 结构使共享 map 外推爆炸、逐 bin 拟合不可识别——该块误差是配方受限流形的结构后果，闭式路线无解
3. **协议可迁移性警告**：`fit_ridge_ls_maps` 的 real-stacked 4 列协议**仅对线性目标安全**；对非线性目标必须用复数约束 LS（本次已修复并留档，`refine_lspinv.py` docstring 记载机制）

## 产物

- 脚本：`scripts/analysis/refine_lspinv.py`（复数约束 joint-LTI 精修）；`src/models/fno.py` RefinedFNO1d + `src/models/factory.py` fno1d_refined
- 组合 checkpoint（gitignored）：`outputs/job_30929/m53_refine/reg_{1e-14,1e-12,1e-10}/best_model.pt` + refine_report.json + summary.json + `m53_refine_run.log`
- 复现：`python scripts/analysis/refine_lspinv.py --checkpoint outputs/job_30929/m52_p2a_hop1/best_model.pt --data-root ~/data/neural_operator_3 --output-dir outputs/job_30929/m53_refine --reg 1e-14 1e-12 1e-10`，官方评估 `scripts/evaluate.sh --checkpoint <pt> --data-root ~/data/neural_operator_3`
