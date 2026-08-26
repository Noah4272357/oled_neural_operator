# M3 正式代理训练与官方评估记录

> 日期：2026-08-24 ｜ 战役：2026-08-24-surrogate-vs-solver（神经算子代理 vs 数值求解器）任务 6（M3 正式代理）
> 决策依据：`p3_decision.md`（形态 A 变体：13ch 线性 FNO width 64 modes 128；判定门 width→128 / modes→252；保底 F5 解析注入）
> 硬性门：`test.relative_l2 ≤ 1e-4`（官方 `scripts/evaluate.py` 为准；未达标 → 判定链 → 保底）

## 1. 正式数据集扩产（生成侧）

- **命令**：`/tmp/m3_generate_driver.py`（T4 驱动模式：直接构造 `DatasetConfig(preroll_s=0.5, integration_substeps=5, ...)`，子进程 4 worker 并行 + `merge_worker_manifests`；cwd=生成侧仓库根，`.venv/bin/python`）——`from_mapping` 不解析 preroll/substeps（T4.1 缺口），必须 Python 驱动
- **配方**：`/tmp/dataset_m3.yaml`——同 P2（preroll 0.5s、integration_substeps 5、seed 20260810、其余字段同生成侧默认），splits `{train: 5000, val: 500, test: 200}`
- **输出**：`~/data/neural_operator_3`（探针档 `neural_operator_3_pre` 未动）
- **耗时**：**~5.7 min**（recipe 16:01:43 → manifest 完成 16:07:38；5700 样本 @ 4 workers，T4 实测外推 6-7 min ✓）
- **验证**：
  - manifest `hdf5_schema_version: 5.0`、`complete: true`、`simulation.preroll_s: 0.5`、`simulation.integration_substeps: 5` ✓
  - splits 计数 5000/500/200，全样本 `status: success` ✓
  - 每样本 501 点（time 501；actuators/force (501,4)、states/q·q_dot·q_ddot (501,3)、sensors/encoder_displacement (501,4)、disturbance/force (501,2)）✓
  - 13ch 输入 = force(4)+q(3)+qdot(3)+qddot(3) → disturbance(2)，与 E 系列问题定义一致 ✓

## 2. 训练配置

`configs/m3_linear_13ch_w64_m128.yaml`（= `configs/f1_linear_13ch.yaml` 改字段）：

| 字段 | 值 | 说明 |
|---|---|---|
| model.width | 64 | 128→64（p3_decision 决策 A 变体） |
| model.modes | 128 | 不变 |
| model.activation | none | 线性不变 |
| data.root | ~/data/neural_operator_3 | 正式档显式路径 |
| data.input_fields | force displacement velocity acceleration | 13ch 不变 |
| optimizer.lr | 2.83e-3 | 不变（与 P1 可比） |
| training.epochs | 500 | 不变 |
| scheduler | cosine eta_min 3e-6 | 不变 |
| batch_size | 64 | 不变 |

`configs/m3_linear_13ch_w64_m128_chain1.yaml`：链式微调配方（batch 8、lr 5e-5、epochs 1200，其余同 M3）——E31 先例（b8 lr5e-5×1200ep 慢爬）。

## 3. 训练日志要点

- 500ep 从零（job 30906，SLURM `slurm_scripts/train_m3_w64_m128.sh`，GPU 4090，CUDA=base env `~/envs/base/bin/python`）：
  - run `experiments/fno1d_w64_b1_seed20260810_20260824-161144/`，~1.6 s/epoch，总 816 s（13.6 min）
  - 末段 train rl2 ~7.8e-3；best val 6.20e-3（epoch 500）
- 链式跳 1（job 30907，`slurm_scripts/train_m3_chain1.sh`，从 500ep best 续训）：
  - run `experiments/fno1d_w64_b1_seed20260810_20260824-162825/`，~2.8 s/epoch，总 ~56 min
  - 前 ~600ep 预热（train 7.8→7.2e-3），cosine 尾段爆发（lr 5e-6 附近 train → 4.1e-3），best val 2.67e-3（epoch 1580）
- 链式跳 2（job 30914，`slurm_scripts/train_m3_chain2.sh`，从跳 1 best 续训）：run `experiments/fno1d_w64_b1_seed20260810_20260824-172753/`，~2.8 s/epoch，总 ~57 min；best val 2.29e-3（epoch 2740）

## 4. 官方评估数值（scripts/evaluate.py，全 test 200 样本，CUDA）

| 层 | 模型 | test.relative_l2 | 与 1e-4 之比 |
|---|---|---|---|
| 层 1（500ep 从零） | fno1d_w64_b1_...-161144/best_model.pt | **6.14e-3** | 61× |
| 层 2 跳 1（链式微调） | fno1d_w64_b1_...-162825/best_model.pt | **2.61e-3** | 26× |
| 层 2 跳 2 | fno1d_w64_b1_...-172753/best_model.pt | **2.30e-3** | 23× |
| 保底（F5 解析注入，正式档） | outputs/ls_init_f1/best_model.pt | **3.01e-6** | 0.03×（余量 ~33×） |

## 5. 判定链（每层结果）

1. **500ep 从零**：6.14e-3 > 1e-4 → 未过，进入链式微调。与 P1（旧数据 5.97e-3）持平 → P2 子步净化对学习型 SGD 无提升（p3_decision §4 不确定性消解：**学习极限主导，非数据问题**）。
2. **链式微调跳 1**（b8 lr5e-5 × 1200ep，从 best 续训，E31 配方）：6.14e-3 → **2.61e-3（-57%）**，仍 > 1e-4 → 续跳。线性模型对慢爬的响应（-57%）远好于非线性 E 系列单跳（-8~-26%）。
3. **链式微调跳 2**（同配方续跳，E28 链式动力学匹配原则）：2.61e-3 → **2.30e-3（-12%）**。收益骤降（-57% → -12%），符合 E 系列"低误差区间收益递减"规律；外推该配方到 1e-4 需 20+ 跳（~20 h GPU），**学习路线实际极限 ≈ 2.3e-3，判定不可达** → 进入保底。
4. **保底（F5 解析注入）**：`outputs/ls_init_f1/best_model.pt`（13ch 线性 modes=252 width=128，零训练）官方评估于正式档 test 200 样本 = **3.01e-6 ≤ 1e-4 → 硬性门通过（余量 ~33×）**。"简易学习型"目标未达，如实披露（学习路线极限 2.30e-3，见 §7）。

## 6. 频谱 / width 诊断结论（p3_decision §4 判定门消解）

- **modes 截断（不确定性 #2 直接量化）**：正式档 test 200 样本，rfft（503 填充）bins 128-251 能量占比 6.52e-4 → 朴素截断地板 ~2.55e-2。**但** FNO1d 残差路径含 1×1 时域卷积（`FNOBlock1d.w`），输出**非带限**——实测 M3 输出 bins 128-251 能量占比 ~6.4e-4（可自由表达高频），截断地板不约束模型。**modes=128 判定门维持**：三组证据（F1 m128 6.03e-3、P1 m252 5.97e-3、M3 m128 6.14e-3）全 ~6e-3 → 截断非瓶颈，升 252 无收益（P1 已证），不升。
- **width 判定门**：F1（w128 m128 旧数据）6.03e-3 vs M3（w64 m128 新数据）6.14e-3 → 宽度非瓶颈；train（7.8e-3）> val（6.2e-3），gap 为负 → 无过拟合；欠拟合亦不成立（w128 无改善）。**保持 64，不升 128**。
- **瓶颈归因**：modes 升/不升、width 降/不降、数据净化有/无三组独立证据全落在 ~6e-3 → 平台 = **SGD 优化/泛化侧学习极限**（与 p3_decision §1 P1 结论一致）；链式微调是唯一已知突破路径（线性模型首跳 -57% 印证）。
- **正式档内蕴复核**（p2_analyze_fourier.py 尺寸自适应版跑 `neural_operator_3`，输出 `/tmp/m3_fourier/fourier_analysis.json`）：线性度地板 ~3-6e-15（算子精确线性 ✓）、13ch LS 地板 4.83e-6（对 1e-4 余量 ~20×）、diff8ch 地板 8.53e-5（余量仅 1.2×，13ch 决策正确性再确认）、u 泄漏 1.78e-5（较探针档 1.35e-5 略升，样本子集差异）。

## 7. 最终判定

**PASS（经保底回退）**：硬性门 `test.relative_l2 ≤ 1e-4` 由 F5 解析注入模型在正式档 test 200 样本上达成（**3.01e-6**，余量 ~33×）。

判定链路径：500ep 从零（6.14e-3）→ 链式微调跳 1（2.61e-3）→ 跳 2（2.30e-3，收益 -57%→-12% 递减，判定学习路线实际极限 ≈2.3e-3、不可达 1e-4）→ **保底 F5 注入（3.01e-6 PASS）**。

**"简易学习型"目标未达，如实披露**：13ch 线性小型 FNO（w64 m128）经 500ep + 2 跳链式微调共 ~2.1 h GPU，最低 2.30e-3（23× 超标）；学习极限瓶颈为 SGD 优化侧（§6 三组独立证据），非容量/截断/数据问题。达标形态为解析注入（417 万参数非"简易"），作为战役结论：**硬门由注入保底达成，学习路线的工程可行性需另行立项（如全局 float64、更大步数预算、或逐 bin LS 初始化学习）**。

## 8. 复现命令

```bash
# 扩产（生成侧 cwd，.venv/bin/python）
python /tmp/m3_generate_driver.py --workers 4   # recipe /tmp/dataset_m3.yaml → ~/data/neural_operator_3

# 500ep 从零
sbatch slurm_scripts/train_m3_w64_m128.sh
# 链式跳 N
sbatch slurm_scripts/train_m3_chainN.sh

# 官方评估
scripts/evaluate.sh --checkpoint experiments/<run>/best_model.pt

# 频谱/截断复核（/tmp/m3_bin128_check.py，不入库）
python /tmp/m3_bin128_check.py ~/data/neural_operator_3 200
```
