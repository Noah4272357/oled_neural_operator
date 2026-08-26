# P2 探针：预滚（preroll）+ 积分子步（substeps）档数据集泄漏验证

> 日期：2026-08-24 ｜ 战役：2026-08-24-surrogate-vs-solver（神经算子代理 vs 数值求解器）P2 探针（生成侧+代理侧）
> 任务 4：P2 探针数据集生成（预滚+子步）+ 泄漏验证。

## 1. 数据集生成（生成侧 `oled-microstage-simulation`）

- **Ruling 2 生效**：CLI 未透传 `preroll_s` / `integration_substeps`。且实证发现更深的坑：
  `DatasetConfig.from_mapping` 不解析 YAML 中这两个键（`configs/dataset.yaml` schema 未接线）——
  纯 YAML fallback 会**静默产出 preroll=0 / substeps=1**（冒烟测试确认 manifest 显示 `preroll_s: 0.0`）。
- **实施**：临时 YAML `/tmp/dataset_p2_pre.yaml`（由 `configs/dataset.yaml` 派生：`output_root`、`splits 500/50/50`、写上 `preroll_s: 0.5` / `integration_substeps: 5` 作 recipe 文档，实际取值由驱动直接构造 dataclass）+ Python 驱动 `/tmp/p2_generate_driver.py`
  （`DatasetConfig(**{**cfg.to_mapping(), "preroll_s": 0.5, "integration_substeps": 5})`，worker 并行协议同 CLI：父进程 spawn 子进程、`merge_worker_manifests` 合并）。
- **命令**（生成侧 `.venv/bin/python`，cwd=生成侧仓库根）：

  ```bash
  .venv/bin/python /tmp/p2_generate_driver.py --workers 4 \
    --output ~/data/neural_operator_3_pre
  ```

- **耗时**：600 样本 4 workers 约 **40 s**（h5 文件 mtime 跨度 36.7 s；远低于预估的数十分钟~2 小时）。全量集（5700 样本）外推 ~6-7 min @4 workers。

## 2. manifest 3.0 验证（全部通过）

- `schema: neural_operator_dataset_manifest/3.0`；`complete: true`；`failure_count: 0`（600/600 success）
- `splits: {train: 500, val: 50, test: 50}`
- `simulation: {preroll_s: 0.5, integration_substeps: 5, solver: exudyn:GeneralizedAlpha:GenericODE2-3DoF}`
- `time_grid: {count: 501, dt_s: 0.001, start_s: 0.0, end_s: 0.5, preroll_s: 0.5}`（每样本 501 点、1 kHz、0.5 s 窗）
- `recommended_problem: inverse_disturbance`（8→2）；`hdf5_schema_version` 不变；方程/M/B/H 不动（preroll 提交已保证）

## 3. 泄漏比验证（analyze_dataset_fourier.py 逻辑，代理侧）

- **脚本适配**：analyzer 硬编码 5000/500/200 的样本索引上限，500/50/50 下会越界崩溃 → 用适配副本
  `/tmp/analyze_p2.py`（cap 与采样步长按 manifest 实际大小推导；对原数据集大小数值等价）。仓库脚本未改。
- **数据集级结果**（analyzer 默认 test 样本 0；`--data-root ~/data/neural_operator_3_pre --output-dir experiments/analysis/2026-08-24`）：

  | 指标 | 基线（neural_operator_2, 2026-08-23） | P2（neural_operator_3_pre） |
  |---|---|---|
  | leakage s | 4.666 | 36.15 |
  | leakage sddot | 2.5e-5 | 1.99e-5 |
  | leakage u | 1.8e-5 | 1.35e-5 |
  | leakage d | 1.4e-4 | 2.3e-4 |
  | raw_8ch LS 地板 | 0.5686 | 0.5599 |
  | diff8ch LS 地板 | 8.8e-5 | 8.2e-5 |
  | 13ch LS 地板 | 5.0e-6 | 4.2e-6 |
  | sddot 解析逆 rl2 (train/test) | 0.0081 / 0.0103 | 0.0079 / 0.0094 |
  | linearity 地板 rl2 | ~4e-15 | ~4e-15 |

  > 注意：数据集级 leakage 的 test 样本新旧 seed 不同（旧 test0 = seed 20260810+5500；新 test0 = seed 20260810+550），
  > 非严格对照；**同种子对照**见下。

- **同种子对照（train 0..9，seed 20260810+全局索引，新旧共有）**：leakage s 旧 ~4.2-5.1 → 新 ~34-45
  （train0: 4.941 → 42.29）。DFT 分解：`e_tone` **完全相同**（train0: 0.0370/0.0370），`e_other` 0.183 → 1.565——
  增长全部来自 **DC bin**（预滚引入的常数偏移被计入非音调能量）。

## 4. 决定性发现：预滚不能移除斜坡（spec §48 假说证伪）

- **q_new ≡ q_old + 常数**：同种子 train0 仿射拟合 `q_new - q_old` 残差 1.6e-9（机器精度，信号量级 1e-3）；train1-9 同。
- **漂移斜率位级一致**：x 通道 0.00177890631 → 0.00177891002（7-9 位有效数字相同）。
- **机制**：双积分器（K=C=0）+ 固定相位相混音调 → `P(t)=∫F` 在窗口内有非零均值（尽管 F 零均值周期）→
  q 呈**稳态线性斜坡**。预滚只把斜坡起点（常数偏移）平移，斜率与波形不变——
  **斜坡不是"启动瞬态"，旧分析（2026-08-23）的归因修正**。
- **后果**：泄漏比不降反升（DC 偏移计入 e_other）；raw 8ch 学习地板不变（0.569→0.560，仍 ≫ 1e-4 目标）——
  **预滚档下 8ch 线性 FNO 依然无法学习 inverse_disturbance**。
- **有效修复不变**：微分类 sddot（泄漏 1.99e-5）、diff8ch 地板 8.2e-5、13ch 地板 4.2e-6——预滚+子步档下全部成立。

## 5. 目标频谱净度

- **高频伪影：无**。18-250 Hz 带能量相对量级 ~1e-6（s 0.0068/6.3、sddot 0.0028/1021、u 400/1.9e8、d 22/3.6e5）。
  substeps=5 未引入任何高频伪影；u 泄漏反而略改善（1.8e-5→1.35e-5）。
- linearity 地板 rl2 ~4e-15（机器精度，f_dist = M qddot - B u 精确线性不变）。

## 6. 结论与建议

1. 预滚+子步数据档**已生成并验证可用**（物理一致、manifest 3.0、无伪影、生成快）。
2. 但**预滚不解决泄漏**：斜坡是稳态固有性质，泄漏比同种子不降反升（DC 偏移伪影）。
   → 后续 M2/M3/M4 决策应依赖 diff8ch/13ch 特征路线；raw 8ch 路线无论预滚与否都不可行（地板 ~0.56）。
3. 生成侧遗留（建议后续任务接线）：`DatasetConfig.from_mapping` 未解析 `preroll_s`/`integration_substeps`
   （YAML 键被静默忽略）+ CLI 未透传（Ruling 2）——本次以临时驱动绕过，正式配方需接线。

## 运行产物

`fourier_analysis.json` 为本 run 产物（不入库，仅记录）；适配分析脚本 `/tmp/analyze_p2.py`、临时 YAML 与驱动均在 /tmp。
