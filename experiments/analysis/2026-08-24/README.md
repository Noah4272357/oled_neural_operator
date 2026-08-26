# P1 探针：13ch 线性 FNO modes=252 纯学习极限

> 日期：2026-08-24 ｜ 战役：2026-08-24-surrogate-vs-solver（神经算子代理 vs 数值求解器）P1 探针（代理侧）
> 任务 2：P1——13ch 线性 FNO modes=252 纯学习极限。

## 1. 运行信息

- **配置**：`configs/p1_linear_13ch_m252.yaml`（复制 `configs/f1_linear_13ch.yaml`，仅 `model.modes: 128 → 252`）
- **SLURM 脚本**：`slurm_scripts/train_p1_m252.sh`（参照 `slurm_scripts/train_linear_f1.sh` 的 SBATCH 头：4×RTX 4090 之一、GPU 1、base env 激活）
- **作业号 / 状态**：`30901` / COMPLETED（`logs/oled_p1_30901.out` 末尾 "Job finished."，`.err` 仅数据根回退警告；集群无 Slurm accounting，sacct 不可用）
- **run 目录**：`experiments/fno1d_w128_b1_seed20260810_20260824-135058/`（自动命名，产物含 best_model.pt / metrics.csv / history.json / train.log）
- **训练配置**：13ch 输入（force+displacement+velocity+acceleration → disturbance）、activation=none、width 128、modes 252、embed/lift 64、b64 / lr 2.83e-3、500ep、cosine eta_min 3e-6、relative_l2 loss
- **成本**：total 806.6 s（~13.4 min GPU），peak CUDA mem 286 MB

## 2. 官方评估结果（Step 4）

```bash
PYTHON_BIN=~/envs/base/bin/python \
  scripts/evaluate.sh --checkpoint experiments/fno1d_w128_b1_seed20260810_20260824-135058/best_model.pt
```

```json
{"checkpoint_epoch": 500, "metrics": {"mse": 0.00011768392460074967,
  "relative_l2": 0.005967686169658821, "rmse": 0.010848222186181}, "samples": 200, "split": "test"}
```

**test.relative_l2 = 5.97e-3**（best_model 即 last，epoch 500；best val rl2 = 6.05e-3；train 终值 rl2 = 7.51e-3）。

## 3. 对照与判定（Step 5）

| 探针 | 配置 | test.relative_l2 |
|---|---|---|
| f1（对照） | 13ch 线性 FNO modes=128，纯学习 | 6.03e-3（job 30892） |
| **P1（本探针）** | **13ch 线性 FNO modes=252，纯学习** | **5.97e-3** |
| F5（参照） | modes=252 解析逆算子注入（零训练） | 2.84e-6（outputs/ls_init_f1） |

**判定逻辑**（战役 spec）：
- `< 1e-4` → 候选 B（线性全频谱纯学习）可行，形态决策向 B 倾斜；
- `1e-4 ~ 1e-3` → 学习极限不足，依赖 P2/P3 生成侧改造；
- `> 1e-3` → 强依赖生成侧。

**判定：test.relative_l2 = 5.97e-3 > 1e-3 → 强依赖生成侧（P2/P3）**。

**要点**：
1. 模态数 128 → 252 对纯 SGD 学习极限**无收益**（6.03e-3 vs 5.97e-3，差异 ~1%），纯学习极限 ~6e-3 不是频谱截断所致，而是优化/泛化侧瓶颈（与 2026-08-23 数据集内蕴分析"纯 SGD 因固有屏障不可达 1e-4"一致）。
2. 同一 modes=252 下，解析注入（F5）达 2.84e-6，与学习极限相差 **~2000×**——能力在权重确定方式（解析 vs SGD），不在模态数。
3. 形态决策：P1 不支持候选 B（纯学习不可行），向依赖生成侧改造（P2/P3）倾斜；P1 同时为 P2/P3 提供代理侧基线。
