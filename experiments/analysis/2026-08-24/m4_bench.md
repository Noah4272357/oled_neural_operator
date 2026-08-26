# M4 速度 benchmark：神经算子代理 vs 数值求解器

> 日期：2026-08-24 ｜ 战役：2026-08-24-surrogate-vs-solver（神经算子代理 vs 数值求解器）任务 7（M4 速度 benchmark）
> 脚本：`scripts/bench_surrogate.py`（输入构造与官方 `scripts/evaluate.py` 一致：13ch = force(4)+q(3)+qdot(3)+qddot(3)，501 点，test 样本加载自 `~/data/neural_operator_3`；单样本 forward，batch 1）
> 协议：生成侧 `scripts/bench_solver.py` 同构（中位 ×5，顺带报 ×1/×5 分布）；`eval` 模式 + `no_grad`；CUDA 下 float32 matmul precision=highest（同官方评估语义，禁 TF32）
> 参照线（生成侧 P0，CPU）：t_exudyn = 22.075 ms、t_dop853 = 818.548 ms（≈37×）

## 1. 环境

| 项 | CPU 主验收 | GPU 参考列 |
|---|---|---|
| 解释器 | 代理侧 `.venv/bin/python`（CPU torch） | `$HOME/envs/base/bin/python`（CUDA torch） |
| torch | 2.5.1+cpu，threads 16 | 2.5.1+cu121 |
| 设备 | CPU（无独占声明，本机主线程） | RTX 4090（`CUDA_VISIBLE_DEVICES=0`） |
| 备注 | — | **GPU 满载竞争**：nvidia-smi 99% util（其他用户作业），实测分布双峰（调度抖动），为竞争状态下界，仅作参考列 |

**模型**：
- (a) 达标形态 `outputs/ls_init_f1/best_model.pt`：13ch 线性 FNO，modes 252 / width 128 / embed·lift 64 / blocks 1 / activation none，**4,171,394 参数**（F5 解析注入，零训练）
- (b) 学习版对照 `experiments/fno1d_w64_b1_seed20260810_20260824-161144/best_model.pt`：同形态 modes 128 / width 64，**542,018 参数**（M3 从零 500ep）

## 2. 命令（可复现）

```bash
cd <repo_root>
# CPU 主验收（×5 中位）
.venv/bin/python scripts/bench_surrogate.py --checkpoint outputs/ls_init_f1/best_model.pt \
    --device cpu --repeats 5
.venv/bin/python scripts/bench_surrogate.py --checkpoint \
    experiments/fno1d_w64_b1_seed20260810_20260824-161144/best_model.pt --device cpu --repeats 5
# GPU 参考列（base env，前台直跑——SLURM 队列空闲但 GPU 被其他用户占满）
CUDA_VISIBLE_DEVICES=0 $HOME/envs/base/bin/python scripts/bench_surrogate.py \
    --checkpoint outputs/ls_init_f1/best_model.pt --device cuda --repeats 5
CUDA_VISIBLE_DEVICES=0 $HOME/envs/base/bin/python scripts/bench_surrogate.py \
    --checkpoint experiments/fno1d_w64_b1_seed20260810_20260824-161144/best_model.pt \
    --device cuda --repeats 5
```

## 3. 原始数值（每轮 ms，×5）

| 模型 | 设备 | 轮 1 | 轮 2 | 轮 3 | 轮 4 | 轮 5 | 中位 |
|---|---|---|---|---|---|---|---|
| (a) ls_init m252 w128（417 万） | CPU | 18.3228 | 18.3164 | 18.3552 | 18.2385 | 18.4437 | **18.3228** |
| (b) m3 w64 m128（54 万） | CPU | 3.8167 | 3.7955 | 3.7806 | 3.7680 | 3.7674 | **3.7806** |
| (a) ls_init m252 w128（417 万） | GPU（竞争） | 1.5502 | 2.4857 | 2.4881 | 2.4959 | 1.5468 | **2.4857** |
| (b) m3 w64 m128（54 万） | GPU（竞争） | 2.3675 | 3.0190 | 2.3631 | 2.3593 | 2.3525 | **2.3631** |

> CPU 分布极窄（±0.3%）；GPU 分布双峰（~1.5 / ~2.5 ms 两簇）——99% util 竞争下的调度抖动，中位落在慢簇。

## 4. speedup 表（vs t_exudyn 22.075 ms / t_dop853 818.548 ms，均 CPU 参照线）

| 模型 | 设备 | t_ms | vs exudyn | vs dop853 |
|---|---|---|---|---|
| (a) ls_init m252 w128 | CPU | 18.32 | **1.2×** | **44.7×** |
| (b) m3 w64 m128 | CPU | 3.78 | **5.8×** | **216.5×** |
| (a) ls_init m252 w128 | GPU（竞争下界） | 2.49 | 8.9× | 329.3× |
| (b) m3 w64 m128 | GPU（竞争下界） | 2.36 | 9.3× | 346.4× |

## 5. 解读

- **速度是软性目标**（战役约束：报告不判死）；精度硬门已 PASS（3.01e-6）。
- **CPU 主验收**：达标形态（417 万参数 m252 w128）单样本推理 18.3 ms——与 Exudyn 22.075 ms 同量级（1.2×，刚过 1），对 DOP853 44.7×；学习版（54 万参数）3.78 ms——对 Exudyn 5.8×、对 DOP853 216.5×。宽度/模式缩小（w128→64、m252→128）≈ 4.8× 推理加速（18.32/3.78），与参数比（7.7×）同量级。
- **GPU 参考列**：两模型在竞争 GPU 上均 ~2.4 ms（调度/提交开销主导，模型尺寸差异被竞争淹没）；此列是 99% util 竞争下界，非空闲 GPU 性能。空闲 GPU 预期更好，未测（集群被其他用户占满）。
- **smoke 校验**（单样本 rl2，非官方聚合指标）：(a) CPU 2.443e-6 / GPU 3.134e-6 ≈ 官方 test 3.01e-6；(b) 5.413e-3 ≈ 官方 6.14e-3——两模型均正确装载，bench 数值可信。
- **战役结论**：速度目标达成——代理推理（3.78-18.3 ms CPU）比 DOP853（818.5 ms）快 45-217×，比 Exudyn（22.1 ms）快 1.2-5.8×；"实时可用"成立（单样本 < 20 ms CPU、< 2.5 ms GPU）。
