# OLED neural operator

FNO1d surrogate for **OLED microstage disturbance estimation**: maps actuator
force + encoder displacement trajectories to the disturbance force trajectory
(inverse disturbance problem), trained on the synthetic HDF5 benchmark
`neural_operator_2` (train 5000 / val 500 / test 200, 501 time points per
sample, dt = 1 ms).

> **Status**: exploration phase — pipeline fully usable. The
> train/evaluate loop is fully usable; 4 historical 500-epoch experiments are
> committed under `experiments/`. The dataset is fully uploaded locally and
> the complete pipeline was smoke-tested end-to-end on 2026-08-23.
>
> **Performance goal achieved (2026-08-24)**: `test.relative_l2 = 3.078e-6`
> (official `scripts/evaluate.py`, re-checked at 2.84e-6) vs. target `≤ 1e-4`
> — a 32–35× margin, at **zero training cost**, via analytic inverse-operator
> injection into a linear FNO (modes=252, full spectrum; F5). The best
> pure-gradient-trained model (E31) reached `2.03e-4`. See
> [`OPTIMIZATION_REPORT.md`](./OPTIMIZATION_REPORT.md) §8 and
> [`experiments/analysis/2026-08-23/ANALYSIS_REPORT.md`](./experiments/analysis/2026-08-23/ANALYSIS_REPORT.md).

## Quick links

- [中文文档（Chinese README）](./README_zh.md) — English version is the canonical source
- [`CLAUDE.md`](./CLAUDE.md) — AI assistant instructions
- [`configs/config.yaml`](./configs/config.yaml) — default experiment config
- [Data](#data) — `~/data/neural_operator_2` (default data root, read-only)
- [`src/utils/paths.py`](./src/utils/paths.py) — data root resolution logic

## Project structure

```
oled-neural-operator/
├── CLAUDE.md  README.md  README_zh.md  requirements.txt
├── configs/config.yaml       # default: 8ch (force + encoder_displacement) → 2ch
├── src/                      # data / models / training / utils (paths, config, ...)
├── scripts/                  # train.py / evaluate.py + train.sh / evaluate.sh
│   └── analysis/             # dataset-intrinsic analysis + analytic inverse-operator injection
├── tests/                    # unittest, stdlib only (config 5 + paths 7)
├── experiments/              # run artifacts; historical runs + 2026-08-23 analysis committed
├── outputs/                  # SLURM job artifacts (job_*) + final analytic model (ls_init_f1/, gitignored)
├── checkpoints/  logs/       # run-artifact placeholders (.gitkeep only)
└── .gitignore
```

Run artifacts live under `experiments/` (config snapshot, `train.log`,
`metrics.csv`, `history.json`, checkpoints). `outputs/`, `checkpoints/`, and
`logs/` are run-artifact placeholders for future artifact classes.

## Data

Dataset: `neural_operator_2`, one `.h5` per simulation. Data root resolution
(see `src/utils/paths.py`), highest priority first:

1. **Explicit path** — `--data-root` on the CLI, or a non-empty `data.root`
   in config (e.g. the absolute path stored in a historical checkpoint);
2. `$DATA_ROOT/neural_operator_2` — project-level environment variable;
3. `~/data/neural_operator_2` — final fallback, with an explicit stderr
   warning (once per process).

`configs/config.yaml` declares `data.root: null` so the path is resolved at
run time; `apply_data_root` writes the resolved absolute path back into the
run's config snapshot and checkpoints, keeping each run self-contained.

A second frozen dataset `neural_operator_3` (manifest 3.0, same generator
physics — equation `M q̈ = B u + f_dist`, `B` and `H` unchanged) adds
**preroll 0.5 s + integration substeps 5** for solver settling/accuracy and
is the official dataset of the M3/M5/M6 campaigns.  Its recommended_problem
is identical (inverse 8→2); M5/M6 run the 13-channel state-input formulation
(see below) via `configs/m5*.yaml` / `configs/m61_spectral.yaml`, which point
`data.root` explicitly at `~/data/neural_operator_3`.

## Default problem configuration

The default problem follows the dataset manifest's `recommended_problem`:
**8 input channels** (`force` ×4 + `encoder_displacement` ×4) →
**2 target channels** (`disturbance`), 501 time points.

```bash
scripts/train.sh --config configs/config.yaml
```

The committed campaign experiments (historical 4 × 500-epoch FNO runs, and
the M3/M5/M6 campaigns) instead use the 13-channel state-input formulation
(`force` + `displacement` + `velocity` + `acceleration` → `disturbance`)
on `neural_operator_3`. Reproduce it with:

```bash
scripts/train.sh \
  --input-fields force displacement velocity acceleration \
  --target-fields disturbance
```

## Reproducibility

- **Seeding**: `seed` in config drives PyTorch generators and worker seeding
  (`src/utils/seed.py`); run directory names embed the seed.
- **Determinism**: DataLoader order is reproducible via seeded generators;
  training/validation/test order is fixed.
- **Checkpoints**: v2 schema stores the resolved config, legacy args, and RNG
  state (python/numpy/torch/cuda); `--resume` restores everything and can
  swap LR/scheduler.
- **Tests**: `python -m unittest discover -s tests -v` (stdlib only, no
  torch/h5py required).
- **M6 pure-spectral recipe** (SGD-trained gate winner, `OPTIMIZATION_REPORT.md` §11):
  ```bash
  .venv/bin/python scripts/analysis/spectral_whiten.py --data-root ~/data/neural_operator_3 \
    --output outputs/spectral_whiten/whiten.pt        # deterministic, ~7 s
  .venv/bin/python scripts/train.py --config configs/m61_spectral.yaml   # 120 ep, ~3.2 min CPU
  .venv/bin/python scripts/evaluate.py --checkpoint <run>/best_model.pt  # official acceptance
  ```

## Status

| Stage | Description | Now |
| --- | --- | --- |
| Exploration | Training/eval loop usable, 4 historical runs committed | ✅ |
| Performance goal | `test.relative_l2 ≤ 1e-4` on `neural_operator_3` (13ch) | ✅ **3.078e-6** (F5, zero-training analytic injection, §8.3) |
| Hard gate via pure training (M6) | `test.relative_l2 ≤ 1e-4` on `neural_operator_3`, SGD-trained | ✅ **4.87e-6** worst seed (20.5× margin) — pure-spectral shared map, 26 trainable params, zero injection |
| 8ch recommended-problem gate (SGD) | `test.relative_l2 ≤ 1e-4` on the official 8ch problem (force + encoder_displacement → disturbance), SGD-trained | ✅ **9.480689e-05** (3/3 seeds, 5.4% margin, 2026-08-26) — PCA-1280 projection + fixed-diag-preconditioned SGD+momentum, zero injection (§12.5) |
| Best learned model | Pure-spectral shared map, SGD 120ep (M6 S5a recipe) | `2.84e-6`–`4.87e-6` (3 seeds) — replaces the E31 FNO limit `2.03e-4` |
| Maturity | Reproducible docs + NOTES.md for all campaigns | — |
| Publication | Paper-ready experiments and figures | — |

---

This project trains and evaluates a one-dimensional Fourier neural operator on
the OLED microstage HDF5 dataset. Experiment choices live in
`configs/config.yaml`; Python modules own construction and runtime behavior.

Train with the configuration defaults:

```bash
scripts/train.sh --config configs/config.yaml
```

Set `PYTHON_BIN` when the desired interpreter is not already active:

```bash
PYTHON_BIN="$HOME/miniconda3/envs/d2l/bin/python" \
  scripts/train.sh --config configs/config.yaml --device cuda
```

Command-line flags override the loaded config. `--epochs` is the number of
additional epochs when `--resume` is supplied.

Select Adam instead of the default AdamW through configuration or the CLI:

```bash
scripts/train.sh --optimizer adam --learning-rate 1e-3
```

## Continue training from a checkpoint

Pass `--resume` a `last_model.pt` checkpoint and set `--epochs` to the number
of additional epochs. To restart from a specific learning rate, pass
`--learning-rate` and do not pass `--keep-resume-learning-rate`:

```bash
PYTHON_BIN="$HOME/miniconda3/envs/d2l/bin/python" \
  scripts/train.sh \
  --resume experiments/PREVIOUS_RUN/last_model.pt \
  --epochs 500 \
  --learning-rate 1e-5 \
  --output-dir experiments/continued_lr1e-5 \
  --device cuda
```

The model and optimizer state are restored, the optimizer learning rate is
reset to `1e-5`, and a fresh configured scheduler is created for the additional
500 epochs. Use a new output directory so the previous run remains intact.

To retain the learning rate stored in the checkpoint instead, add
`--keep-resume-learning-rate` and omit `--learning-rate` unless you also want
to change the configured scheduler's reference learning rate.

Each saved run creates an isolated directory beneath `experiments/` unless
`--output-dir` is explicitly supplied. A run contains the resolved
`config.yaml`, `train.log`, `metrics.csv`, `history.json`, and the enabled
`best_model.pt` and `last_model.pt` checkpoints. An explicit non-empty output
directory is protected unless `--overwrite` is supplied.

Evaluate without retraining:

```bash
scripts/evaluate.sh --checkpoint experiments/RUN/best_model.pt
```

Version-2 checkpoints include the resolved config and reproducibility state,
while retaining the historical `args`, `model_state_dict`,
`optimizer_state_dict`, and `scheduler_state_dict` keys.

## Package layout

- `src/data`: dataset indexing, preprocessing selection, and DataLoader assembly
- `src/models`: the FNO implementation and model factory
- `src/training`: component factories, epoch loops, validation, and trainer
- `src/utils`: config, checkpoints, logging, device selection, and seeding
- `scripts/train.py`: training CLI and application assembly
- `scripts/evaluate.py`: checkpoint-only evaluation
- `scripts/train.sh` and `scripts/evaluate.sh`: direct shell launchers
