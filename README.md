# JEMA — final paper code

This branch publishes the frozen source used for the September 25–26 final
ordered-replay SMAC suite. `src/` is copied unchanged from the deployed suite,
not replaced by the earlier `learning-wm` implementation. The deployed DreamerV3
infrastructure is vendored under `external/dreamerv3`, with its license.

## Installation

Use Python 3.11, CUDA 12 and StarCraft II **4.10.0** with the SMAC v1 maps.

```bash
uv sync --locked --extra smac --extra cuda12
export SC2PATH=/path/to/StarCraftII
PYTHON=.venv/bin/python scripts/run_final.sh --map 2s3z --seed 1302 --arm jema
```

The launcher trains to the map budget and then runs a dedicated 100-episode
greedy evaluation. Outputs go into a fresh directory under `runs/`.
Set `WANDB_PROJECT`, optionally `WANDB_ENTITY` and `WANDB_RUN_GROUP`, to enable
W&B. Training and evaluation get separate unique IDs. Assign a GPU using
`CUDA_VISIBLE_DEVICES`; launch one process per GPU.

## Final suite

Training seeds: **1302, 2771, 7636**.

| Budget | Maps |
| --- | --- |
| 100k | 2m_vs_1z, 2s_vs_1sc, 2s3z, 3m, 3s_vs_3z, 3s_vs_4z, 8m, MMM, so_many_baneling |
| 200k | 3s_vs_5z, 2c_vs_64zg |
| 400k | corridor |

| `--arm` | Outcome → local | Factual JEPA → local |
| --- | ---: | ---: |
| jema | 1.0 | 0.1 |
| outcomes (O-JEMA) | 1.0 | 0.0 |
| jepa_only (G-JEPA) | 0.0 | 0.1 |
| detached | 0.0 | 0.0 |

The primary ablation maps are 2s3z, 3s_vs_4z and 2c_vs_64zg. These switches
control gradients into local features, not removal of the joint objectives.
This documents the experiment protocol, not a claim that every run completed.

## Protocol

The launcher reproduces the final manifest's overrides in
`scripts/train_args.json`: ordered replay at update boundaries, compact replay,
train ratio 128, no pretraining burn-in, categorical policy readout, actor LR
3e-5. All other settings come from the frozen `baseline` configuration in
`src/majepa/configs.yaml`. Do not invoke that profile alone to reproduce JEMA;
use the launcher so the suite overrides are applied.

Final evaluation uses four workers, 100 episodes total and environment seeds
`training_seed + 100000 + worker_index`. It evaluates the endpoint checkpoint,
not the best periodic checkpoint. Periodic evaluation is separate.

The suite ran on A100 PCIe GPUs. Identical source and settings do not guarantee
bitwise equivalence across hardware/software stacks. No exact training-resume
claim is made. The launcher is portable; cloud credentials, pod management,
queue state, results and diagnostic scripts are intentionally excluded.

## Source layout

`src/majepa/main.py` constructs the experiment; `train.py` schedules collection
and updates; `evaluation.py` handles evaluation; `marl/`, `training/`,
`world_model/` and `models/` contain the agent and losses. `envs/` contains the
environment adapters. No algorithm refactoring was performed for this release.
