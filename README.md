# World MARL

## Official DreamerV3 Baseline

The DreamerV3 baseline runs the pinned `danijar/dreamerv3` implementation as
an external process. `world_marl` does not reimplement or replace its RSSM,
replay, losses, actor, critic, or training loop. The integration owns only
environment setup, launch metadata, fixed-checkpoint evaluation, step
accounting, and normalized artifacts.

Initialize the pinned source and create its dependency-isolated environment:

```bash
git submodule update --init --recursive
uv sync --python 3.11 --extra dev
uv run world-marl-setup-dreamerv3 --accelerator cpu     # macOS/local smoke
uv run world-marl-setup-dreamerv3 --accelerator cuda12  # Linux GPU
```

Run the canonical 500K DMC Reacher Easy experiment:

```bash
uv run world-marl-train-dmc-dreamerv3 \
  --task dmc_reacher_easy \
  --seed 0 \
  --total-env-steps 500000 \
  --platform cuda \
  --save-every-seconds 300 \
  --wandb-project world-marl \
  --wandb-entity osaze-obahor
```

The model, optimizer, replay, and imagination settings come directly from the
upstream `dmc_proprio` config. Use `--official-budget` for its current 1.1M-step
preset. `--debug --platform cpu` is available only for installation smoke tests
and is not a learning result.

Evaluate the latest periodically saved checkpoint without selecting among
checkpoints:

```bash
uv run world-marl-eval-dmc-dreamerv3 \
  runs/dreamerv3/dmc_reacher_easy/seed_0/<timestamp> \
  --episodes 20 \
  --success-threshold 900
```

Each experiment contains the untouched upstream log directory plus:

- `launch.json`: exact command, upstream commit, configs, seed, and budget;
- `process.log`: complete upstream stdout/stderr;
- `normalized/training_episodes.jsonl`: online episode returns and steps;
- `normalized/training_curve.json`: 10K-transition bins;
- `normalized/official_reference.json`: bundled official five-seed curve;
- `normalized/training_curve.png`: this run against the official curve;
- `evaluation/*/evaluation_summary.json`: all held-out returns and separate
  train, evaluation, and total real-transition counts.

The canonical contract is recorded in
`configs/dreamerv3/dmc_proprio.yaml`. The pinned upstream revision is
`e3f02248693a79dc8b0ebd62c93683888ddaccfe`.

## JaxMARL + MeltingPot POC

## Architecture

- Melting Pot / dmlab2d: Python-side substrate stepping.
- Shimmy: PettingZoo-style compatibility wrapper.
- `world_marl.envs.MeltingPotVectorAdapter`: batching, RGB normalization, reset,
  optional scalar-observation channels, step, auto-reset, and rollout-friendly
  tensors.
- JAX / Flax / Distrax / Optax: IPPO and MAPPO policies, GAE, and PPO updates.

## Setup

```bash
uv sync --python 3.11 --extra dev
```

For Melting Pot/JaxMARL work, install the Melting Pot extra:

```bash
uv sync --python 3.11 --extra dev --extra meltingpot
```

For DMC JEPA work, install only the DMC extra. This avoids pulling TensorFlow
through Melting Pot on small cloud disks:

```bash
uv sync --python 3.11 --extra dmc
```

The important optional Melting Pot pins are:

- `dm-meltingpot==2.4.0`
- `dmlab2d==1.0.0`
- `shimmy[meltingpot]==2.0.1`

The common JAX/JaxMARL pins are:

- `jaxmarl[algs]==0.1.0`
- `jax==0.4.36`
- `jaxlib==0.4.36`

### CUDA

```bash
uv sync --python 3.11 --extra dev --extra cuda12
```

DMC + CUDA on cloud:

```bash
uv sync --python 3.11 --extra dmc --extra cuda12
```

### Basic CMDs 

```bash
uv run world-marl-train-e2e \
  --algorithm mappo \
  --substrate coins \
  --num-envs 8 \
  --rollout-steps 128 \
  --total-env-steps 200000 \
  --eval-episodes 50 \
  --num-runs 1 \
  --max-cycles 500 \
  --observation-size 44 \
  --include-observation-scalars \
  --append-agent-id \
  --stochastic-eval \
  --learning-rate 0.00025 \
  --update-epochs 4 \
  --num-minibatches 8 \
  --ent-coef 0.02 \
  --negative-control freeze-policy \
  --min-improvement 0.2
```

```bash
uv run world-marl-train-e2e \
  --algorithm ippo \
  --substrate coins \
  --num-envs 4 \
  --rollout-steps 128 \
  --total-env-steps 100000 \
  --eval-episodes 50 \
  --num-runs 3
```

Each end-to-end `metrics.jsonl` row includes rollout diagnostics for debugging
learning failures:

- sampled action counts/frequencies, both aggregate and per agent;
- sampled-policy entropy, aggregate and per agent;
- rollout rewards and completed-episode returns split by agent;
- value prediction mean/std, GAE target mean/std, and value explained variance;
- generic info/event counters, including coin-related and `coin_consumed` keys
  if the Melting Pot wrapper exposes them.

### JEPA Model-Based RL

The maintained JEPA route learns action-conditioned latent dynamics and trains a
stochastic actor plus critic through imagined latent rollouts. It uses no
observation decoder, no EMA target encoder, no real-environment checkpoint
search, and no task-specific failure buffer.

Install DMC and run a local smoke test:

```bash
uv sync --extra dmc
uv run python -m world_marl.scripts.write_dmc_vector_launcher \
  --preset smoke \
  --tasks reacher/easy \
  --seeds 0 \
  --gpus 0 \
  --out-root runs/jepa_smoke
bash runs/jepa_smoke/launcher.sh
```

Generate the fixed 500K experiment:

```bash
uv run python -m world_marl.scripts.write_dmc_vector_launcher \
  --preset jepa_500k \
  --tasks reacher/easy cartpole/swingup finger/spin cheetah/run walker/walk \
  --seeds 0 1 2 3 4 \
  --gpus 0 1 \
  --out-root runs/jepa_500k
bash runs/jepa_500k/launcher.sh
```

The 500K preset uses 497,664 training transitions plus 1,280 held-out
world-model validation transitions. Final 20-episode evaluation is tracked
separately. All W&B curves use actual training environment steps on the x-axis,
and the reported final score comes from the deterministic latest policy.

For the model, objective, schedule, and reproducibility contract, see
[`src/world_marl/jepa/ARCHITECTURE.md`](src/world_marl/jepa/ARCHITECTURE.md).

## Tests

```bash
uv run pytest
```
