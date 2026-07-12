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

The important pins are:

- `dm-meltingpot==2.4.0`
- `dmlab2d==1.0.0`
- `shimmy[meltingpot]==2.0.1`
- `jaxmarl[algs]==0.1.0`
- `jax==0.4.36`
- `jaxlib==0.4.36`

### CUDA

```bash
uv sync --python 3.11 --extra dev --extra cuda12
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

Each run writes:

- `config.json`
- `versions.json`
- `random_baseline.json`
- `initial_policy_evaluation.json`
- `metrics.jsonl`
- `returns.png`
- `checkpoint/checkpoint.msgpack`
- `checkpoint/metadata.json`
- `reload_evaluation.json`
- `outcome.json`

The top-level experiment directory also writes `summary.json`.

Each `metrics.jsonl` row includes rollout diagnostics for debugging learning
failures:

- sampled action counts/frequencies, both aggregate and per agent;
- sampled-policy entropy, aggregate and per agent;
- rollout rewards and completed-episode returns split by agent;
- value prediction mean/std, GAE target mean/std, and value explained variance;
- generic info/event counters, including coin-related and `coin_consumed` keys
  if the Melting Pot wrapper exposes them.


## Tests

```bash
uv run pytest
```
