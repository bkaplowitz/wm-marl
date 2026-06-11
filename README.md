# JaxMARL-Style PPO on Melting Pot

This project validates a JAX/JaxMARL-style PPO training pipeline on Melting Pot
substrates through Shimmy. It does **not** claim that the upstream JaxMARL
baseline scripts run unchanged on Melting Pot. JaxMARL is used as a dependency
and implementation reference; the reusable local IPPO/MAPPO trainers in this
repository are the project baselines for future work.

## Architecture

- Melting Pot / dmlab2d: Python-side substrate stepping.
- Shimmy: PettingZoo-style compatibility wrapper.
- `world_marl.envs.MeltingPotVectorAdapter`: batching, RGB normalization, reset,
  optional scalar-observation channels, step, auto-reset, and rollout-friendly
  tensors.
- JAX / Flax / Distrax / Optax: IPPO and MAPPO policies, GAE, and PPO updates.

The first milestone targets macOS arm64 with Python 3.11 and the `coins`
substrate.

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

### CUDA / A100 Setup

On a Linux CUDA 12 machine, such as an A100 partition, install the CUDA-enabled
JAX extra:

```bash
uv sync --python 3.11 --extra dev --extra cuda12
```

For shared or MIG-style GPU slices, it is usually safer to avoid aggressive JAX
memory preallocation:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# or, if preallocation is preferred:
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.60
```

Verify that JAX sees the GPU before launching a long run:

```bash
uv run world-marl-verify-install \
  --require-gpu \
  --algorithm mappo \
  --observation-size 44 \
  --include-observation-scalars \
  --append-agent-id
```

Melting Pot/dmlab2d environment stepping remains Python-side. The A100 speeds up
JAX/Flax policy inference and PPO updates, especially with larger rollouts and
higher-resolution observations.

## Level A: Integration Pass

```bash
uv run world-marl-verify-install
```

This imports the stack, constructs `coins`, resets/steps the environment,
collects a short rollout, and runs one jitted PPO update.

## Level B: Learning Validation

For a faster, weaker learning probe on CPU, downsample the RGB observations and
shorten episodes:

```bash
uv run world-marl-train-e2e \
  --algorithm ippo \
  --substrate coins \
  --num-envs 4 \
  --rollout-steps 64 \
  --total-env-steps 10000 \
  --eval-episodes 5 \
  --num-runs 1 \
  --max-cycles 200 \
  --observation-size 22 \
  --negative-control none \
  --min-improvement 0.0
```

This is useful for iteration, but the full validation command below is the
stronger acceptance test.

For an A100-backed MAPPO run, start with a larger but still practical single-seed
validation. MAPPO uses each agent's local observation for the actor and a
centralized critic observation built from all agents' observations plus a
target-agent identity channel. `--include-observation-scalars` appends scalar
Melting Pot observation keys, such as `COLLECTIVE_REWARD`, as constant channels
after RGB; `--append-agent-id` then appends one-hot identity channels.

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

The command runs repeated training runs and a frozen-policy negative control by
default. Each run evaluates three policy baselines:

- random policy;
- initial untrained policy;
- final reloaded checkpoint policy.

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

Pass criteria:

- all runs complete;
- checkpoints reload and evaluate in a fresh subprocess;
- trained policy beats random in at least two thirds of runs;
- trained policy beats the initial untrained policy in at least two thirds of runs;
- aggregate trained mean return beats both random and initial policy by at least
  `0.2` per agent;
- late training-window mean beats early training-window mean;
- the negative control does not beat its own initial untrained policy by the same
  improvement threshold.

## Tests

```bash
uv run pytest
```

The test suite covers adapter shape/reset behavior, GAE values, PPO parameter
updates, synthetic surrogate improvement, checkpoint equality, fixed-policy
evaluation, and a real Melting Pot observation forward pass when the runtime is
available.
