# JaxMARL + MeltingPot POC

## Architecture

- Melting Pot / dmlab2d: Python-side substrate stepping.
- Shimmy: PettingZoo-style compatibility wrapper.
- `world_marl.envs.MeltingPotVectorAdapter`: batching, RGB normalization, reset,
  optional scalar-observation channels, step, auto-reset, and rollout-friendly
  tensors.
- `world_marl.envs.GymnaxVectorAdapter`: single-agent Gymnax environments exposed
  as one-agent vector tasks via `--substrate gymnax:<env-id>`.
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

Single-agent Gymnax environments use the same trainer with a singleton agent
axis:

```bash
uv run world-marl-train-e2e \
  --algorithm ippo \
  --substrate gymnax:CartPole-v1 \
  --num-envs 16 \
  --rollout-steps 128 \
  --total-env-steps 50000 \
  --eval-episodes 20 \
  --num-runs 1 \
  --max-cycles 500 \
  --negative-control none \
  --min-improvement 0.0
```

### SIGReg-JEPA CartPole Milestone

The `world-marl-train-jepa` command trains a minimal decoder-free SIGReg-JEPA
imagination actor-critic on single-agent Gymnax tasks. For milestone 1, the
target is CartPole and the model learns latent prediction, reward prediction,
continue prediction, and actor/critic updates from imagined latent rollouts.

The JEPA target branch uses `stopgrad(encoder(o_t+k))`, the model has no
observation decoder, and actor/critic updates freeze the encoder/world-model
backbone. The default regularizer is a JAX implementation of the sketched
SIGReg objective used by LeWorldModel; the older second-order isotropy penalty
is still available with `--regularizer isotropy` for ablations. Controls such as
`no-action-world-model`, `shuffled-action-replay`, `no-policy-update`,
`no-sigreg`, and `weak-sigreg` are first-class CLI modes.

```bash
uv run world-marl-train-jepa \
  --env gymnax:CartPole-v1 \
  --num-envs 32 \
  --total-env-steps 25000 \
  --replay-capacity 50000 \
  --chunk-length 32 \
  --batch-size 128 \
  --model-updates-per-iter 2 \
  --model-horizon 1 \
  --imag-horizon 5 \
  --context-window 1 \
  --latent-dim 128 \
  --regularizer sigreg \
  --sigreg-weight 0.05 \
  --eval-episodes 100 \
  --num-runs 5 \
  --controls none no-action-world-model shuffled-action-replay no-policy-update \
  --out-dir runs/jepa_cartpole
```

Each run writes JEPA model metrics, open-loop latent rollout metrics,
collapse/SIGReg diagnostics, evaluation returns, a checkpoint, and reload
evaluation artifacts.

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
