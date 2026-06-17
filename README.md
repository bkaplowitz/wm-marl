# JaxMARL CoinGame World-Model Validation

## Architecture

Current focus is the native JaxMARL `coin_game`, not Melting Pot `coins`.

- `world_marl.envs.JaxMARLCoinGameVectorAdapter`: two-agent JaxMARL CoinGame
  exposed through the same vector-env interface used by training and
  validation.
- CoinGame observations are flat vector states with shape `(36,)` per agent.
- CoinGame has 5 discrete actions per agent.
- JAX / Flax / Distrax / Optax: IPPO/MAPPO policies, GAE, PPO updates, and
  flow-matching models.
- The Melting Pot adapter remains in the repo for earlier integration work, but
  it is not the current CoinGame validation target.

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
  --prefit-world-model \
  --num-envs 4 \
  --rollout-steps 128 \
  --total-env-steps 100000 \
  --eval-episodes 50 \
  --num-runs 3 \
  --wm-random-rollouts 8 \
  --wm-initial-rollouts 8 \
  --wm-fit-steps 500 \
  --wm-flow-type linear
```

### Conditional Action Flow on JaxMARL CoinGame

The current flow-matching milestone targets native JaxMARL CoinGame and models
the trained behavior policy's state-conditioned joint-action distribution:

```text
p(joint_action_t | state_t)
```

It collects `(state_t, joint_action_t)` pairs from either random actions or a
saved vector-mode IPPO/MAPPO checkpoint, trains a conditional flow over
normalized joint actions, trains a categorical behavior-cloning baseline as a
discrete sanity check, and evaluates on heldout states. This is
policy-distribution imitation, not a dynamics/world-model step.

```bash
uv run world-marl-train-coin-flow \
  --target-source checkpoint \
  --policy-checkpoint runs/<e2e_run>/run_000/checkpoint \
  --num-envs 128 \
  --collect-steps 2048 \
  --train-steps 5000 \
  --batch-size 1024 \
  --generated-samples 4096 \
  --flow-integration-steps 16 \
  --eval-episodes 50 \
  --max-cycles 50
```

That run writes `conditional_action_dataset.json`,
`conditional_action_split.json`, `conditional_action_validation.json`,
`conditional_action_validation.png`, `conditional_action_samples.json`,
`conditional_classifier_checkpoint/`, and `conditional_flow_checkpoint/`.
The main pass criteria are finite losses, classifier cross entropy beating the
marginal action baseline, conditional-flow action accuracy beating the marginal
baseline, conditional-flow distribution JS beating uniform, and checkpoint
reload equality.

### Discrete CoinGame Dynamics

The first actual world-model milestone predicts the environment transition and
reward:

```text
p(next_joint_state_{t+1}, reward_t | state_t, joint_action_t)
```

This uses the native JaxMARL CoinGame vector state. Each agent observes a
flattened `3 x 3 x 4` grid, so the model decodes every entity position into a
categorical cell id and predicts the next cell for each entity with softmax
heads. This is intentionally discrete rather than flow matching, because
CoinGame positions and actions are discrete. Reward is derived analytically from
the same `state_t, joint_action_t` pair and validated against the environment
reward.

```bash
uv run world-marl-train-coin-dynamics \
  --num-envs 128 \
  --collect-steps 4096 \
  --train-steps 5000 \
  --batch-size 1024 \
  --max-cycles 100 \
  --out-dir runs/coin_dynamics
```

The run writes `transition_dataset.json`, `metrics.jsonl`,
`training_summary.json`, `prediction_metrics.json`, `sample_predictions.json`,
`dynamics_training.png`, `checkpoint/`, `reload_evaluation.json`, and
`outcome.json`.

The main validation metric is exact heldout next-state prediction. Metrics are
reported separately for deterministic transitions and stochastic transitions:
when a coin is collected, CoinGame respawns that coin with environment RNG, so
the exact next coin location is not fully determined by `state, joint_action`
alone.

### PPO/MAPPO Artifacts

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
