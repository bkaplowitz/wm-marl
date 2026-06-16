# JaxMARL + MeltingPot POC

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

### State-Representation World Model Validation

Before using flow matching or imagined rollouts, validate that a simple world
model can fit the rollout representation itself. This command collects real
Melting Pot `coins` transitions:

```text
obs_t, joint_action_t, reward_t, done_t, obs_{t+1}
```

It embeds observations with deterministic pooled RGB/channel-stat features,
then trains a supervised residual model with four heads:

- next state representation: `z_t, a_t -> z_t + delta_z`;
- reward: `z_t, a_t -> r_t`;
- done: `z_t, a_t -> done_t`;
- behavior policy: `z_t -> a_t`.

Smoke/default run:

```bash
uv run world-marl-validate-state-model \
  --substrate coins \
  --num-envs 4 \
  --collect-steps 512 \
  --train-steps 1000 \
  --batch-size 256 \
  --observation-size 22 \
  --pool-size 4
```

The run writes `config.json`, `versions.json`, `transition_dataset.json`,
`representation.json`, `metrics.jsonl`, `training_summary.json`,
`prediction_metrics.json`, `sample_predictions.json`, `prediction_dashboard.png`,
`state_recoveries.png`, `checkpoint/`, `reload_evaluation.json`,
`evaluation.json`, and `outcome.json`.

The first pass criterion is intentionally modest: finite training losses,
checkpoint reload equality, state-conditioned behavior-policy prediction beating
the marginal action baseline, and at least one transition signal beating a
persistence or zero-delta baseline. Reward, done, full-state distribution,
changed-feature, delta, and nearest-frame recovery metrics are reported so we
can see what the representation recovers before adding a harder generative
model.

### Flow Matching / GMMs on Coins

The flow-matching code can now be exercised against the live Melting Pot
`coins` substrate. The workflow models a two-agent joint-action distribution:

1. collect joint actions from Shimmy/Melting Pot `coins`, either from random
   actions or from a saved IPPO/MAPPO checkpoint;
2. split those actions into train and heldout samples;
3. fit an empirical 2D GMM over normalized train action pairs
   `(player_0, player_1)`;
4. train the JAX flow-matching MLP on that GMM;
5. sample 2D points from the learned flow;
6. decode samples back to the two agents' discrete action IDs;
7. compare the generated distribution against heldout rollout actions,
   train-empirical, GMM-sample, and uniform baselines.

Checkpoint-source imitation run:

```bash
uv run world-marl-train-coin-flow \
  --target-source checkpoint \
  --policy-checkpoint runs/<e2e_run>/run_000/checkpoint \
  --num-envs 8 \
  --collect-steps 2048 \
  --train-steps 5000 \
  --batch-size 512 \
  --generated-samples 1024 \
  --eval-episodes 50 \
  --max-cycles 500 \
  --observation-size 44 \
  --include-observation-scalars \
  --append-agent-id
```

Larger random-source local/A100 run:

Each run writes `config.json`, `versions.json`, `rollout_dataset.json`,
`distribution_split.json`, `gmm.json`, `metrics.jsonl`,
`training_summary.json`, `generated_action_samples.json`,
`distribution_validation.json`, `distribution_validation.png`, `checkpoint/`,
`evaluation.json`, and `outcome.json`. The command prints stage updates and
progress bars by default; add `--quiet` to only emit the final JSON outcome.
The PNG is a distribution dashboard with probability heatmaps, absolute-error
heatmaps versus heldout actions, sorted action-pair probabilities, and JS/total
variation bars.

The key distribution fields are:

- `distribution_validation.flow_js_divergence`
- `distribution_validation.uniform_js_divergence`
- `distribution_validation.strict_flow_beats_uniform`
- `distribution_validation.reload_max_abs_point_diff`

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
