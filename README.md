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

### Legacy State-Representation Validation

This older validator fits a supervised one-step model to Melting Pot-style image
rollouts. It is useful historical scaffolding, but it is no longer the main
CoinGame path. The active world-model integration below uses JaxMARL CoinGame
vector states.

```text
obs_t, joint_action_t, reward_t, done_t, obs_{t+1}
```

It embeds observations with deterministic pooled RGB/channel-stat features,
then trains a supervised residual model with five heads:

- next state representation: `z_t, a_t -> z_t + delta_z`;
- reward: `z_t, a_t -> r_t`;
- reward event: `z_t, a_t -> 1[|r_t| > eps]`;
- done: `z_t, a_t -> done_t`;
- behavior policy: `z_t -> a_t`.

Minibatches oversample rare reward events and large state-change transitions,
and the transition objective includes full-state, delta, and changed-feature
loss terms. This keeps mostly-static pixels/features from hiding the parts of
the transition that actually changed.

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
changed-feature, delta, reward-event, and nearest-frame recovery metrics are
reported so we can see what the representation recovers before adding a harder
generative model.

### Flow Matching / GMMs on JaxMARL CoinGame

The flow-matching code is exercised against native JaxMARL CoinGame. The
workflow models a two-agent joint-action distribution:

1. collect joint actions from JaxMARL CoinGame, either from random actions or
   from a saved vector-mode IPPO/MAPPO checkpoint;
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
  --max-cycles 500
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

Milestone 1 state-conditioned action validation uses the same rollout source,
but changes the objective from global `p(joint_action)` to
`p(joint_action | state)`. It collects `(state_t, joint_action_t)` pairs,
trains a conditional flow over normalized joint actions, trains a categorical
behavior-cloning baseline as a discrete sanity check, and evaluates on heldout
states. This is policy-distribution imitation, not a dynamics/world-model step.

```bash
uv run world-marl-train-coin-flow \
  --conditional-action \
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
