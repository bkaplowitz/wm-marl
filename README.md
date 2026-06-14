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

### Flow Matching / GMMs on Coins

The flow-matching code can now be exercised against the live Melting Pot
`coins` substrate. The workflow models a two-agent joint-action distribution:

1. collect joint actions from Shimmy/Melting Pot `coins`, either from random
   actions or from a saved IPPO/MAPPO checkpoint;
2. fit an empirical 2D GMM over normalized action pairs `(player_0, player_1)`;
3. train the JAX flow-matching MLP on that GMM;
4. sample 2D points from the learned flow;
5. decode samples back to the two agents' discrete action IDs and evaluate them
   in `coins`.

This is a wiring/validation step for flow matching on the game, not yet a claim
that flow matching learns a strong coins strategy. The strongest next target
distribution should come from MAPPO, expert, or scripted rollouts.

Random-source smoke test:

```bash
uv run world-marl-train-coin-flow \
  --num-envs 1 \
  --collect-steps 8 \
  --train-steps 3 \
  --batch-size 16 \
  --generated-samples 8 \
  --eval-episodes 1 \
  --max-cycles 20 \
  --observation-size 22 \
  --flow-integration-steps 4
```

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

Use the same `--observation-size`, `--include-observation-scalars`, and
`--append-agent-id` flags that were used when the checkpoint was trained. The
flow run evaluates random, source-checkpoint, and flow-generated policies.

Larger random-source local/A100 run:

```bash
uv run world-marl-train-coin-flow \
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

Each run writes `config.json`, `versions.json`, `rollout_dataset.json`,
`gmm.json`, `metrics.jsonl`, `training_summary.json`,
`generated_action_samples.json`, `checkpoint/`, `evaluation.json`, and
`outcome.json`. The command prints stage updates and progress bars by default;
add `--quiet` to only emit the final JSON outcome.

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
