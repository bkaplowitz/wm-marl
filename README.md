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

### DMC JEPA World-Model Validation

The `world-marl-validate-dmc-world-model` command is the first
continuous-control rung. It collects random state-observation rollouts from
Google DeepMind Control Suite tasks and fits the SIGReg-JEPA latent world model
to:

```text
p(z_next, reward, continue | z, continuous_action)
```

With the default `--policy-train-steps 0`, this command does **not** train a
continuous actor and does not use MPC. It answers the narrower question: can the
latent dynamics, reward head, and continue head fit action-conditioned
continuous-control transitions better than no-action and shuffled-action
controls?

Install the optional DMC dependency first:

```bash
uv sync --extra dmc
```

Then run a small CartPole Swingup fit:

```bash
uv run world-marl-validate-dmc-world-model \
  --env dmc:cartpole/swingup \
  --num-envs 16 \
  --collect-steps 2048 \
  --validation-steps 512 \
  --train-steps 5000 \
  --batch-size 256 \
  --chunk-length 32 \
  --open-loop-horizon 5 \
  --latent-dim 128 \
  --regularizer sigreg \
  --regularizer-weight 0.05 \
  --controls none no-action-world-model shuffled-action-replay frozen-random-world-model \
  --out-dir runs/dmc_jepa
```

Good first DMC tasks are `dmc:cartpole/swingup`, `dmc:pendulum/swingup`, and
`dmc:reacher/easy`. For faster accelerator-friendly iteration, the same command
also accepts Brax tasks with `--env brax:<env_name>`.

Install the optional Brax dependency with:

```bash
uv sync --extra brax
```

Then run the same validation loop on a JAX-native Brax environment:

```bash
uv run world-marl-validate-single-agent-world-model \
  --env brax:reacher \
  --num-envs 256 \
  --collect-steps 2048 \
  --validation-steps 512 \
  --train-steps 5000 \
  --batch-size 512 \
  --chunk-length 32 \
  --open-loop-horizon 5 \
  --latent-dim 128 \
  --regularizer sigreg \
  --regularizer-weight 0.05 \
  --controls none no-action-world-model shuffled-action-replay frozen-random-world-model \
  --out-dir runs/brax_jepa
```

Start with state observations; pixel observations and larger recurrent/history
models are later milestones.

As a first sequence-model probe, use a larger context window in model-only mode.
This validates history-conditioned latent dynamics without yet training the
actor from history-initialized imagined rollouts:

```bash
uv run world-marl-validate-single-agent-world-model \
  --env brax:reacher \
  --num-envs 512 \
  --collect-steps 2048 \
  --validation-steps 1024 \
  --train-steps 5000 \
  --batch-size 1024 \
  --chunk-length 32 \
  --context-window 4 \
  --open-loop-horizon 15 \
  --latent-dim 128 \
  --regularizer sigreg \
  --regularizer-weight 0.05 \
  --controls none no-action-world-model shuffled-action-replay \
  --out-dir runs/brax_jepa_history
```

To test Milestone 2, add frozen-world-model policy training. This uses direct
latent-imagination RL as the main algorithm: the actor is optimized through
imagined latent rollouts in the frozen world model, with no MPC/search and no
JEPA backbone updates during actor/value training:

```bash
uv run world-marl-validate-dmc-world-model \
  --env dmc:cartpole/swingup \
  --num-envs 16 \
  --dmc-workers 16 \
  --collect-steps 4096 \
  --validation-steps 1024 \
  --train-steps 5000 \
  --critic-warmup-steps 1000 \
  --critic-horizon 32 \
  --policy-train-steps 3000 \
  --policy-objective direct \
  --policy-return-mode reward-only \
  --imag-horizon 15 \
  --policy-selection-interval 500 \
  --policy-selection-episodes 20 \
  --policy-eval-episodes 100 \
  --policy-eval-num-envs 16 \
  --value-clip 100 \
  --batch-size 256 \
  --chunk-length 32 \
  --open-loop-horizon 15 \
  --latent-dim 128 \
  --regularizer sigreg \
  --regularizer-weight 0.05 \
  --controls none no-action-world-model shuffled-action-replay frozen-random-world-model \
  --out-dir runs/dmc_jepa_policy
```

This second mode reports both world-model fit metrics and real-environment
policy returns:

- random policy return;
- freshly reset actor return before imagination training;
- real-return critic warmup diagnostics;
- trained actor return after frozen-model imagination training, using the best
  actor selected by periodic paired real-environment validation;
- paired no-action, shuffled-action, and frozen-random-world-model controls.

The next engineering rung turns the offline validation into an online data loop.
After the first frozen-model policy phase, the selected actor collects fresh real
Brax transitions, the replay buffer is updated, the world model is refit, and the
actor continues training in the updated latent model. Keep this as a lightweight
single-seed pipeline check before spending on multi-seed controls:

```bash
uv run world-marl-validate-single-agent-world-model \
  --env brax:reacher \
  --num-runs 1 \
  --num-envs 512 \
  --collect-steps 2048 \
  --validation-steps 1024 \
  --train-steps 3000 \
  --critic-warmup-steps 500 \
  --critic-horizon 32 \
  --policy-train-steps 1500 \
  --policy-objective direct \
  --policy-return-mode reward-only \
  --imag-horizon 15 \
  --policy-selection-interval 500 \
  --policy-selection-episodes 32 \
  --policy-eval-episodes 64 \
  --online-iterations 1 \
  --online-collect-steps 1024 \
  --online-train-steps 1500 \
  --online-policy-train-steps 1000 \
  --batch-size 1024 \
  --chunk-length 32 \
  --open-loop-horizon 15 \
  --latent-dim 128 \
  --regularizer sigreg \
  --regularizer-weight 0.05 \
  --controls none no-action-world-model \
  --out-dir runs/brax_jepa_reacher_online_dev
```

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
