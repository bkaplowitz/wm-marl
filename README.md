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
  --sigreg-weight 0.05 \
  --controls none no-action-world-model shuffled-action-replay \
  --out-dir runs/dmc_jepa
```

Good first DMC tasks are `dmc:cartpole/swingup`, `dmc:pendulum/swingup`, and
`dmc:reacher/easy`. Start with state observations; pixel observations, online
data collection, and model/policy co-training are later milestones.

To test the next rung, add frozen-world-model policy training. This still does
not use MPC and does not update the JEPA backbone during actor/value training:

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
  --policy-objective candidate-distill \
  --num-policy-candidates 64 \
  --candidate-min-gap 0.001 \
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
  --sigreg-weight 0.05 \
  --controls none no-action-world-model shuffled-action-replay \
  --out-dir runs/dmc_jepa_policy
```

This second mode reports both world-model fit metrics and real-environment
policy returns:

- random policy return;
- freshly reset actor return before imagination training;
- real-return critic warmup diagnostics;
- trained actor return after frozen-model imagination training, using the best
  actor selected by periodic paired real-environment validation;
- paired no-action and shuffled-action controls.

The default policy objective is training-only candidate distillation: the
frozen latent model scores sampled action candidates for replay states, and the
actor is trained toward the best candidate only when the predicted action-value
gap is nontrivial. Evaluation still uses the direct actor; no MPC/search is used
at evaluation time.

The next rung turns the offline validation into an online data loop. After the
first frozen-model policy phase, the selected actor collects fresh real DMC
transitions, the replay buffer is updated, the world model is refit, and the
actor continues training in the updated latent model:

```bash
uv run world-marl-validate-dmc-world-model \
  --env dmc:cartpole/swingup \
  --num-envs 16 \
  --dmc-workers 1 \
  --collect-steps 8192 \
  --validation-steps 2048 \
  --train-steps 8000 \
  --critic-warmup-steps 1000 \
  --critic-horizon 32 \
  --policy-train-steps 3000 \
  --policy-objective candidate-distill \
  --num-policy-candidates 64 \
  --candidate-min-gap 0.001 \
  --imag-horizon 15 \
  --policy-selection-interval 500 \
  --policy-selection-episodes 20 \
  --policy-eval-episodes 30 \
  --online-iterations 1 \
  --online-collect-steps 2048 \
  --online-train-steps 3000 \
  --online-policy-train-steps 1500 \
  --batch-size 256 \
  --chunk-length 32 \
  --open-loop-horizon 15 \
  --latent-dim 128 \
  --regularizer sigreg \
  --sigreg-weight 0.05 \
  --controls none \
  --num-runs 3 \
  --out-dir runs/dmc_jepa_online_cartpole \
  --allow-fail
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
