# JEPA Model-Based RL Architecture

This document describes the single maintained JEPA algorithm. Historical
checkpoint-search, champion, hard-start, candidate-refit, CEM, and task-specific
experimental paths have been removed from the training CLI.

## Algorithm

The model learns action-conditioned dynamics directly in representation space:

```text
z_t = encoder(observation_t)
p(z_{t+1}, reward_t, continue_t | z_history, action_history)
```

It never reconstructs observations. Actor and critic heads learn from latent
rollouts produced by this world model, which makes the complete system
model-based reinforcement learning rather than only representation learning.

One training phase is:

1. collect real transitions with the current stochastic policy;
2. append them to replay;
3. update the JEPA world model from the full replay;
4. update actor and critic through latent imagination;
5. continue with the resulting latest policy.

There is no real-environment checkpoint search, accept/reject gate, or champion
policy. The final deterministic evaluation uses the policy produced by the last
scheduled optimizer update.

## World Model

The maintained vector-control model contains:

- an MLP observation encoder;
- an MLP continuous-action encoder;
- a causal transformer over latent/action history;
- residual, normalized latent dynamics;
- reward and continuation heads.

The predictor uses two pre-norm transformer blocks, RoPE attention, four heads,
RMSNorm, SiLU, and GEGLU feed-forward layers in the reference configuration.
Both latent and transformer width are 128, with an eight-step context window.

The target latent is the same encoder applied to the future observation:

```text
target = stopgrad(encoder(observation_next))
```

The world-model objective combines cosine latent prediction, symlog two-hot
reward prediction, continuation BCE, and SIGReg anti-collapse regularization.
There is one online encoder and no EMA target encoder. The EMA in this algorithm
belongs to the critic, not the JEPA encoder.

## Actor-Critic

The actor and critic are three-hidden-layer MLPs of width 64 with LayerNorm.
The control stack retains the reusable DreamerV3 stabilization mechanisms while
keeping the JEPA world model unchanged:

- stochastic tanh-normal actor for training and collection;
- deterministic mean action for final evaluation;
- squash-aware entropy coefficient `3e-3`;
- REINFORCE actor gradient, avoiding gradients through model errors;
- 15-step latent imagination and lambda returns (`lambda=0.95`);
- EMA p95-p5 return normalization (`decay=0.99`);
- symlog two-hot value targets with clip `100`;
- EMA target critic (`decay=0.98`);
- slow-value regularization (`1.0`);
- replay critic loss (`0.3`) over every state in length-64 sequences;
- SiLU, RMSNorm, small output initialization, optimizer warmup, and AGC.

Actor-critic updates do not update the encoder, transformer, reward head, or
continuation head. World-model parameters change only during the world-model
part of each phase.

## Data Schedule

The initial replay contains four independent 80-step random segments per
environment. It is regenerated from the run seed in a separate temporary
environment adapter, so the online environments remain at their first reset.
This reproduces the previously cached bootstrap exactly without requiring an
external replay file.

After bootstrap, training is tightly interleaved: 64 vector steps, 1024
world-model updates, and 512 actor-critic updates per phase. The encoder remains
trainable during online world-model updates.

| Preset | Online phases | Train transitions | Held-out transitions | Train + held-out | WM updates | Policy updates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `jepa_100k` | 91 | 98,304 | 1,280 | 99,584 | 94,464 | 47,872 |
| `jepa_500k` | 481 | 497,664 | 1,280 | 498,944 | 493,824 | 247,552 |

Final evaluation interactions are reporting overhead and are tracked separately.

## Reproducibility

The maintained presets use isolated named RNG streams for initialization,
world-model updates, policy updates, replay sampling, online collection, and
validation. They also request deterministic accelerator reductions and highest
JAX matrix-multiplication precision.

Each run records:

- resolved arguments and dependency versions;
- initial and validation replay SHA-256 fingerprints;
- parameter and target-critic fingerprints after every phase;
- recent replay fingerprints every phase and full replay fingerprints at
  recovery checkpoints;
- exact training, validation, and evaluation transition counts.

All training metrics use `budget/train_env_steps` as the W&B x-axis. Episode
returns are logged at the real training transition where the episode finishes.

## Maintained Interfaces

- `world-marl-train-dmc-jepa`: train one or more runs;
- `write_dmc_vector_launcher.py`: generate `smoke`, `jepa_100k`, or
  `jepa_500k` launchers;
- `world-marl-eval-jepa-wm`: general held-out world-model evaluation.

The canonical implementation is intentionally small enough that changing a
scientific assumption requires an explicit code or preset change rather than
activating an old experimental flag.
