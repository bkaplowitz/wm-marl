# DreaMARL Architecture

## Definition

DreaMARL is a decoder-free visual model-based reinforcement learning algorithm.
Its maintained implementation has two layers:

1. `agent.py` is the locked local learner. It contains the visual encoder,
   categorical latent state, causal Transformer world model, JEPA objectives,
   actor, critic, replay learning, and imagination.
2. `marl/core.py` preserves the explicit agent axis, applies the same local
   learner with parameters shared across agents, and conditions its single
   transition model on the synchronized peer latent-action set.

There is no architecture switch based on team size. The same algorithm runs
for every `A >= 1`; `A=1` is exactly the locked single-agent learner.

## Team Data Contract

Multi-agent replay preserves environment, time, and agent identity:

```text
image           [B, T, A, 64, 64, 3]
action          [B, T, A, ...]
reward          [B, T, A]
agent_present   [B, T, A]
agent_alive     [B, T, A]
action_mask     [B, T, A, U]
is_first        [B, T]
is_last         [B, T]
is_terminal     [B, T]
```

`marl/core.py` applies a lossless layout transformation around the local
learner:

```text
[B, T, A, ...] <-> [B * A, T, ...]
```

The same transformation is used for collection state, policy outputs, replay
state, training batches, and reports. Environment-level boundary fields remain
shared across agents. Inactive agents are excluded from optimized losses, and
the optional action mask prevents invalid discrete actions.

For `A=1`, the peer set is empty and its context is exactly zero: local outputs,
gradients, optimizer updates, and recurrent carry match the locked local
learner. For `A>1`, observations, critics, and actors remain local while the
world transition is explicitly conditioned on peer latent-action effects.

## Local Visual State

Each agent receives only its local `64 x 64 x 3` RGB observation. A single
convolutional encoder is shared across all agents.

| Property | Value |
| --- | ---: |
| Base depth | 64 |
| Stage multipliers | `[2, 3, 4, 4]` |
| Kernel | 5 |
| Activation | SiLU |
| Normalization | RMS |
| Encoder output width | 4096 |

The online encoder is optimized end to end. A structurally identical target
encoder is updated after each model update with source rate `0.01`, equivalent
to EMA decay `0.99`. Target representations are stop-gradient values.

The stochastic state contains 32 categorical variables with 64 classes. The
causal Transformer produces an 8192-wide deterministic state:

```text
x_t = concat(z_(t-1), a_(t-1))
h_t = CausalTransformer(x_1, ..., x_t)
q(z_t | h_t, E(o_t))
p(z_t | h_t)
s_t = concat(h_t, flatten(z_t))
```

Both posterior and prior use a 1% uniform mixture. Dynamics and representation
KL terms use one free nat.

### Causal Transformer

| Property | Value |
| --- | ---: |
| Model width | 512 |
| Layers | 2 |
| Attention heads | 8 |
| Feed-forward expansion | 4 |
| Context | 64 transitions |
| Deterministic output | 8192 |

Replay sequences are evaluated in parallel under a strict causal mask.
Collection and imagination use the same weights through a bounded recurrent
key-value cache. Tests cover causal isolation, reset behavior, and agreement
between parallel and recurrent execution.

## Decoder-Free World Modeling

No image decoder is constructed or optimized. Pixel observations enter the
online and EMA target encoders, while predictive targets and imagined
transitions remain in representation space.

The world-model objective has four representation terms.

### Posterior representation prediction

```text
P_post(s_t) -> stop_gradient(E_target(o_t))
```

### Action-conditioned dynamics prediction

```text
P_dyn(h_t) -> stop_gradient(E_target(o_t))
```

### Fixed-count masked spatial prediction

Exactly half of the encoder image grid is sampled without replacement. The
corresponding input patches are replaced with value `128`. The masked online
representation and current deterministic state predict target-encoder spatial
tokens at the hidden locations:

```text
P_spatial(h_t, E_online(mask(o_t)))
    -> stop_gradient(spatial(E_target(o_t)))
```

The maintained topology is `fixed_count` with mask ratio `0.5`.

### Anti-collapse regularization

SIGReg is applied to online encoder tokens with 17 knots and 256 random
projections using pooled aggregation.

All predictive terms use normalized cosine distance. The complete model loss
is:

```text
L_model =
    2.00 * L_posterior_JEPA
  + 2.00 * L_dynamics_JEPA
  + 1.00 * L_spatial_JEPA
  + 0.05 * L_SIGReg
  + 1.00 * L_reward
  + 1.00 * L_continuation
  + 1.00 * L_dynamics_KL
  + 0.10 * L_representation_KL
```

Reward is predicted with a 255-bin symlog two-hot head and continuation with a
binary head. The shared heads produce one reward and continuation target per
agent from that agent's local state.

## Multi-Agent Transition

The only causal Transformer remains the authoritative transition model. Before
each temporal step, a shared peer encoder maps every stopped-gradient
stochastic-state and action pair into model space. A permutation-invariant
masked mean then forms one context for each focal agent. Self tokens are
excluded:

```text
u_t^i = Project_local(z_t^i, a_t^i)
c_t^i = Mean({Project_peer(stopgrad(z_t^j, a_t^j)) | j != i})
h_t^i = CausalTransformer(u_t^i + tanh(g) * c_t^i, history_t^i)
```

The local transition projection is unchanged. The peer projection is separately
normalized, and its learned per-channel gate starts at zero and is bounded by
`tanh`. Thus training starts from the proven local transition and can adopt
peer information without changing the local input's scale or meaning. The
stop-gradient boundary prevents focal-agent losses from moving teammate
representations through the interaction edge, while the peer encoder, gate,
and Transformer learn normally. The resulting deterministic state drives the
posterior, prior, JEPA predictor, reward head, continuation head, actor, and
critic exactly as before. There is no second world model, centralized
observation encoder, agent identifier, or centralized critic. Inactive peers
are masked, and the empty peer set produces an exact zero context.

Replay training, online collection, replay-context reconstruction, open-loop
reports, and recursive imagination all call this same transition. Thus peer
actions are observed causes rather than policy-dependent hidden variables.

## Imagination And Control

Each decentralized actor samples from its own local latent state. The complete
team action is then assembled before the world advances:

```text
a_t^i ~ pi(a | s_t^i)
{s_t+1^i}_{i=1}^A ~ F({s_t^i, a_t^i}_{i=1}^A)
```

All actors, critics, reward heads, and continuation heads still consume only
their corresponding local predicted state. Joint information is confined to
the learned simulator during centralized training; execution remains
decentralized.

| Property | Value |
| --- | ---: |
| Imagination length | 15 |
| Discount horizon | 333 |
| Lambda | 0.95 |
| Entropy coefficient | `3e-4` |
| Actor | 3 x 1024 RMS/SiLU |
| Continuous policy | bounded Normal, std `[0.1, 1.0]` |
| Discrete policy | categorical, 1% uniform mixture |
| Critic | 3 x 1024, symlog two-hot |
| Slow critic source rate | `0.02` per update |

The actor uses the DreamerV3 score-function objective, percentile return-range
normalization, and entropy regularization. The critic uses lambda returns,
slow-value regularization, and replay value learning.

## Optimization And Data

| Property | Value |
| --- | ---: |
| Batch size | 16 environment sequences |
| Sequence length | 64 |
| Replay context | 1 transition |
| Replay sampling | uniform |
| Replay capacity | 5 million environment transitions |
| Training ratio | 256 |
| Learning rate | `4e-5` |
| Warmup | 1000 optimizer updates |
| Adaptive gradient clipping | `0.3` |
| Compute dtype | BF16 |

The online encoder, causal dynamics, JEPA predictors, reward and continuation
heads, actor, and critic share one optimizer. The EMA target encoder and slow
critic are not optimizer members. All learned parameters are shared over the
agent axis, so parameter count is independent of `A`.

## Evaluation And Reporting

Training curves use completed environment episodes against counted environment
transitions. Multi-agent runs report both reward conventions explicitly:

- `per_agent_return_mean`: episode sum of the mean reward over agents;
- `team_return_sum`: episode sum of rewards over agents and time.

`score` remains a compatibility alias for `per_agent_return_mean`. Agent
transitions are tracked separately as `environment_steps * A`. Fixed evaluation
uses held-out workers, never enters replay, evaluates the latest policy, and
performs no checkpoint search.

## Source Layout

```text
src/dreamarl/
  agent.py                 locked local world model and actor-critic assembly
  config.py                launch specification and reproducibility manifest
  configs.yaml             maintained architecture and task profiles
  contracts.py             machine-readable invariants
  envs/                    visual DMC and Melting Pot adapters
  marl/
    core.py                agent-axis runtime and synchronous imagination
    axes.py                reversible team/local tensor layouts
    spaces.py              team/local observation and action spaces
  models/                  visual encoder and categorical latent components
  training/                world-model, actor-critic, replay, and reporting
  world_model/             causal Transformer backend
  ablations/               reproducible non-canonical scientific controls
```

The empirical single-agent lock-in is recorded in `VISUAL_LOCKIN.md`.
