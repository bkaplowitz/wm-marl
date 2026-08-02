# DreaMARL Architecture

## Scope

DreaMARL is a first-party, reconstruction-regularized, JEPA world-model agent
with shared parameters and decentralized execution. The same computation is
used for one or many agents:

```text
policy tensors: [batch, agent, ...]
replay tensors: [batch, time, agent, ...]
```

Changing the number of agents changes tensor geometry only. It does not select
a different model, optimizer, objective, replay schedule, or imagination
regime. An agent consumes only its own observation, previous action, and
recurrent state. Neural parameters are shared across agents.

The executable learner is `world_marl.dreamarl`. The pinned Dreamer-CDP tree
provides Embodied infrastructure, while the agent, dynamics, training loop,
environment adapter, configuration, and launcher are maintained here. A frozen
copy of M3 is retained only as the single-agent numerical parity oracle and is
not imported by the active learner.

## Observation Representation

Each agent observes a `64 x 64 x 3` image. The existing M3 convolutional encoder
produces a 4,096-dimensional spatial feature before belief compression. The
convolutional decoder is retained as a representation regularizer and for
visual reports; decoder gradients do not update the representation
(`dec_grad: false`). No second image encoder is introduced.

## Core Belief

The established local belief is retained unchanged:

```text
b_t^i = [h_t^i, flatten(s_t^i)]
```

- deterministic state `h`: 8,192 dimensions;
- stochastic state `s`: 32 categorical variables with 64 classes;
- categorical uniform mixture: 1%;
- causal Transformer width: 512;
- Transformer layers: 4;
- attention heads: 8;
- feed-forward expansion: 4;
- recurrent KV context: 64 transitions;
- RMS normalization and SiLU activations.

The local transition consumes the previous stochastic state and local action.
The posterior additionally consumes the current local encoder output. The
prior, posterior KL losses, and JEPA predictor are the established M3 path.
The JEPA target is the stopped-gradient current encoder output and its cosine
prediction loss has weight 500.

## Local Memory Sidecar

The production candidate augments the belief with four persistent local memory
tokens without replacing or modifying the core transition. The 4,096 encoder
features are interpreted as 16 spatial tokens of width 256. Four learned local
queries pool these features into memory targets.

For every agent, one shared-parameter Transformer block updates:

```text
m_t^i = U(m_{t-1}^i, encoder_tokens(o_t^i), a_{t-1}^i)
```

The block contains:

- four memory tokens of width 256;
- four attention heads;
- memory-token self-attention;
- cross-attention to the current local observation tokens;
- local previous-action conditioning;
- a two-times feed-forward expansion.

Memory is reset by `is_first`. It never receives another agent's observation,
action, belief, reward, or identity. Thus execution remains decentralized and
memory computation scales linearly with agent count.

## Memory Imagination

Real replay supplies a posterior memory and a stopped-gradient target from the
current observation tokens. The imagination prior has no access to that
observation. It predicts the next memory from the previous memory, core belief,
and local action:

```text
memory_prior_t = P(m_{t-1}, b_{t-1}, a_{t-1})
memory_target_t = stop_gradient(Q(encoder_tokens(o_t)))
L_memory = cosine_distance(memory_prior_t, memory_target_t)
```

`L_memory` has fixed weight 100. During imagined rollouts, predicted memory is
fed recurrently into the next memory transition. Target and prior RMS, standard
deviation, cosine error, and both learned gates are logged to detect collapse
or a sidecar that remains unused.

## Safe Control Interface

Actor, critic, reward, and continuation heads consume:

```text
u_t = b_t + g * P_control(m_t)
```

`P_control` first compresses the four memory tokens to width 256 and then
projects to the belief width. The scalar `g` is initialized to exactly zero.
Consequently, enabling the module cannot alter policy or value inputs at
initialization. Training can open the gate only when the additional features
provide useful gradients. The sidecar uses an isolated initialization stream,
so adding its parameters does not perturb the core model's initialization.

## Actor-Critic and Imagination

The control stack is shared across agents:

- actor: 3 hidden layers of width 1,024;
- continuous actions: bounded Normal, standard deviation in `[0.1, 1.0]`,
  with 1% mixture;
- discrete actions: categorical with 1% mixture;
- critic: 3 hidden layers of width 1,024 and a 255-bin symlog two-hot output;
- imagination length: 15;
- return horizon: 333;
- lambda: 0.95;
- entropy coefficient: `3e-4`;
- slow-value update rate: `0.02` per update;
- replay-value loss enabled.

Every agent samples from the same local policy using its own recurrent state.
Imagined trajectories preserve complete agent groups for replay starts and
loss reduction, but there is no centralized attention or privileged policy
input.

## Training Contract

- batch size: 16 joint sequences;
- sequence length: 64 plus one stored replay-context transition;
- replay capacity: 5 million joint environment transitions;
- uniform online replay;
- MeltingPot train ratio: 256;
- encoder learning rate: `6e-6`;
- dynamics learning rate: `4e-4`;
- other-module learning rate: `4e-5`;
- adaptive gradient clipping: `0.3`;
- BF16 compute.

One environment step is one joint transition regardless of agent count. Replay
stores local images, actions, core recurrent entries, memory entries, and
decoder entries with an explicit agent axis. `is_first` resets recurrent state,
`is_last` ends a sequence, and `is_terminal` alone disables continuation
bootstrapping.

The established model contains approximately 164.9 million trainable
parameters. The four-token sidecar adds approximately 7.37 million parameters,
or about 4.5%, independent of agent count. Exact totals vary slightly with the
environment action space.

## Acceptance Contract

The sidecar is retained only if it satisfies both gates:

1. It improves held-out one-step and recurrent memory prediction without
   representation collapse.
2. The prediction improvement produces better end-to-end control on
   Externality, then transfers to Coop Mining without regressing Pure
   Coordination.

Token-count sweeps, centralized observations, joint-action inputs, auxiliary
event objectives, and task-specific rules are outside this architecture.
