# DreaMARL Architecture

## Purpose

DreaMARL is a first-party, reconstruction-regularized JEPA world-model agent
for decentralized multi-agent control. Each actor observes only its own local
history. During centralized training, the world transition additionally sees
the synchronized states and realized actions of the other controlled agents.

The maintained implementation lives in `world_marl.dreamarl`. The pinned
Dreamer-CDP checkout supplies only Embodied infrastructure. A frozen M3 source
snapshot remains as the single-agent parity oracle and is never imported by
the active learner.

## Tensor And Replay Contract

Policy tensors retain an explicit agent dimension:

```text
[environment, agent, ...]
```

Replay retains both time and agent dimensions:

```text
[batch, time, agent, ...]
```

Every transition stores local observations, local actions, and per-agent
rewards. Episode boundaries remain joint:

```text
observation  [B, T, A, ...]
action       [B, T, A, ...]
reward       [B, T, A]
is_first     [B, T]
is_last      [B, T]
is_terminal  [B, T]
```

`is_last` ends a replay sequence. Only `is_terminal` disables value
bootstrapping. Raw observations remain the source of truth; stored recurrent
entries are replay-context caches rather than authoritative latent targets.

Melting Pot training preserves each agent's reward. Benchmark reporting sums
the mean reward across agents at every environment step, yielding mean
per-agent episode return. One environment step counts as one joint transition,
not one transition per agent.

## Local Predictive State

Each agent encodes its own `64 x 64 x 3` image with the M3 convolutional
encoder. The encoder produces a 4,096-dimensional pre-bottleneck feature. The
decoder is retained for visual prediction reports and representation
regularization; decoder gradients do not update the representation.

The local recurrent belief is:

```text
b_t^i = [h_t^i, flatten(s_t^i)]
```

- deterministic state: 8,192 dimensions;
- stochastic state: 32 categorical variables with 64 classes;
- causal Transformer width: 512;
- Transformer layers: 4;
- attention heads: 8;
- recurrent KV context: 64 transitions;
- RMS normalization and SiLU activations.

The local transition consumes the previous stochastic state and the agent's
own action. The posterior also consumes the current local encoder output. Its
JEPA predictor regresses the stopped-gradient encoder target with cosine
distance. The categorical prior/posterior and KL objectives are unchanged from
the verified M3 reduction.

## Structured Local Memory

Each agent also maintains four persistent memory tokens of width 256. Four
learned queries pool the existing encoder features into local memory targets.
One shared-parameter block updates memory using:

```text
m_t^i = U(m_{t-1}^i, encoder_tokens(o_t^i), a_{t-1}^i)
```

The block contains local self-attention, cross-attention to current local
observation tokens, previous-action conditioning, and a two-times feed-forward
expansion. It never receives another agent's observation, state, action,
reward, or identity.

Real observations produce stopped-gradient memory targets. Imagination uses a
local memory prior:

```text
memory_prior_t  = P(m_{t-1}, b_{t-1}, a_{t-1})
memory_target_t = stop_gradient(Q(encoder_tokens(o_t)))
L_memory        = cosine_distance(memory_prior_t, memory_target_t)
```

The actor, critic, reward head, and continuation head consume the established
belief plus the structured memory control representation. The same interface
is used for every agent count.

## Joint Interaction Transition

The local transition remains the complete fallback model. DreaMARL adds only a
peer-caused correction:

```text
local_next_i = F_local(z_t^i, a_t^i)
delta_i      = F_interaction(z_t^{-i}, a_t^{-i}; z_t^i, a_t^i)
next_i       = local_next_i + delta_i
```

For every focal agent, its local state-action token becomes an attention query.
Only other valid agents become keys and values. Attention parameters are
shared, and there are no numerical agent identifiers, so the transition is
permutation equivariant for homogeneous agents.

The interaction block has:

- token width: 256;
- attention heads: 4;
- one leave-one-out relational attention block;
- feed-forward expansion: 2;
- explicit validity masking;
- corrections for the deterministic belief and imagined memory prior.

The output projection is initialized to exactly zero. Initial outputs are
therefore numerically identical to the local-memory model. The output
projection learns first; after it moves away from zero, gradients reach the
attention and token projections. With one agent or no valid peer, the masked
context and residual are exactly zero without an all-masked softmax.

The branch cannot become a larger single-agent transition: every output term
depends on leave-one-out peer context. Its 5,772,032 parameters do not grow with
agent count. On Externality, the complete candidate has 178,030,092 parameters,
compared with 172,258,060 for the local-memory reference.

## Synchronous Imagination

Each imagined timestep is atomic:

```text
1. Every local actor samples its action from z_t^i.
2. Actions are assembled into a_t^{1:A}.
3. One grouped transition predicts every next local state.
4. Agent i receives only z_{t+1}^i.
```

No agent is advanced before another agent's current action is known. The actor
and critic never receive peer states or interaction attention outputs directly.
Centralized information affects control only through predictions of the focal
agent's own future local state.

## Actor, Critic, And Optimization

- actor: 3 hidden layers of width 1,024;
- discrete policy: categorical with 1% mixture;
- continuous policy: bounded Normal with standard deviation in `[0.1, 1.0]`;
- critic: 3 hidden layers of width 1,024 with 255-bin symlog two-hot output;
- imagination length: 15;
- return horizon: 333;
- lambda: 0.95;
- entropy coefficient: `3e-4`;
- slow-value update rate: `0.02`;
- replay-value loss enabled;
- encoder learning rate: `6e-6`;
- dynamics learning rate: `4e-4`;
- remaining-module learning rate: `4e-5`;
- adaptive gradient clipping: `0.3`;
- BF16 compute.

Melting Pot uses batches of 16 joint sequences, sequence length 64, uniform
online replay, capacity 5 million joint transitions, and train ratio 256. No
task-specific reward shaping, checkpoint selection, recent-replay rule, or
centralized critic is part of the algorithm.

## Evaluation And Observability

Training reports mean per-agent episode return. Fixed evaluation uses the
latest checkpoint, deterministic policy modes, an explicit evaluation seed,
and an exact number of complete episodes. It writes every episode return and
every per-agent return to `evaluation_summary.json`.

World-model reports include, at recursive horizons 1, 2, 4, and 8:

- JEPA latent cosine error;
- per-agent reward mean absolute error;
- continuation Brier error.

Training also reports deterministic-belief and memory residual RMS. These
metrics establish whether a control improvement accompanies a real change in
joint prediction rather than only additional parameter count.

## Correctness Contract

The maintained tests require:

1. exact zero-residual containment of the local model;
2. exact one-agent reduction and finite all-masked behavior;
3. permutation equivariance;
4. sensitivity to peer actions;
5. unchanged local parameter initialization;
6. grouped synchronous imagination;
7. explicit per-agent reward preservation;
8. unchanged optimizer, replay, actor, critic, and training schedules across
   agent counts.
