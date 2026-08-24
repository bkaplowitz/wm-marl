# DreaMARL Architecture

## Scope and terminology

DreaMARL is a decoder-free model-based reinforcement learning algorithm. Its
authoritative state is a categorical stochastic latent paired with a causal
Transformer state. Predictive losses compare learned embeddings with
stop-gradient EMA targets; the model does not reconstruct observations.

The implementation exposes three related configurations:

1. **DreaMARL** applies the local world model, actor, and critic to one agent.
2. **Independent DreaMARL** shares those parameters across an explicit agent
   axis but keeps learning and execution observation-local.
3. **DreaMARL-CTDE** preserves the local actor and adds a training-only joint
   JEPA simulator plus a centralized attention critic.

DreaMARL-CTDE **one-step** is manifest version `1.1`. DreaMARL-CTDE
**two-step** is version `2`; it retains every one-step component and adds a
bounded self-fed predictive objective. Historical experiment-stage labels are
retained only for old manifests and checkpoint compatibility.

## Local executable model

Every agent has the same executable information path:

```text
local observation history_i
          |
          v
shared observation encoder
          |
          v
local categorical posterior + local causal Transformer_i
          |
          v
shared actor -> local action_i
```

Parameters are shared across agents, but recurrent state is not. A focal
actor's distribution is a function only of its own observation/action history.
There is no agent identifier, peer residual, communication token, joint state,
or centralized value input in the online policy.

### Observation representation

Visual observations use the compact DreamerV3-style convolutional encoder:

| Property | Value |
| --- | ---: |
| Input | 64 x 64 RGB |
| Base depth | 64 |
| Stage multipliers | `[2, 3, 4, 4]` |
| Kernel | 5 |
| Activation | SiLU |
| Normalization | RMS |
| Visual embedding width | 4096 |

Vector observations, including SMAC local observations, use the same local
encoder interface without constructing a visual decoder.

An EMA copy of the encoder supplies predictive targets. Its source rate is
`0.01`, corresponding to decay `0.99`, and target values are always
stop-gradient.

### Local latent dynamics

The stochastic state has 32 categorical variables with 64 classes. Both prior
and posterior use a 1% uniform mixture. The deterministic state is 8192 wide,
and the flattened local feature used by policy and value has width 10240.

```text
x_t^i = concat(z_(t-1)^i, a_(t-1)^i)
d_t^i = T_local(x_1^i, ..., x_t^i)
z_t^i ~ q(z | d_t^i, E(o_t^i))
s_t^i = concat(d_t^i, flatten(z_t^i))
```

| Local Transformer property | Value |
| --- | ---: |
| Model width | 512 |
| Layers | 2 |
| Attention heads | 8 |
| Feed-forward expansion | 4 |
| Context | 64 transitions |
| Deterministic output | 8192 |

Replay evaluation uses strict causal attention and the same 64-transition
sliding context as recurrent collection and imagination. A 128-transition,
loss-excluded replay prefix reconstructs the two-layer key/value cache and
absolute rotary position before optimized sequence positions are evaluated.

### Local JEPA objective

The local world model has three predictive representation losses:

```text
posterior: P_post(s_t) -> stop_gradient(E_ema(o_t))
dynamics:  P_dyn(d_t)  -> stop_gradient(E_ema(o_t))
spatial:   P_mask(d_t, E(mask(o_t)))
                          -> stop_gradient(spatial(E_ema(o_t)))
```

The spatial objective hides exactly half of the visual encoder grid without
replacement. SIGReg uses 17 knots and 256 random projections to prevent
representation collapse. Its statistics are computed per agent and averaged,
so adding agents does not multiply the regularizer strength. Spatial masking
is disabled for non-visual SMAC observations.

Local reward uses a 255-bin symlog two-hot head; continuation and discrete
action availability use binary heads. The maintained local loss scales are:

| Loss | Scale |
| --- | ---: |
| Posterior JEPA | 2.0 |
| Dynamics JEPA | 2.0 |
| Spatial JEPA | 1.0 |
| SIGReg | 0.05 |
| Reward | 1.0 |
| Continuation | 1.0 |
| Action availability | 1.0 |
| Dynamics KL | 1.0 |
| Representation KL | 0.1 |

## Explicit agent-axis contract

Multi-agent replay retains environment, time, and agent identity:

```text
observation          [B, T, A, ...]
action               [B, T, A, ...]
reward               [B, T, A]
agent_present        [B, T, A]
agent_alive          [B, T, A]
controllable_alive   [B, T, A]
action_mask          [B, T, A, U]
is_first             [B, T]
is_last              [B, T]
is_terminal          [B, T]
```

The local learner is wrapped by a reversible transformation:

```text
[B, T, A, ...] <-> [B * A, T, ...]
```

`agent_present` describes the fixed roster slot, while
`controllable_alive` describes whether the policy may choose a non-no-op action.
This distinction is important in SMAC: dead units remain present, expose only
the legal no-op action, and remain valid liveness targets. Inactive or absent
agents are excluded from policy losses, normalization, and representation
statistics as appropriate.

At `A=1`, folding is an identity in behavior: local outputs, losses, recurrent
state, gradients, and optimizer updates match the single-agent learner.

## Independent DreaMARL

Independent DreaMARL advances each folded local state with its own action:

```text
s_(t+1)^i ~ F_local(s_t^i, a_t^i)
a_(t+1)^i ~ pi(a | s_(t+1)^i)
```

Team starts stay synchronized for accounting and evaluation, but neither the
local transition nor critic consumes peer tensors. It is the controlled
multi-agent extension of the single-agent algorithm and the architectural
control for DreaMARL-CTDE.

## DreaMARL-CTDE

DreaMARL-CTDE changes centralized training, not decentralized execution. It
adds:

- a joint action-conditioned JEPA simulator;
- joint reward, continuation, action-availability, and liveness heads;
- a centralized attention critic over synchronized local posterior states;
- a separate joint-world optimizer group.

The actor and executable local state are unchanged.

### Policy-churn regularization

The optional `ctde-pcr` profile leaves the joint simulator, critic, replay
objective, and executable policy inputs unchanged. Each actor update also
samples an independent replay batch, reconstructs its local states after the
standard burn-in, and minimizes

```text
KL(stop_gradient(pi_previous) || pi_current)
```

under the same stopped environment action mask. `pi_previous` is an exact
one-optimizer-update-delayed copy of the actor. The KL coefficient is scaled to
2% of the current score-function objective magnitude. Reference states,
world-model parameters, the joint simulator, and the critic receive no gradient
from this term.

### Joint JEPA simulator

For each synchronized team state and factual joint action, the joint simulator
receives one token per roster slot:

```text
x_t^i = [stop_gradient(s_t^i), a_t^i,
         agent_present_t^i, controllable_alive_t^i]
```

There are no fixed agent IDs or agent-position embeddings. Two
permutation-equivariant agent-attention layers mix the current team; a causal
temporal Transformer then carries joint history.

| Joint simulator property | Value |
| --- | ---: |
| Token width | 256 |
| Agent-attention layers | 2 |
| Agent-attention heads | 4 |
| Temporal layers | 4 |
| Temporal heads | 4 |
| Temporal context | 16 transitions |
| Feed-forward expansion | 4 |
| Dropout | 0.1 |

Its main target is the next EMA local observation embedding for every agent:

```text
J(S_t, a_t) -> {predicted_e_(t+1)^1, ..., predicted_e_(t+1)^A}
target_i = stop_gradient(E_ema(o_(t+1)^i))
```

The prediction is also aligned with the stopped online embedding using a
smooth-L1 interface loss. This matters because the existing local posterior
consumes an online-encoder-shaped observation embedding during real execution.
The model therefore predicts what each focal agent will locally observe, not a
privileged joint actor feature.

The joint hidden state additionally predicts next reward, continuation, legal
action mask, and controllable liveness. Factual replay actions are aligned as
source action `a_t` to destination observation and signals at `t+1`.

### Centralized imagination

At every imagined step:

1. Each actor samples independently from its executable local state.
2. Actions are regrouped into the synchronized joint action.
3. The existing local Transformer advances each focal temporal proposal using
   only that focal state and action.
4. The joint JEPA predicts each agent's next local observation embedding.
5. The existing local posterior completes each next local state from its local
   temporal proposal and predicted local embedding.
6. The joint heads provide imagined reward, continuation, action availability,
   and monotonic liveness.

```text
s_t^i -> pi -> a_t^i -------------------------------+
  |                                                   |
  +-> T_local(s_t^i, a_t^i) -> d_(t+1)^i             |
                                                      v
{s_t^j, a_t^j}_j ---------------------------> joint JEPA
                                                      |
                                        predicted e_(t+1)^i
                                                      |
d_(t+1)^i --------------------------------------------+
                                                      v
                                           local posterior
                                                      |
                                                      v
                                               s_(t+1)^i
```

The posterior interface is shared with execution. Centralized information is
used only to model the local consequence of the joint action during training;
the deployed policy is never given the joint hidden state.

### Centralized attention critic

The critic receives the synchronized, stopped local posterior states before
the sampled joint action. A two-layer, four-head, 256-wide agent-attention
module produces one permutation-equivariant value feature per agent, followed
by a two-layer 256-wide symlog two-hot value head.

The critic has no action-conditioned central input, preventing direct action
leakage into score-function advantages. Critic gradients stop at local world
states. The actor remains parameter-shared and observation-local.

### Optimizer ownership

DreaMARL-CTDE uses four disjoint parameter groups, each at learning rate
`4e-5`:

1. local world model;
2. joint world model;
3. actor;
4. centralized critic.

EMA target encoders and slow value networks are not optimizer members. Joint
losses cannot update the local encoder, local dynamics, actor, or critic through
their stopped inputs. Actor and critic training use the existing score-function
objective, lambda returns, percentile return-range normalization, entropy
regularization, and slow-value targets.

## One-step objective

One-step DreaMARL-CTDE trains the factual transition for every valid replay
source. Its joint losses are:

| Joint loss | Scale |
| --- | ---: |
| EMA embedding cosine | 2.0 |
| Online-interface smooth L1 | 1.0 |
| Reward | 1.0 |
| Continuation | 1.0 |
| Action availability | 1.0 |
| Controllable liveness | 1.0 |

The replay prefix warms both local and joint causal caches without loss
gradients. The joint cache is rebuilt from current parameters, so a replay
sequence does not mix stale online cache values with updated weights.

## Two-step self-fed objective

Two-step DreaMARL-CTDE preserves the complete one-step objective and adds one
bounded test of self-fed prediction.

For each learner batch, at most 128 uniformly sampled valid anchors are chosen
from paths that remain in one episode. From source `t`:

1. The factual one-step prediction supplies `predicted_e_(t+1)`.
2. The frozen local advance and deterministic posterior interface
   (`sample=False`) construct the predicted local state at `t+1`.
3. The complete first-step local carry and joint Transformer carry are detached.
4. The factual replay action at `t+1` drives the second joint step.
5. The resulting prediction is supervised against EMA embedding, stopped online
   interface, reward, continuation, action mask, and liveness at `t+2`.

The liveness input to the second step is the detached, monotonic first-step
prediction; fixed-roster presence remains factual. Death at a destination is a
target, not a reason to discard the source transition.

Losses are scattered onto the full source-aligned replay grid so the learner's
ordinary validity masking and normalization remain correct. Gradients update
only the last joint prediction step. Posterior KL between predicted and factual
interfaces is reported diagnostically with both paths stopped; it is not an
optimized loss and cannot update the local posterior.

Setting `rollout_steps=1` does not instantiate the anchor sampler, consume its
randomness, add multistep metrics, or alter the one-step computation.

## Execution boundary

Collection and evaluation synchronize only:

- the local observation encoder;
- local causal dynamics and posterior;
- the shared actor.

They do not synchronize or call:

- the joint JEPA simulator;
- joint reward/continuation/mask/liveness heads;
- the centralized critic.

An execution-boundary intervention holds a focal observation/action history
fixed, perturbs peer histories, and requires the focal online policy
distribution to remain unchanged. This is the operational decentralized-
execution contract.

## Evaluation and reporting

Training curves use completed environment episodes against counted environment
transitions. Fixed evaluation uses held-out workers, restores the latest
complete checkpoint, performs no checkpoint selection, and does not enter
experience into replay.

SMAC uses a fixed roster, legal-action masks, shared benchmark reward, and
continuing episodes. Reports include:

- deterministic fixed-evaluation win rate and wins;
- enemy deaths and survivors;
- ally deaths and survival;
- timeout frequency;
- legacy SMAC reward;
- corrected damage, death, and combat-outcome diagnostics;
- one-step and, when enabled, two-step predictive metrics.

Predictive metrics verify that the training objective is active. Algorithmic
decisions are based on control outcomes, especially sustained win rate, rather
than representation cosine alone.

## Source map

```text
src/dreamarl/
  agent.py                 local learner and optimizer ownership
  config.py                launch specification and manifests
  configs.yaml             maintained model and environment profiles
  contracts.py             machine-readable execution invariants
  envs/
    smac.py                SMAC-v1 agent-axis adapter and diagnostics
  marl/
    axes.py                reversible local/team tensor layouts
    core.py                independent and CTDE integration
  models/
    ctde.py                joint JEPA and centralized attention critic
  training/
    ctde.py                bounded two-step anchor and loss utilities
    learner.py             local world model and behavior learning
  world_model/
    transformer.py         causal local Transformer
```

The single-agent empirical lock is recorded in
[the single-agent results](results/single_agent.md). Source attribution and
external revision boundaries are recorded in [provenance.md](provenance.md).
