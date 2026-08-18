# DreaMARL Architecture

## Definition

DreaMARL is a decoder-free visual model-based reinforcement learning algorithm.
Its maintained implementation has two layers:

1. `agent.py` is the locked local learner. It contains the visual encoder,
   categorical latent state, causal Transformer world model, JEPA objectives,
   actor, critic, replay learning, and imagination.
2. `marl/core.py` preserves the explicit agent axis, applies the same local
   learner with parameters shared across agents, and preserves synchronized
   team trajectory identity without exposing peer tensors to local modules.

`A=1` is exactly the locked single-agent learner. B0 is a shared-independent
MARL baseline with strict decentralized execution for every `A>1`. B1 keeps
that execution graph and adds a training-only agent-axis JEPA objective.

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
shared across agents. Inactive agents are excluded from optimized losses,
normalization statistics, and SIGReg. Observed action masks are enforced during
collection; a supervised availability head supplies masks for imagination.

For `A=1`, local outputs, gradients, optimizer updates, and recurrent carry
match the locked local learner. For `A>1`, each actor, critic, and transition
uses only its focal observation/action history. Peer histories cannot affect a
focal policy distribution.

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

Replay sequences are evaluated in parallel under a strict causal and
64-transition sliding-window mask, including when the input cache is nonempty.
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
projections. Batch and time are pooled independently for each agent, inactive
samples are removed from the statistic, and the per-agent statistics are
averaged. This preserves the single-agent objective strength instead of
multiplying it by the team size.

All predictive terms use normalized cosine distance. The complete model loss
is:

```text
L_model =
    2.00 * L_posterior_JEPA
  + 2.00 * L_dynamics_JEPA
  + 1.00 * L_spatial_JEPA
  + 1.00 * L_agent_JEPA        # B1 only
  + 0.05 * L_SIGReg
  + 1.00 * L_action_mask
  + 1.00 * L_reward
  + 1.00 * L_continuation
  + 1.00 * L_dynamics_KL
  + 0.10 * L_representation_KL
```

Reward is predicted with a 255-bin symlog two-hot head and continuation with a
binary head. The shared heads produce one reward and continuation target per
agent from that agent's local state.

## B0 Multi-Agent Transition

The causal Transformer remains the authoritative local transition model. The
same parameters are applied independently to every folded agent row:

```text
u_t^i = Project_local(z_t^i, a_t^i)
h_t^i = CausalTransformer(u_t^i, history_t^i)
```

Replay training, online collection, replay-context reconstruction, open-loop
reports, and recursive imagination all call this same local transition.
Inactive agents preserve their recurrent state and are excluded from optimized
losses. No peer encoder, peer gate, communication state, team teacher, or
centralized critic is instantiated in B0.

The public `[B,T,A,...]` representation and grouped team imagination are
retained deliberately. Future B1+ modules can construct training-only targets
at that boundary without changing what the deployed B0 actor observes.

## B1 Agent-Axis JEPA

B1 is the first controlled addition to B0. At each eligible replay timestep,
25--50% of complete active agents are removed from the online branch, while at
least one active agent remains visible and one remains hidden. The online set
encoder maps the remaining local encoder embeddings into eight slots of width
256 using two cross-attention/FFN layers and four heads. A separate context
encoder summarizes only the visible agents' history-conditioned local world
states. A two-layer slot predictor combines both sources and predicts the
complete team representation produced by an EMA teacher that sees every active
agent:

```text
online content: T_online({E(o_t^j): visible j}) -> K content slots
online history: T_history({s_t^j: visible j})  -> K history slots
prediction:     P_team(content slots, history slots) -> K complete-team slots
target:         T_ema({E_ema(o_t^j): active j})      -> K complete-team slots
team loss:      mean_k cosine(prediction_k, stop_gradient(target_k))
```

`T_ema` has the same architecture as `T_online`; its weights follow the online
set encoder with EMA rate `0.01` and are excluded from the optimizer. It also
receives the existing EMA local encoder embeddings, and every target is
stop-gradient. Both set encoders have no agent-position embeddings and are
permutation invariant over homogeneous team members. Explicit active and
visible counts prevent the attention mean from erasing roster size. Complete
members first compete across slots and are then normalized within each slot,
as in Slot Attention. Learned queries determine that content partition but are
not carried additively into the returned slots. At the first layer they
multiplicatively gate attended member content, breaking initial slot symmetry
while producing exactly zero output in the absence of member content. This
closes the constant-query shortcut without initializing all slots identically.
The content gates use learned slot codes normalized competitively across slots
for every feature, with mean gate one; the codes have no additive path into the
teacher representation.

Three safeguards make the team target nontrivial. A shared content decoder maps
each predicted slot and each full-online source slot into the local EMA content
space. Before matching, the active-agent mean is removed from EMA content and
the slot mean is removed from decoded content. Balanced stop-gradient Sinkhorn
matching therefore identifies agent-relative distinctions rather than the
common scene component; it gives every slot equal mass and every active EMA
agent equal coverage, so the source of the EMA teacher cannot collapse to one
aggregate direction. A second soft assignment explicitly
requires every completely hidden agent to match at least one predicted slot.
Finally, the full online team slots receive a variance hinge and inter-slot
decorrelation penalty. Slot standard deviation, effective rank, matching
cosines, and assignment entropy are reported throughout training. The
same-time masked objective is:

```text
L_k0 = L_complete_team_slots
     + 1.00 * L_predicted_set_matching
     + 1.00 * L_source_set_matching
     + 1.00 * L_hidden_agent_coverage
     + 0.10 * L_slot_variance
     + 0.10 * L_slot_decorrelation
```

Set matching uses ten Sinkhorn iterations at temperature `0.02`. The transport
plan is stop-gradient; gradients update slot content through the matched cosine
cost without differentiating through the assignment itself.

The future branch keeps every local state paired with its own replay action
before any permutation-invariant pooling. Hidden current content remains zero,
but its observed replay action is retained as part of the training-only joint
action. An action conditioner and a separate set encoder produce action-aware
source slots. A two-layer transition predictor combines those slots with the
masked current-team prediction and targets the next complete EMA team state:

```text
action members: C(where(visible, E(o_t^j), 0), a_t^j, masks)
action slots:   T_action({action member_t^j: active j})
future slots:   F_team(prediction_t, action_slots_t)
future target:  stop_gradient(T_ema({E_ema(o_t+1^j): active j}))
```

Transitions crossing `is_first` and transitions without an active source or
target team are excluded. The future slots are supervised both by aligned
teacher-slot cosine and by balanced matching to all next-step EMA members. The
complete B1 objective is:

```text
L_agent_JEPA = 0.10 * L_k0
             + 1.00 * L_future_team_slots
             + 1.00 * L_future_set_matching
```

All B1 inputs from the locked local encoder/world state are stop-gradient. The
online team modules receive their full auxiliary gradient, but B1 cannot
reshape the established single-agent representation. B1 remains training-only
and does not yet add a centralized critic, a local team belief, or explicit
teammate-policy modelling. The actor, critic, imagination transition, and
online recurrent carry are the B0 path and never receive team slots or peer
tensors.

A frozen utility probe can evaluate a trained B1 checkpoint on retained replay
without updating any parameter. It compares the aligned joint-action future
loss against cross-batch action shuffling, within-team state/action ownership
shuffling, and a copy-current-state persistence baseline. Positive loss gaps
are required evidence that the future representation uses the correct joint
action rather than merely exploiting temporal persistence.

## Imagination And Control

At execution, the shared actor consumes only each agent's local world feature:

```text
a_t^i ~ pi(a | s_t^i)
 s_t+1^i ~ F(s_t^i, a_t^i)
```

The world-model attention cache is carried forward exactly in collection and
imagination. An intervention test perturbs peer histories while fixing the focal
observation history and requires its simulator state and action distribution to
remain exactly unchanged.

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
slow-value regularization, and replay value learning. Return normalization is
updated only by live imagined agents, and replay-value validity is aligned to
source states `0..T-2`.

## Optimization And Data

| Property | Value |
| --- | ---: |
| Batch size | 16 environment sequences |
| Sequence length | 64 |
| Replay context | 128 loss-excluded transitions |
| Replay sampling | uniform |
| Replay capacity | 5 million environment transitions |
| Training ratio | 256 |
| Learning rate | `4e-5` |
| Warmup | 1000 optimizer updates |
| Adaptive gradient clipping | `0.3` |
| Compute dtype | BF16 |

The replay prefix spans `context * layers = 64 * 2` transitions. It is scanned
without loss gradients to reconstruct every retained upper-layer KV entry;
the absolute Transformer position is stored with replay entries so RoPE state
is reconstructed exactly under fixed parameters.

The online encoder, causal dynamics, JEPA predictors, reward and continuation
heads, actor, and critic share one optimizer. The EMA target encoder and slow
critic are not optimizer members. All learned parameters are shared over the
agent axis. B0 parameter count is independent of `A`; B1 adds the same fixed
training-only team modules for every multi-agent team size, so its parameter
count is likewise independent of the specific `A > 1` value.

## Evaluation And Reporting

Training curves use completed environment episodes against counted environment
transitions. Multi-agent runs report both reward conventions explicitly:

- `per_agent_return_mean`: episode sum of the mean reward over agents;
- `team_return_sum`: episode sum of rewards over agents and time.

`score` remains a compatibility alias for `per_agent_return_mean`. Agent
transitions are tracked separately as `environment_steps * A`. Fixed evaluation
uses held-out workers, never enters replay, evaluates the latest policy, and
performs no checkpoint search. Inline evaluation preserves both the action RNG
counter and pending JAX policy synchronization. Generic `train_eval` and
`parallel*` runners are rejected for `A>1` because their reporting is not MARL
reward-aware; first-party `train` plus explicit curve evaluation is supported.

### Environment seed semantics

Each worker receives a derived seed. For Melting Pot, that value is supplied
when the underlying Lab2D substrate is constructed because Shimmy 2.0.1 ignores
its `reset(seed)` argument. This controls the Lab2D seed stream, but the pinned
Lab2D backend does not guarantee trajectory determinism: separately constructed
environments can return different observations under the same construction seed
and identical action sequence. Launch manifests describe this as
`construction_seed_controlled_not_trajectory_deterministic`; a recorded seed is
not a bitwise replay guarantee.

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
