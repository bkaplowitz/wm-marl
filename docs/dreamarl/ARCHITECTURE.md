# DreaMARL Architecture

## Purpose

DreaMARL is a reconstruction-free stochastic Transformer world model for
cooperative multi-agent reinforcement learning. It learns a single coherent
latent world, advances every agent from the same joint state and joint action,
and trains shared decentralized policies with centralized value learning.

The maintained implementation is first-party code under
`world_marl.dreamarl`. Pinned external repositories are evaluation baselines;
they are not runtime dependencies of DreaMARL.

## Non-Negotiable Design Rules

1. All time, environment, and agent axes remain explicit.
2. Environment resets, true terminals, and agent lifecycles have separate
   representations.
3. The actor observes only the focal agent's latent belief. The critic may
   observe all active agents' beliefs.
4. Every imagined transition consumes one complete joint action and advances
   all agents together.
5. Pixel reconstruction is not a training objective. Future observations are
   supervised through stopped target-encoder embeddings.
6. Collection, sequence learning, and imagination use JAX transformations;
   Python loops are orchestration only.
7. CoinGame is a contract and control gate, not a special case in the model.

## Trajectory Contract

Replay is transition-centric so terminal observations are never replaced by
auto-reset observations:

```text
observations        [T, env, agent, ...]
next_observations   [T, env, agent, ...]
actions        [T, env, agent, ...]
rewards        [T, env, agent]
team_rewards   [T, env]
agent_alive         [T, env, agent]
next_agent_alive    [T, env, agent]
action_mask         [T, env, agent, action]  # discrete actions, optional
next_action_mask    [T, env, agent, action]  # discrete actions, optional
is_first       [T, env]
is_last        [T, env]
is_terminal    [T, env]
valid          [T, env]
```

`is_last` cuts temporal state before the next replay record. `is_terminal`
alone disables Bellman bootstrapping. `agent_alive` masks inactive agents
without ending the joint environment. Adapters must preserve the real
successor in `next_observations` and put any auto-reset observation into the
next record's `observations` with `is_first = True`.

Replay samples are `[time, batch, agent, ...]` views of this contract. The
first state of every sampled window is an artificial temporal cut, not an
environment terminal.

## World Model

For each agent `i` at state `t`:

```text
e_t^i = Encoder(o_t^i)
h_t^i = LocalTransformer(history_<t^i)
p(z_t^i | h_t^i)                         prior
q(z_t^i | h_t^i, e_t^i)                  posterior
b_t^i = concat(h_t^i, z_t^i)             local belief
```

The encoder and local temporal Transformer share parameters across agents.
The stochastic state is a set of categorical variables trained with balanced
prior/posterior KL and free bits.

One masked cross-agent block receives every active local belief and action:

```text
c_t^{1:A} = CrossAgent({b_t^i, a_t^i}_{i=1}^A, alive_t)
```

Because each action-tagged token attends to every other active token, each
transition is conditioned on the complete joint action. The resulting context
is appended to each local temporal history, producing the next temporal state,
stochastic prior, reward, continuation, and embedding prediction.

The representation target is:

```text
target_t+1^i = stop_gradient(TargetEncoder(o_t+1^i))
```

The target encoder is an exponential moving average of the online encoder.
There is no observation decoder in the learning path.

Lifecycle and legal-action heads receive the masked local context, pooled
joint context, and a slot identity. Inactive agents cannot affect active-agent
dynamics, but the model can still predict a later respawn from the joint
state. Agent death is therefore not hard-coded as irreversible.

The world-model objective is:

```text
L_world =
    L_JEPA
  + beta_dyn * L_prior
  + beta_rep * L_posterior
  + beta_reward * L_team_reward
  + beta_agent_reward * L_agent_reward
  + beta_continue * L_continue
  + beta_alive * L_agent_alive
  + beta_action_mask * L_action_mask
```

Losses are masked by transition validity, lifecycle, and reset boundaries.
Reported loss terms are normalized independently; adding agents or ensemble
members must not silently multiply optimizer scale.

Team reward is the mean over active-agent rewards, which keeps its scale
independent of team size. Team reward, per-agent reward, and centralized value
use a shared 255-bin symlog two-hot representation over [-20, 20]; raw
task-scale returns are never clipped before this transform.

## Actor-Critic and Imagination

The shared actor maps one local belief and local action mask to one agent's
action distribution:

```text
pi(a_t^i | b_t^i)
```

The actor never receives another agent's private observation or belief. The
centralized critic receives the masked set of all beliefs and predicts team
return. Parameter sharing supports different active-agent counts while an
optional learned identity embedding handles heterogeneous roles.

Imagined rollouts begin from posterior replay states. At every imagined step:

1. all active actors produce one joint action;
2. the cross-agent transition consumes the complete joint action;
3. every local Transformer cache advances exactly once;
4. all next stochastic beliefs are sampled from their priors;
5. centralized reward, continuation, and value predictions form lambda
   returns;
6. actor and critic updates use the same coherent imagined world.

No focal-agent rollout is permitted to evolve while teammate states remain
fixed or are independently resampled.

Actor advantages are normalized by a checkpointed exponential moving average
of the 5th-to-95th percentile lambda-return range. The critic still learns the
unclipped task-scale return through its symlog two-hot distribution.

## Maintained Default Configuration

The model defaults are task-independent. Environment adapters supply only the
number of agent slots and discrete action count.

| Component | Maintained setting |
| --- | --- |
| Shared vector encoder | 2 x 256 MLP, SiLU, RMSNorm |
| Shared visual encoder | 4 stride-2 CNN blocks, depth 32/64/128/256 |
| Embedding width | 128 |
| Local temporal model | 3 causal Transformer layers |
| Temporal width / heads / context | 256 / 4 / 64 |
| Stochastic state | 16 categorical variables x 16 classes |
| Cross-agent model | 1 masked attention layer, width 256, 4 heads |
| Shared actor | 2 x 256 MLP, categorical policy |
| Centralized critic | masked agent attention plus 256-wide value head |
| Imagination | horizon 15, discount 0.99, lambda 0.95 |
| Replay | 100,000 joint transitions, sequences 64, batch 16 |
| World / actor / critic LR | 1e-4 / 3e-5 / 1e-4 |
| Encoder / critic EMA decay | 0.99 / 0.98 |
| Entropy coefficient | 3e-4 |
| Global gradient clipping | 100 for world model and actor-critic |

For two-agent CoinGame with five actions and 36-dimensional local vector
observations, this resolves to 5,130,385 world-model parameters, 264,965 actor
parameters, and 1,315,839 critic parameters: 6,711,189 trainable parameters in
total. The run manifest recomputes these counts from initialized parameter
trees because visual encoder size depends on image geometry.

The loss coefficients are JEPA 1.0, dynamics KL 0.5, representation KL 0.1,
team reward 1.0, per-agent reward 0.25, continuation 1.0, lifecycle 0.5, and
legal-action prediction 0.25. KL free nats is 1.0 and categorical unimix is
0.01.

## Training Protocol

The maintained CoinGame control gate uses 64 parallel environments, 4,096
initial uniformly random transitions, 64 initial learner updates, then
alternates 1,024 newly collected transitions with 16 prefetched compiled
model/actor/critic update transactions. These are protocol settings rather
than environment-specific model changes.

Evaluation uses deterministic actions from the latest policy. It never
selects or restores a better checkpoint. Evaluation interactions are recorded
separately and never counted in the training budget. A checkpoint contains
all optimizer states, target networks, return-normalization state, PRNG state,
environment state, temporal caches, replay contents, replay cursor, and replay
sampler RNG.

## Computational Contract

- Environments and agents are batched with `vmap` or equivalent fused axes.
- Time, recurrent collection, and imagination use `lax.scan`.
- Recurrent temporal inference uses a bounded KV cache shared with the
  parallel sequence path.
- Shapes are static within a compiled run; inactive agents use masks.
- Replay transfer is prefetched and contains contiguous sequence windows.
- The run manifest records parameter count, learner and environment
  throughput, and imagined-to-real ratio.

## Required Gates

The maintained learner cannot be promoted unless all of these pass:

1. shape and lifecycle contract tests;
2. temporal causality and episode-isolation tests;
3. recurrent KV-cache versus parallel-sequence equivalence;
4. inactive-agent and illegal-action masking;
5. complete-joint-action sensitivity;
6. target-encoder stop-gradient and EMA routing;
7. coherent-imagination alignment across agents;
8. tiny-dataset world-model overfit;
9. deterministic rerun equivalence;
10. a real-environment control gate against MAPPO under matched accounting.

Environment-specific reward shaping, task-specific replay rules, and hidden
checkpoint selection are outside the algorithm.
