# DreaMARL Architecture

## Scope

DreaMARL is a reconstruction-free, stochastic, joint world model for
cooperative multi-agent reinforcement learning from local visual observations.
It uses centralized training and decentralized execution (CTDE): the learned
world and value models can use the complete synchronized team state during
training, while each deployed actor uses only its own observation and action
history.

The maintained implementation has one architecture and one training path. It
does not switch models, objectives, replay schedules, or optimization rules as
the number of agents changes. A one-agent input is a valid reduction of the
same tensor program, but single-agent parity is not an organizing constraint.

## Tensor Contract

Environment interactions are joint transitions. Replay preserves both time and
agent axes:

```text
local observation     [B, T, A, ...]
local action          [B, T, A, ...]
local reward          [B, T, A]
is_first/is_last      [B, T]
is_terminal           [B, T]
```

Here `B` is the environment batch, `T` is time, and `A` is agent count. One
environment step counts once regardless of `A`. `is_last` prevents replay and
return sequences from crossing resets; only `is_terminal` disables value
bootstrapping.

Local neural modules may fold `B` and `A` solely to share parameters. The joint
world model never folds away the agent axis.

## Local Execution State

Every agent observes a local `64 x 64 x 3` RGB image. One shared convolutional
encoder maps it to a 4,096-dimensional embedding:

```text
e_t^i = Encoder(o_t^i)
```

A two-layer causal Transformer of width 768 and 12 attention heads maintains
the local policy belief:

```text
l_t^i = LocalBelief(e_t^i, a_{t-1}^i, l_{<t}^i)
```

Its attention cache is bounded to the latest 64 transitions. Parameters are
shared across agents, but caches and inputs are private. There are no agent
identifiers, peer observations, peer actions, rewards, values, or joint-world
features in this path.

The decentralized actor is:

```text
a_t^i ~ pi(a | l_t^i)
```

It is a three-layer MLP of width 1,024 with a categorical output and 1% uniform
mixture for Melting Pot. Calling `Agent.policy()` cannot access the joint world
model or centralized critic.

## Joint Predictive State

The authoritative environment state is one stochastic joint latent, not a set
of independent local simulators. Its carry contains:

```text
global token          g_t       [B, 768]
agent states          d_t       [B, A, 768]
categorical latents   s_t       [B, A, 32, 64]
```

The posterior infers this state from the complete synchronized set of local
embeddings and local beliefs:

```text
q(S_t | S_{t-1}, a_{t-1}^{1:A}, e_t^{1:A}, l_t^{1:A})
```

The prior advances it with the complete joint action:

```text
p(S_{t+1} | S_t, a_t^{1:A})
```

Both use a two-layer, width-768, 12-head set Transformer over one global
token and `A` agent tokens. Agent positional identifiers are absent, making the
model permutation equivariant for homogeneous agents. Every next-agent
prediction is generated from the same sampled world state.

The world predicts:

- every agent's next local encoder embedding;
- a reward for every agent;
- one joint continuation probability.

The principal JEPA target is the stopped-gradient next encoder embedding. The
world objective combines one-step cosine prediction, categorical dynamics and
representation KL terms, and open-loop JEPA overshooting at horizons 2, 4, and
8. Overshooting weights are 0.5, 0.25, and 0.125 and reset-crossing targets are
masked.

The maintained model has no observation decoder and no reconstruction loss.
World-model learning is entirely predictive in representation space. Optional
visual probes may train a separate decoder from frozen checkpoints, but that
probe is not part of the algorithm, optimizer, checkpoint, or parameter count.

## Synchronous Imagination

An imagined transition is atomic:

```text
1. Each actor samples from its own local belief.
2. The actions are assembled into one joint action.
3. The joint prior advances exactly once.
4. The world predicts every next local embedding.
5. Each local belief advances from its own predicted embedding and action.
```

No agent advances before the current actions of all agents are known. The
imagination horizon is 15. This lets local actors react to interaction effects
through their predicted future local observations without giving them
centralized information at execution time.

## Centralized Value Learning

The critic consumes a pooled feature of the complete joint latent state:

```text
V_t = V(GlobalPool(S_t))
```

It is a three-layer width-1,024 MLP with a 255-bin symlog two-hot output. The
team objective is the mean of the predicted per-agent rewards, matching the
benchmark's mean-per-agent episode return. The same team advantage trains each
shared local actor with REINFORCE and lambda returns.

Control settings are:

- imagination length: 15;
- return horizon: 333;
- lambda: 0.95;
- entropy coefficient: `3e-4`;
- slow-value update rate: `0.02`;
- replay-value loss: enabled;
- actor and critic gradients into the world model: disabled.

## Replay And Optimization

The maintained Melting Pot configuration uses:

- 16 joint sequences per batch;
- 64 burn-in transitions plus 64 optimized transitions;
- 50% uniform and 50% recency replay sampling;
- replay capacity of five million joint transitions;
- train ratio 256;
- BF16 compute;
- adaptive gradient clipping of 0.3.

Learning rates are `6e-6` for the visual encoder, `4e-5` for the local belief
and joint world, and `4e-5` for prediction heads, actor, and critic.
Replay context is recomputed from raw observations as learner-side burn-in;
implementation-specific recurrent caches are not authoritative replay data.

The Externality-sized model contains 81,847,303 trainable parameters:

| Module | Parameters |
| --- | ---: |
| Joint world | 44,488,448 |
| Local belief | 17,326,848 |
| Central critic | 6,034,687 |
| Visual encoder | 3,492,864 |
| Decentralized actor | 2,897,928 |
| Reward head | 3,933,439 |
| Continuation head | 3,673,089 |
| **Total** | **81,847,303** |

Parameters are shared across agents, so model size does not grow with team
size. Activation and attention compute do grow with `A`.

## Evaluation

Training return is the mean per-agent episode return. Fixed evaluation restores
the latest checkpoint, uses deterministic actor modes and an explicit seed,
and records every team and per-agent return. Evaluation episodes are reporting
only: they neither train the model nor select a checkpoint.

World-model reports expose one-step and 2/4/8-step latent error, per-agent reward
error, and continuation calibration. These are paired with return curves to
distinguish model improvement from control improvement.

## Correctness Invariants

Automated contracts require:

1. local-policy isolation from every peer and joint-state tensor;
2. an explicit agent axis throughout the joint posterior and prior;
3. permutation equivariance under consistent agent reordering;
4. cross-agent sensitivity to a changed focal action;
5. one coherent global state sample for all predicted agents;
6. atomic synchronous imagination;
7. vector reward preservation and one joint continuation;
8. a centralized critic used only during training;
9. validity at `A=1` without an agent-count-dependent architecture branch;
10. a 50/50 uniform-recency replay contract.

## Source Map

- `agent.py`: loss composition, joint imagination, actor, and critic;
- `local_belief.py`: strictly local recurrent execution state;
- `joint_model.py`: joint posterior, prior, JEPA losses, and overshooting;
- `perception.py`: local visual encoder and optional offline visual probe;
- `axes.py`: explicit local/joint tensor transformations;
- `meltingpot.py`: environment and reward-vector contract;
- `train.py` and `evaluation.py`: training and fixed evaluation;
- `config.py`, `configs.yaml`, `contracts.py`, `launcher.py`, and `runtime.py`:
  reproducible public execution and machine-checkable manifests.
