# Milestone 3: Stochastic JEPA Transformer

## Scope

Milestone 3 changes one causal component of the exact Milestone 2
Dreamer-CDP system: the recurrent deterministic RSSM transition becomes a
causal Transformer temporal state. The visual encoder, categorical stochastic
state, JEPA target and loss, prediction heads, actor-critic, replay schedule,
training ratio, and evaluation protocol remain fixed.

This is a development comparison. Seed 0 is used while the implementation is
being validated. A candidate must pass the registered two-seed promotion gate
before it can replace the JEPA-RSSM reference.

## Temporal Contract

Define the completed state-action pair

```text
x_t = concat(z_t, a_t).
```

The temporal state available before observing `o_t` is

```text
h_t = Transformer(x_0, ..., x_{t-1}).
```

The posterior, prior, and control flow are

```text
e_t = Encoder(o_t)
z_t ~ q(z_t | h_t, e_t)
z_t ~ p(z_t | h_t)                 # during imagination
a_t ~ pi(a_t | h_t, z_t)
x_t = concat(z_t, a_t)
h_{t+1} = Transformer.append(x_t)
```

Consequently, `h_t` cannot depend on `z_t`, `a_t`, `o_t`, or any future
quantity. `is_first_t` clears all preceding temporal history before `h_t` is
computed.

## Two Equivalent Execution Paths

Training uses a parallel causal pass over replay sequences. Collection and
imagination use a fixed-size recurrent cache. Both paths share every parameter
and must agree numerically while the active episode fits in the context window.

The initial implementation in
`src/world_marl/jepa_transformer/temporal.py` establishes:

- strict shifted-input alignment;
- episode-segment causal masking;
- reset-safe recurrent caches;
- shared parallel and recurrent parameters;
- fixed context and memory bounds;
- a state projection compatible with the existing Dreamer heads.

## Imagination Start States

Dreamer starts imagined trajectories from multiple replay positions. A GRU
state can be copied directly; a Transformer state also requires its preceding
context. Milestone 3 therefore reconstructs each selected start cache from the
causal replay window ending immediately before that state.

The replay entry stores only one pair and its boundary metadata per time step.
It does **not** store a full context or per-layer KV cache at every transition.
When starts are selected, a bounded sliding-window gather constructs the
required caches. This preserves the existing number and distribution of
imagination starts without quadratic replay storage.

Reducing `imag_last`, shortening imagination, or using history-free starts is
not permitted in the primary M2-to-M3 comparison because each would change the
policy-learning problem in addition to the temporal architecture.

## Implementation Gates

1. **Temporal unit gate**
   - current/future pairs cannot affect `h_t`;
   - reset boundaries erase prior history;
   - cached and parallel execution agree;
   - context overflow is explicit.
2. **RSSM integration gate**
   - posterior and prior shapes match M2;
   - all selected imagination starts receive correct causal histories;
   - reward, continuation, actor, and critic consume `(h_t, z_t)`;
   - no pixel reconstruction gradient reaches the representation.
3. **Runtime gate**
   - one full-width GPU smoke reaches a world-model and actor update;
   - parameters, peak VRAM, update speed, and imagined-step speed are logged;
   - no representation or stochastic-state collapse is observed.
4. **Control gate**
   - seed-0 short runs precede the registered two-seed comparison;
   - final promotion requires at least 95% of JEPA-RSSM aggregate AUC;
   - a memory-sensitive task and rollout efficiency must justify the change.

## Non-Goals

Milestone 3 does not introduce a ViT, alternative JEPA loss, EMA target
encoder, new exploration rule, replay prioritization, actor objective, MARL
structure, or task-specific hyperparameters. Those changes belong to later
milestones or require a separate causal hypothesis.
