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

## Execution Path

Training, collection, and imagination use the same fixed-size recurrent KV
cache and the same Transformer parameters. Training scans over replay time with
JAX/Ninjax, while collection and imagination append one completed pair at a
time. The preliminary independent kernel in
`src/world_marl/jepa_transformer/temporal.py` establishes:

- strict shifted-input alignment;
- episode-segment causal masking;
- reset-safe recurrent caches;
- recurrent and full-sequence numerical equivalence;
- fixed context and memory bounds;
- a state projection compatible with the existing Dreamer heads.

The upstream integration is implemented in
`src/world_marl/jepa_transformer/upstream/m3_rssm.py`. A generated runtime
copies the immutable Dreamer-CDP source and applies only this registered
overlay; the official checkout is never modified.

## Imagination Start States

Dreamer starts imagined trajectories from multiple replay positions. A GRU
state can be copied directly; a Transformer state also requires its preceding
context. Milestone 3 therefore reconstructs the chunk-entry cache from a
64-transition replay prefix. During the subsequent 64 training transitions,
cache snapshots are retained only inside the compiled update and selected
directly for imagination starts.

The replay entry stores only one pair and its boundary metadata per time step.
It does **not** store a full context or per-layer KV cache at every transition.
This preserves the existing number and distribution of imagination starts
without quadratic replay storage.

## Registered Model

The causal Transformer has width 512, four pre-norm blocks, eight attention
heads, a 4x feed-forward expansion, rotary positions, and a 64-pair context.
It projects its output to the unchanged 8192-dimensional deterministic feature.
The categorical state remains 32 variables with 64 classes. The actor still
uses 15 imagined transitions and all Dreamer-CDP optimization settings remain
unchanged.

The measured full system has 164,884,230 trainable parameters. Of these,
94,427,648 belong to the complete dynamics module, including the Transformer,
prior, posterior, and JEPA predictor. The matched JEPA-RSSM reference has
216,280,326 parameters.

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
   - a 25,000-transition Reacher Easy seed-0 run precedes the registered
     two-seed comparison;
   - final promotion requires at least 95% of JEPA-RSSM aggregate AUC;
   - a memory-sensitive task and rollout efficiency must justify the change.

## Non-Goals

Milestone 3 does not introduce a ViT, alternative JEPA loss, EMA target
encoder, new exploration rule, replay prioritization, actor objective, MARL
structure, or task-specific hyperparameters. Those changes belong to later
milestones or require a separate causal hypothesis.
