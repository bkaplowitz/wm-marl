# DreaMARL Architecture

## 1. Design Invariant

DreaMARL is the multi-agent scaling of the registered M3 algorithm. It is one
agent-axis-native architecture, not a collection of single-agent and
multi-agent implementations.

All collected and replayed trajectories use an explicit agent axis:

```text
policy: [batch, agent, ...]
replay: [batch, time, agent, ...]
```

The encoder, decoder, world model, actor, critic, losses, and optimizers share
their parameters across that axis. Neural computation folds `batch * agent`,
runs the unchanged M3 modules, and restores the explicit axis afterward. Loss
reductions average over agents in the same way M3 averages over batch items.

There is no conditional module activation, alternate learner, or changed
training regime based on the number of agents. For one agent, every fold and
unfold is an identity reshape. The resulting parameter tree and numerical
computation therefore reduce to M3.

## 2. First-Party Source

The executable algorithm is maintained under `world_marl.dreamarl`. DreaMARL
executes its own `main.py`, `agent.py`, `rssm.py`, `transformer_rssm.py`,
configuration, axis operations, and environment contracts directly. These
active modules do not import or subclass the frozen M3 reference. The pinned
Dreamer-CDP checkout supplies only the third-party Embodied runtime, replay
storage, device distribution, neural-network primitives, and utilities.

The exact registered M3 source is preserved under `world_marl.dreamarl.m3` as
the reduction oracle. Its provenance is:

- Dreamer-CDP commit `a851fa3e3d70b624b094ee1810ad4bb602346092`;
- the registered causal-Transformer RSSM integration from this repository;
- the upstream MIT license retained beside the source.

No generated overlay, inherited M3 agent, or prior M3 launcher is used by the
DreaMARL executable. The frozen reference is test-only and remains deliberately
unmodified so exact one-agent reduction can be checked against it.

## 3. Shared M3 Model Lifted Across Agents

Every agent uses the same shared model parameters. Agent count is only a
tensor extent supplied by the environment contract. It is not a mode flag:
no model component, loss, optimizer, schedule, or replay rule is enabled or
disabled when that extent changes.

### Visual representation

- Input image: `64 x 64 x 3` pixels.
- Encoder: the M3 convolutional visual encoder.
- Decoder: the M3 visual decoder, retained for reconstruction training and
  open-loop reporting.
- Reconstruction targets do not backpropagate through the representation when
  `dec_grad: false`, matching the registered configuration.

### Stochastic state

- 32 categorical variables;
- 64 classes per variable;
- 2,048-dimensional flattened stochastic sample;
- 1% uniform mixture for categorical regularization.

### Causal Transformer dynamics

- model width: 512;
- layers: 4;
- attention heads: 8;
- feed-forward expansion: 4;
- recurrent KV context: 64 transitions;
- deterministic output state: 8,192 dimensions;
- RMS normalization, SiLU activations, and rotary position encoding.

At each time step the temporal input is the previous stochastic state joined
with the previous bounded action. The recurrent KV cache summarizes causal
history. The posterior combines the current encoded observation with the
Transformer state; the prior predicts the stochastic state from the
Transformer state alone.

### Prediction heads

The shared latent state feeds:

- reward prediction;
- continuation prediction;
- visual reconstruction;
- the bounded-Normal actor;
- the distributional value critic.

### JEPA objective

The deterministic state predicts the current encoder token. The target is a
stopped-gradient token from the same encoder because `slowenc.enable: false`
in the registered M3 configuration. The loss is cosine distance with weight
500. KL dynamics and representation losses retain the original free-nat and
loss-scale settings.

## 4. Control

The actor and critic are unchanged from M3:

- actor: three hidden layers of width 1,024;
- continuous distribution: bounded Normal with standard deviation in
  `[0.1, 1.0]` and 1% action mixture;
- critic: three hidden layers of width 1,024 with a 255-bin symlog two-hot
  output;
- imagination length: 15;
- effective return horizon: 333;
- lambda: 0.95;
- actor entropy coefficient: `3e-4`;
- slow-value update rate: `0.02` every update;
- replay-value loss enabled.

The actor is decentralized in the out-of-the-box scaling: each agent acts from
its own M3 latent state, while all agents share one policy. No agent-count-
dependent actor or critic is introduced.

The current reduction baseline applies the same temporal model independently
to each folded agent trajectory. This is a uniform operation for every agent
count, including one; it is not a conditional fallback. Any future
cross-agent operator must likewise be defined over the agent axis for every
axis extent and have an explicit one-agent reduction. It must never be
activated by an `if num_agents > 1` branch.

## 5. Training Regime

Agent count does not change the M3 training configuration:

- batch size: 16 joint sequences;
- sequence length: 64 plus 64 replay-context transitions;
- replay capacity: 5 million joint transitions;
- replay sampling: uniform and online;
- visual-DMC train ratio: 256;
- optimizer: the registered M3 multi-transform optimizer;
- encoder learning rate: `6e-6`;
- dynamics learning rate: `4e-4`;
- remaining module learning rate: `4e-5`;
- AGC: `0.3`;
- BF16 compute;
- identical warmup, checkpoint, RNG, reporting, and evaluation semantics.

One environment step remains one joint environment step regardless of agent
count. Increasing the number of agents adds aligned observations and actions to
that joint transition; it does not multiply the update cadence or silently
change the real-to-replay ratio.

The registered one-agent model has 164,884,230 trainable parameters. Parameter
sharing means this count is independent of the number of agents in the current
out-of-the-box scaling.

## 6. Lifecycle Contract

Joint lifecycle fields remain environment-level scalars:

- `is_first` resets all recurrent agent states at a joint episode boundary;
- `is_last` marks any episode or time-limit boundary;
- `is_terminal` alone disables continuation bootstrapping;
- `reward` is the cooperative environment reward used by every shared-policy
  agent.

These fields are broadcast across the agent axis immediately before the M3
loss. Local observations, actions, recurrent entries, and decoder entries keep
the explicit agent axis in replay.

## 7. Single-Agent Reduction Gate

Before interpreting any MARL result, the singleton geometry must pass:

1. frozen M3 source and configuration identity;
2. equal parameter names, shapes, dtypes, and total count;
3. equal outputs and recurrent state on fixed inputs and RNG seeds;
4. equal loss terms and gradients on fixed replay batches;
5. equal optimizer state and parameters after one update;
6. equal replay samples and environment-step accounting;
7. equal latest-policy evaluation on Reacher Easy, Cheetah Run, and Hopper Hop
   under one shared configuration.

The single-agent DMC adapter only inserts and removes a singleton agent axis.
It makes no random calls and does not alter observation, action, reward, or
episode values.

## 8. Current Scope

The model and replay contracts are agent-axis-native. The currently connected
executable environment is visual DMC through the singleton adapter. Therefore,
the implementation has proven numerical `A=1` reduction and supports `A=N`
tensor geometry, but it cannot yet claim an executed native MAMuJoCo result. A
native MAMuJoCo adapter must preserve the same joint-step, lifecycle, reward,
observation, and action contracts without changing the learner or schedule.

Matching M3 numerically on a one-agent MAMuJoCo configuration is expected only
when that environment configuration is semantically equivalent to the M3 task.
Changing simulator observations, rewards, action repeat, termination, or camera
semantics changes the benchmark even though the learner remains identical.

The first multi-agent experiment is intentionally the unchanged, parameter-
shared M3 model described above. Cross-agent communication, centralized value
information, or other MARL-specific mechanisms are scientific interventions,
not implicit consequences of changing agent count. They are considered only
after this out-of-the-box baseline identifies an actual multi-agent failure.
