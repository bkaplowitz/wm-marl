# DreaMARL Architecture

## 1. Purpose and Invariants

DreaMARL is a reconstruction-regularized, JEPA-based model-based RL algorithm
with centralized interaction modelling and decentralized control. It extends
the registered single-agent M3 architecture over an explicit agent axis:

```text
policy tensors: [batch, agent, ...]
replay tensors: [batch, time, agent, ...]
```

One architecture handles every agent count. Agent count changes tensor
geometry only; it does not select a different actor, critic, optimizer,
training schedule, replay rule, or objective. All neural parameters are shared
across agents. At one agent, the interaction residuals and their gradients are
exactly zero without a Python branch on the agent count.

The core separation is:

- local latent state for the posterior, actor, critic, and decoder;
- interaction-aware latent predictions for the dynamics prior, JEPA predictor,
  reward model, and continuation model.

This prevents centralized information from leaking into decentralized policy
inputs while allowing the world model to explain other agents.

## 2. First-Party Implementation

The executable algorithm lives in `world_marl.dreamarl`. Its agent, dynamics,
interaction model, environment contract, training loop, configuration, and
entrypoint are maintained in this repository. The pinned Dreamer-CDP checkout
provides Embodied infrastructure such as replay storage, device transforms,
neural primitives, and logging; it is not the active algorithm implementation.

The exact successful M3 source is preserved under `world_marl.dreamarl.m3` as
a frozen one-agent oracle. Its files are hash-checked against commit
`a851fa3e3d70b624b094ee1810ad4bb602346092` and are never imported by the
active learner.

## 3. Local M3 State

Each agent receives its own `64 x 64 x 3` observation and maintains the local
belief

```text
b_t^i = (h_t^i, z_t^i).
```

### Visual representation

- M3 convolutional encoder;
- M3 convolutional decoder retained as a representation regularizer and for
  open-loop visual reports;
- decoder gradients do not update the representation (`dec_grad: false`).

### Stochastic state

- 32 categorical variables;
- 64 classes per variable;
- 2,048-dimensional flattened stochastic sample;
- 1% uniform mixture in the categorical distribution.

### Local causal dynamics

- causal Transformer width: 512;
- layers: 4;
- heads: 8;
- feed-forward expansion: 4;
- recurrent KV context: 64 transitions;
- deterministic state: 8,192 dimensions;
- RMS normalization, SiLU activations, and rotary positions.

The temporal transition is independent for each agent:

```text
h_{t+1,local}^i = T_M3(h_{<=t}^i, z_t^i, a_t^i).
```

The posterior remains strictly local:

```text
q_local(z_{t+1}^i) = q(z_{t+1}^i |
  h_{t+1,local}^i, encoder(o_{t+1}^i)).
```

No other-agent observation, posterior, message, or future representation is an
input to this posterior.

## 4. Causal Interaction Model

At time `t`, actions are sampled from local beliefs. The interaction token for
agent `j` is

```text
x_t^j = [h_t^j, flatten(z_t^j), encode(a_t^j)].
```

A single shared multi-head attention block computes

```text
m_t^i = Mixer(query=b_t^i, keys/values={x_t^j : j != i}).
```

The mixer has width 128 and 4 heads. It has no agent IDs or agent-axis
positional embeddings. State and action remain paired. Attention logits are
computed in float32, self-attention is excluded, and masked softmax returns
exact zeros when no other agent is available. A `has_other` gate is applied
after every output projection, including its bias.

The interaction module uses a dedicated initialization RNG stream derived from
the run seed. Adding its parameters therefore does not perturb the M3
initialization stream.

### Replay alignment

Replay already supplies `prevact[:, t] = action[:, t - 1]`. For replay row
`t`, the mixer source is the previous local belief and that previous action:

```text
source_h = concat(initial_h, posterior_h[:, :-1])
source_z = concat(initial_z, posterior_z[:, :-1])
message_{t-1} = Mixer(source_h, source_z, prevact_t)
```

Messages are masked at episode resets. They never consume observations or
posterior states from the prediction target timestep.

## 5. Dual Priors and JEPA Prediction

The local prior is the unchanged M3 prior:

```text
p_local = p_M3(z_{t+1}^i | h_{t+1,local}^i).
```

The joint prior is a zero-initialized residual:

```text
p_joint_logits = p_local_logits
               + delta_prior(h_{t+1,local}^i, m_t^i).
```

The two KL directions deliberately use different priors:

```text
L_dyn = KL(stop_gradient(q_local) || p_joint)
L_rep = KL(q_local || stop_gradient(p_local)).
```

Thus joint information improves forward prediction but cannot train the local
posterior toward a privileged centralized target.

The JEPA predictor is also residual:

```text
predicted_token = P_M3(h_{t+1,local}^i)
                + delta_predictor(h_{t+1,local}^i, m_t^i)
L_JEPA = cosine_distance(predicted_token,
                         stop_gradient(encoder(o_{t+1}^i))).
```

`slowenc.enable` is false, so the target is the stopped-gradient current
encoder token. The JEPA loss weight is 500. KL free nats and all other M3 loss
scales remain unchanged.

## 6. Local and World Features

The local feature is

```text
f_local = [h_local, flatten(z)].
```

Only `f_local` enters:

- the actor;
- the critic and slow critic;
- the posterior;
- the visual decoder.

The world feature is another zero-initialized residual:

```text
f_world = f_local + delta_world(h_local, message).
```

Only `f_world` enters:

- reward prediction;
- continuation prediction.

MeltingPot currently supplies the arithmetic mean of per-agent rewards as one
team-reward target. That scalar is broadcast to shared per-agent heads. A
pooled team head is intentionally not introduced in this first causal test.

## 7. Decentralized Control and Joint Imagination

The M3 actor and critic are unchanged:

- shared actor: 3 hidden layers of width 1,024;
- continuous policy: bounded Normal with standard deviation in `[0.1, 1.0]`
  and 1% mixture;
- discrete policy: categorical with 1% mixture;
- critic: 3 hidden layers of width 1,024 and a 255-bin symlog two-hot output;
- imagination length: 15;
- effective return horizon: 333;
- lambda: 0.95;
- entropy coefficient: `3e-4`;
- slow-value update rate: `0.02` each update;
- replay-value loss enabled.

During imagination, all agents first sample actions from local features. Local
states and sampled actions are regrouped by joint imagined environment, the
mixer computes same-time interaction messages, each local Transformer advances
independently, and the joint priors sample all next stochastic states. Reward
and continuation use world features; actor and critic continue using local
features.

Imagination starts are ordered explicitly:

```text
[B*A, T] -> [B, A, T] -> [B, A, K]
         -> [B, K, A] -> [B*K*A].
```

The inverse permutation is applied before replay-value bootstrapping and loss
reduction. Open-loop reports select complete agent groups and execute the same
joint imagination path.

## 8. Training Regime

The M3 regime is preserved for every agent count:

- batch size: 16 joint sequences;
- sequence length: 64 plus 1 stored replay-context transition;
- replay capacity: 5 million joint environment transitions;
- uniform online replay;
- MeltingPot train ratio: 256;
- encoder learning rate: `6e-6`;
- dynamics learning rate: `4e-4`;
- other-module learning rate: `4e-5`;
- AGC: `0.3`;
- BF16 compute;
- identical optimizer warmup, replay cadence, imagination, and reporting.

One environment step is one joint step regardless of agent count. Parameters
are shared, so increasing the number of agents does not replicate the model.
The registered M3 model has 164,884,230 trainable parameters. On Externality
Mushrooms, the complete interaction model has 174,151,818 parameters, adding
9,267,588 parameters. The exact count varies slightly with action width and is
independent of agent count.

The first-party training loop always writes a checkpoint at normal completion,
in addition to periodic time-based checkpoints.

## 9. Lifecycle Contract

MeltingPot benchmarks use fixed homogeneous agent sets and joint episode
boundaries. Environment-level fields are:

- `is_first`: reset every recurrent agent state;
- `is_last`: episode or time-limit boundary;
- `is_terminal`: disable continuation bootstrapping only for terminal states;
- `reward`: mean per-agent benchmark return used as the team reward.

Local images, actions, recurrent entries, and decoder entries preserve the
agent axis in replay. The current fixed-agent benchmark mask is all valid
except that interaction outputs are suppressed on reset rows.

## 10. Controls and Reduction Gates

The decisive experiment has three arms:

1. frozen independent M3: no interaction parameters;
2. aligned Interaction JEPA: correct joint environment context;
3. shuffled Interaction JEPA: equal interaction capacity, but context is
   rolled across complete environment trajectories.

The shuffled roll is constant over sequence time and keeps each state paired
with its action. It preserves parameter count, optimizer, mixer calls, temporal
coherence, agent count, and marginal feature distribution while destroying
only cross-agent correspondence.

At `A=1`, the canonical aligned model must match frozen M3 for every shared
parameter and optimizer state, policy action, recurrent carry, replay update,
algorithmic loss, and shared parameter update under fixed data and RNG. Its
message, residual outputs, interaction gradients, and interaction updates must
all be exactly zero. At `A>1`, messages and interaction gradients must become
nonzero without changing the actor, critic, posterior, optimizer, or schedule.

Aligned interaction is promoted only if it improves both model prediction and
return over independent M3 and the equal-capacity shuffled control across
multiple seeds, without materially regressing coordination tasks.
