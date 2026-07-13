# Current JEPA Reacher Architecture

This document describes only the architecture and training protocol of the
currently running full experiment
[`jepa-entropy-decay-300k-500k-reacher-seed2`](https://wandb.ai/osaze-obahor/world-marl/runs/hppw7dmr).
It uses 497,664 real training transitions and finishes with a fixed 100-episode
evaluation of the latest policy. The implementation lives in
[`models.py`](models.py), [`training.py`](training.py), and
[`train_dmc_jepa.py`](../scripts/train_dmc_jepa.py).

## System Overview

The system consists of a deterministic, action-conditioned latent world model
and an actor-critic trained from recurrent latent imagination:

```text
observation o_t
    |
    v
shared MLP encoder ---------------------------------------+
    |                                                     |
    v                                                     v
latent z_t                                      stopgrad target z_{t+k}
    |
    +----> actor MLP ----> tanh-Normal action a_t
    |
    +----> critic MLP ---> value distribution V(z_t)
    |
    v
latent projection + action MLP
    |
    v
2-block causal transformer over 8 latent/action steps
    |
    +----> residual MLP predictor ---> predicted latent z_hat_{t+1}
    +----> reward head -------------> reward distribution
    +----> continue head -----------> p(not terminal)
```

The model never reconstructs observations. It predicts future representations,
rewards, and continuation in latent space. The reinforcement-learning component
starts imagination from real replay states, rolls the actor through the learned
dynamics, and updates actor and critic from imagined lambda returns.

There is one causal transformer, not separate transformer encoder and predictor
stacks. The observation encoder and latent predictor are MLPs. The target is the
same observation encoder applied to a future observation with stopped gradient;
there is no separate or EMA target encoder.

## Environment Contract

The running experiment uses DMC proprioceptive state observations:

- the DMC observation dictionary is flattened into one `float32` vector;
- actions are continuous and retain the environment's native bounds;
- there is no pixel encoder, frame stacking, or action repeat;
- one adapter step is one DMC control step;
- episodes are capped at 1,000 steps;
- 16 independently seeded environments collect data in parallel.

For `reacher/easy`, the flattened observation dimension is 6 and the action
dimension is 2.

## World Model

### Observation Encoder

The shared encoder maps the current observation to a 128-dimensional latent:

```text
o_t
 -> symlog
 -> Dense(128), SiLU
 -> Dense(128), SiLU
 -> Dense(128)
 -> RMSNorm
 -> z_t
```

The encoder is trained online throughout the run. Future targets use this same
encoder and `stop_gradient`; its parameters are not maintained by an EMA.

### Action-Conditioned Transformer

A continuous action is embedded by `Dense(128) -> SiLU -> Dense(128)`. At each
time step, the transformer token is:

```text
token_t = Dense(z_t) + action_encoder(a_t)
```

The dynamics trunk contains two pre-norm transformer blocks with:

| Property | Value |
| --- | ---: |
| Width | 128 |
| Context window | 8 steps |
| Blocks | 2 |
| Attention heads | 4 |
| Head dimension | 32 |
| Position encoding | RoPE on queries and keys |
| Attention mask | Causal and episode-boundary aware |
| Feed-forward | GEGLU, inner width 512 |
| Activation | SiLU |
| Normalization | RMSNorm |
| Dropout | None |

A final RMSNorm produces the dynamics hidden state. Collector-imposed bootstrap
cuts are excluded by the replay sampler. Natural terminal/reset boundaries may
occur inside sampled windows; done-aware attention and loss masks prevent
information and prediction targets from crossing those boundaries.

### Prediction Heads

Given the latest transformer hidden state:

- **Latent predictor:** one-hidden-layer width-128 MLP predicts a latent update.
  The update is added to the current latent and RMS-normalized.
- **Reward head:** one-hidden-layer width-128 MLP predicts 255 symlog two-hot
  logits over support `[-20, 20]`. Its output kernel starts at zero.
- **Continue head:** one-hidden-layer width-128 MLP predicts one continuation
  logit trained against `1 - done`.

The model is recurrently unrolled for five supervised prediction steps. Later
steps consume the model's previous predicted latent, so the five-step loss
directly trains short open-loop behavior rather than only one-step teacher
forcing. The running model uses one dynamics head.

### World-Model Objective

The world-model loss is:

```text
L_WM = L_cosine_latent
     + 0.05 * L_SIGReg
     + L_reward_twohot
     + L_continue_BCE
```

- `L_cosine_latent` compares normalized predictions with normalized stopped
  future latents.
- SIGReg uses 1,024 random projections and 17 integration knots to keep the
  representation distributed and resist collapse.
- Reward and continuation losses both have weight 1.0.
- Invalid targets after terminal boundaries are masked from every loss.

This is a deterministic latent model: there is no RSSM stochastic state,
observation decoder, reconstruction loss, or KL loss.

## Actor-Critic

The actor and critic read the current encoded latent directly. They do not use
the transformer and are not recurrent. DMC state observations provide the
Markov state needed at execution time.

### Actor

For Reacher, the actor is:

```text
z_t
 -> RMSNorm
 -> 3 x [Dense(64), SiLU]
 -> Dense(4) = [mean_1, mean_2, log_std_1, log_std_2]
```

Both heads apply RMSNorm once at their input. They do not normalize after every
hidden layer.

The actor output kernel uses scale `0.01`. Log standard deviations are clipped
to `[log(0.1), 0]`, giving pre-squash standard deviations in `[0.1, 1.0]`.
Training, imagination, and online collection sample from the Gaussian and apply
`tanh`; final evaluation uses the deterministic squashed mean action.

### Imagination and Actor Objective

Each actor update samples 1,024 valid eight-step contexts uniformly from the
full replay and imagines 15 recurrent world-model steps. During this update:

- all world-model parameters are frozen;
- sampled actions are detached before entering the world model;
- the actor uses a squash-corrected REINFORCE gradient, not gradients through
  the learned dynamics;
- rewards and continuation probabilities come from the world model;
- the EMA target critic bootstraps 15-step lambda returns;
- returns use `gamma = 1 - 1/333 = 0.996996996997` and `lambda = 0.95`;
- returns are clipped to `[-100, 100]`;
- the stopped EMA target-critic value is used as the actor baseline;
- actor scores are divided by an EMA of the batch p95-p5 return range, with
  decay `0.99` and a minimum scale of 1;
- tanh-Normal entropy uses coefficient `3e-3` through 300,000 training
  transitions, then decays linearly toward `3e-4` at 500,000 transitions.

The run collects exactly 497,664 training transitions, so its final actor update
uses entropy coefficient approximately `3.315e-4`.

### Critic

The critic is:

```text
z_t
 -> RMSNorm
 -> 3 x [Dense(64), SiLU]
 -> Dense(255) symlog two-hot value logits
```

Its output kernel starts at zero. The critic combines:

1. distributional loss against clipped lambda returns from imagined rollouts;
2. slow-value regularization toward an EMA target critic, coefficient `1.0`;
3. replay critic loss, coefficient `0.3`, on every state in uniformly sampled
   real replay sequences of length 64.

The target critic uses EMA decay `0.98`. Only the value head is EMA-updated;
there is no EMA world model or encoder. Critic warmup is disabled.

### Parameter Update Boundaries

Three masked optimizers enforce ownership:

| Update | Trainable parameters | Frozen parameters |
| --- | --- | --- |
| World model | encoder, action encoder, transformer, predictor, reward and continue heads | actor and critic |
| Actor | actor head | world model and critic |
| Critic | value head | world model and actor |

The actor and critic heads are freshly initialized once before the initial
policy fit. They are then carried forward through every online phase; there is
no periodic policy reset, checkpoint selection, or champion replacement.

## Reference Hyperparameters

### Model and Optimization

| Parameter | Value |
| --- | ---: |
| Latent dimension | 128 |
| Transformer dimension | 128 |
| Transformer blocks / heads | 2 / 4 |
| Transformer MLP ratio | 4 |
| Context window | 8 |
| Supervised model horizon | 5 |
| Dynamics ensemble size | 1 |
| World-model batch size | 16 sequences |
| Replay chunk length | 64 |
| Input transform | symlog |
| Activation / normalization | SiLU / RMSNorm |
| Latent objective | cosine with stopped target |
| Anti-collapse regularizer | SIGReg, weight 0.05 |
| Reward / value prediction | 255-bin symlog two-hot |
| Two-hot support | `[-20, 20]` |
| World-model / actor / critic learning rate | `4e-5` each |
| Adam epsilon | `1e-8` |
| Optimizer warmup | 1,000 updates per optimizer |
| Adaptive gradient clipping | 0.3 |
| Global gradient clip: model / actor / critic | disabled / 10 / 100 |

### Control

| Parameter | Value |
| --- | ---: |
| Actor / critic hidden width | 64 / 64 |
| Actor / critic hidden layers | 3 / 3 |
| Policy imagination batch | 1,024 |
| Policy imagination horizon | 15 |
| Actor gradient | REINFORCE |
| Return | lambda return |
| Discount / lambda | `0.996996996997` / `0.95` |
| Actor baseline | value |
| Return normalization | EMA p95-p5, decay 0.99 |
| Value clip | 100 |
| Entropy | tanh-Normal; `3e-3` through 300k, linear decay toward `3e-4` at 500k |
| Target critic EMA | 0.98 |
| Slow-value coefficient | 1.0 |
| Replay critic coefficient | 0.3 |
| Replay critic batch / horizon | 16 / 64 |

## Parameter Count

Parameter count depends slightly on observation and action dimensions. The
instantiated `reacher/easy` model has:

| Component | Trainable parameters |
| --- | ---: |
| JEPA world model, including reward and continue heads | 694,912 |
| Actor | 16,964 |
| Critic | 33,279 |
| **Total** | **745,155** |

The EMA target critic adds no trainable parameters; it maintains a moving copy
of the 33,279 critic values. Optimizer states are not included above.

## Data and Update Schedule

### Reset-Rich Bootstrap

The initial replay is collected with uniformly random actions in a separate set
of environments. Each of the 16 environments contributes four independently
reset 80-step segments:

```text
320 vector collection steps x 16 environments = 5,120 train transitions
```

Collector cuts ensure replay sequences cannot cross the artificial 80-step
boundaries. Because bootstrap uses a separate adapter, the online environments
remain at their first reset until policy collection begins.

A separate fixed-seed held-out replay contains:

```text
80 vector collection steps x 16 environments = 1,280 validation transitions
```

Held-out transitions are used only for world-model diagnostics. They never train
the model, select a policy, gate an update, or enter the replay buffer.

### Initial and Online Training

After bootstrap:

1. run 1,280 world-model updates on the initial replay;
2. initialize the policy heads and run 1,280 actor-critic updates;
3. repeat tightly interleaved online phases using the latest stochastic policy.

Each online phase performs:

```text
64 vector collection steps x 16 environments = 1,024 new transitions
1,024 world-model updates from full replay
512 actor-critic updates from full replay
```

The replay capacity is 1,000,000 total transitions, so the run retains all
collected training data. Sampling is uniform over valid contiguous starts and
environment streams. The encoder remains trainable in every world-model phase.

| Online phases | Train transitions | Held-out transitions | Train + held-out | WM updates | Policy updates |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 481 | 497,664 | 1,280 | 498,944 | 493,824 | 247,552 |

The 500k run therefore contains exactly 497,664 training
transitions; phase granularity leaves it 2,336 transitions below 500,000.
Validation and final evaluation interactions are tracked separately.

## Evaluation and Reporting

Online return curves come from episodes completed during stochastic training
collection; they do not spend additional evaluation interactions. The final
evaluation:

- loads no selected checkpoint and performs no checkpoint search;
- evaluates the latest policy produced by the final optimizer update;
- uses deterministic mean actions;
- uses fixed evaluation seed `9_000_000`;
- runs for 100 episodes.

The failure threshold 100 and success threshold 900 are reporting labels for
Reacher diagnostics only. They never affect collection, replay sampling, losses,
or policy updates.

W&B training metrics use `budget/train_env_steps` as their x-axis. Final rows
also report validation, policy-evaluation, and total real interactions
separately. A 100-episode Reacher evaluation can add up to 100,000 reporting
interactions, but those interactions are not counted as training data.

## Reproducibility

The current run uses isolated named RNG streams for:

- initialization;
- initial and online collection;
- world-model replay sampling;
- actor-critic replay sampling and imagination;
- validation.

Runs request deterministic accelerator reductions and highest JAX matrix
multiplication precision. Every run records resolved arguments, dependency
versions, replay fingerprints, parameter fingerprints, target-critic
fingerprints, exact transition counts, and reload-equivalence diagnostics.

Recovery checkpoints are written every 16 online phases and at the final phase.
They exist for fault recovery only and do not participate in policy selection.
