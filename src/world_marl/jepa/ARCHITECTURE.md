# JEPA Model-Based Reinforcement Learning

This document is the blueprint for the JEPA agent instantiated by the currently
running `jepa-wm-only-500k-reacher-seed1` and
`jepa-wm-only-500k-reacher-seed2` experiments. It describes the resolved
configuration at commit `6c9ef0a` and no alternative configurations.

The implementation is defined by [`models.py`](models.py),
[`training.py`](training.py), [`replay.py`](replay.py), and
[`train_dmc_jepa.py`](../scripts/train_dmc_jepa.py).

The name `wm-only` refers only to the routing of recent replay: recent
transitions are emphasized for world-model fitting but not for actor start
states or real critic targets. The experiment still trains a complete
world-model plus actor-critic agent.

## 1. Algorithm Overview

The agent has three trainable components:

1. a deterministic JEPA world model that predicts future latent states,
   rewards, and continuation probabilities;
2. a stochastic actor that selects continuous actions from latent states;
3. a distributional critic that estimates returns from latent states.

The world model never reconstructs observations. It learns a predictive latent
space directly. The actor-critic learns from trajectories generated inside this
latent world, while an auxiliary critic loss anchors value estimates to rewards
observed in real replay.

```text
real observation o_t
        |
        v
  symlog + MLP encoder
        |
        +---------------------------> latent z_t
        |                                  |
        |                                  +--> actor --> action distribution
        |                                  |
        |                                  +--> critic --> value distribution
        |
        +--> stopped future target z_{t+k}

latent/action history
        |
        v
  causal transformer
        |
        +--> residual latent predictor --> z_hat_{t+1}
        +--> reward distribution -------> r_hat_t
        +--> continue logit ------------> c_hat_t
```

At execution time, only the observation encoder and actor are needed. During
learning, the transformer world model produces imagined trajectories and the
critic supplies the long-term bootstrap used by lambda returns.

## 2. Environment and State Contract

The reference experiment uses `dmc:reacher/easy` with proprioceptive vector
observations:

| Property | Value |
| --- | ---: |
| Parallel environments | 16 |
| Flattened observation dimension | 6 |
| Continuous action dimension | 2 |
| Episode limit | 1,000 adapter steps |
| Action repeat | 1 |
| Frame stacking | None |
| Pixel observations | None |

The DMC observation dictionary is flattened in a stable key order into one
`float32` vector. Actions retain the environment's native lower and upper
bounds. One collected transition corresponds to one call to the DMC control
environment.

## 3. JEPA World Model

### 3.1 Observation Encoder

The encoder maps an observation to a 128-dimensional latent state:

```text
o_t
 -> symlog
 -> Dense(128), SiLU
 -> Dense(128), SiLU
 -> Dense(128)
 -> RMSNorm
 -> z_t in R^128
```

The elementwise input transform is

```text
symlog(x) = sign(x) * log(1 + abs(x)).
```

One encoder is shared by current observations and future target observations.
The future target is stopped for the latent prediction loss:

```text
z_target = stop_gradient(encoder(o_future)).
```

There is no separately parameterized or EMA target encoder. The online encoder
continues learning during every world-model update.

### 3.2 Action-Conditioned Temporal Model

Continuous actions are embedded as:

```text
a_t -> Dense(128), SiLU -> Dense(128).
```

For every time step, the transformer input token is the sum of a projected
latent and the action embedding:

```text
x_t = Dense(z_t) + action_encoder(a_t).
```

The temporal model is a local causal transformer:

| Property | Value |
| --- | ---: |
| Transformer width | 128 |
| Context window | 8 transitions |
| Transformer blocks | 2 |
| Attention heads | 4 |
| Dimension per head | 32 |
| Attention position encoding | Rotary position embedding |
| Feed-forward block | GEGLU |
| Feed-forward inner width | 512 |
| Activation | SiLU |
| Normalization | Pre-norm RMSNorm |
| Dropout | None |

Attention is both causal and episode aware. A token can attend only to earlier
tokens in the same episode and within the eight-step window. Collector-imposed
bootstrap cuts are excluded at replay sampling time, while natural terminal
boundaries are handled by the attention and loss masks.

### 3.3 Prediction Heads

The final transformer state at a transition is consumed by three heads.

**Latent dynamics**

```text
h_t + one_step_embedding
 -> Dense(128), SiLU
 -> Dense(128)
 -> add current latent z_t
 -> RMSNorm
 -> z_hat_{t+1}
```

The latent predictor is residual: it predicts an update to the current latent,
not an unrelated absolute vector.

**Reward prediction**

```text
h_t -> Dense(128), SiLU -> Dense(255) reward logits.
```

The logits parameterize a categorical distribution over 255 equally spaced
bins in symlog space on `[-20, 20]`. Scalar rewards are encoded into the two
adjacent bins and trained by cross-entropy. Predictions are decoded by taking
the distributional expectation in symlog space and applying `symexp`. The final
reward kernel is initialized to zero.

**Continuation prediction**

```text
h_t -> Dense(128), SiLU -> Dense(1) continue logit.
```

The target is `1 - done`. During imagination, its sigmoid is used as the
probability that the imagined trajectory remains active.

The live model has one deterministic dynamics head. It has no stochastic latent
state, decoder, reconstruction objective, KL objective, or dynamics ensemble.

### 3.4 Recurrent Multi-Step Training

The supervised world-model horizon is five transitions. Starting from real
latent/action histories, the first next latent is predicted. That prediction is
then fed back into the temporal model for the next transition, and this process
is repeated for five steps:

```text
z_t -> z_hat_{t+1} -> z_hat_{t+2} -> ... -> z_hat_{t+5}.
```

Thus, later losses train the same recurrent path used by actor imagination;
they are not five independent one-step teacher-forced predictions. Reward and
continuation are supervised at every recurrent step. Targets that would cross
an episode boundary are masked.

### 3.5 World-Model Objective

For normalized prediction and target vectors, the latent loss is

```text
L_latent = mean_masked(1 - cosine(z_hat, stop_gradient(z_target))).
```

The complete objective is

```text
L_WM = L_latent
     + 0.05 * L_SIGReg
     + 1.0 * L_reward_twohot
     + 1.0 * L_continue_BCE.
```

SIGReg applies 1,024 random one-dimensional projections and 17 integration
knots to the encoded replay latents. It encourages the projected latent
distribution to match an isotropic Gaussian and prevents a low-rank or
collapsed representation without requiring negative examples.

## 4. Actor-Critic

The actor and critic read encoded latent states directly. They are feed-forward
networks; temporal reasoning used for learning occurs in the world model.

### 4.1 Stochastic Actor

For the two-dimensional Reacher action space, the actor is:

```text
z_t
 -> RMSNorm
 -> Dense(64), SiLU
 -> Dense(64), SiLU
 -> Dense(64), SiLU
 -> Dense(4)
 -> [mean_1, mean_2, log_std_1, log_std_2].
```

RMSNorm is applied once at the input, not after every hidden layer. The output
kernel uses scale `0.01`. Log standard deviations are clipped to
`[log(0.1), 0]`, so pre-squash standard deviations remain in `[0.1, 1.0]`.

An action is produced by sampling a Gaussian, applying `tanh`, and linearly
mapping the result to the environment bounds. Online data collection is
stochastic. Final evaluation uses `tanh(mean)` deterministically.

### 4.2 Distributional Critic

The critic is:

```text
z_t
 -> RMSNorm
 -> Dense(64), SiLU
 -> Dense(64), SiLU
 -> Dense(64), SiLU
 -> Dense(255) value logits.
```

Its final kernel is initialized to zero. Values use the same 255-bin symlog
two-hot representation and `[-20, 20]` support as rewards.

An EMA target critic is updated after every critic step with decay `0.98`. Only
the value head is delayed by EMA; there is no EMA world model or encoder.

### 4.3 Latent Imagination

Every actor-critic update samples 1,024 valid replay contexts of length eight.
Contexts containing an episode boundary are rejected. Starting from the encoded
context, the algorithm generates a 15-step imagined rollout:

1. sample an action from the actor;
2. predict the next latent, reward, and continuation with the frozen world
   model;
3. evaluate the latent with the EMA target critic;
4. append the predicted latent and sampled action to the rolling context;
5. repeat for 15 transitions.

World-model parameters are frozen during this process. The actor uses
squash-corrected REINFORCE: sampled actions are detached before entering the
world model, and the policy is updated through action log-probabilities rather
than through derivatives of learned dynamics.

### 4.4 Lambda Returns and Actor Loss

The imagined return target is

```text
G_t^lambda = r_hat_t
           + gamma * c_hat_t
             * ((1 - lambda) * V_bar(z_{t+1})
                + lambda * G_{t+1}^lambda),
```

with an EMA-critic bootstrap at the end of the 15-step rollout. The resolved
constants are:

| Parameter | Value |
| --- | ---: |
| Discount `gamma` | `1 - 1/333 = 0.996996996997` |
| Lambda | `0.95` |
| Imagined horizon | 15 |
| Return clip | `[-100, 100]` |

Each time step is weighted by predicted discounted survival:

```text
w_0 = 1
w_t = product_{i < t}(gamma * c_hat_i).
```

The actor advantage is the clipped lambda return minus the stopped EMA-critic
value. Its scale is divided by an EMA of the batch's 95th-to-5th percentile
return range. The EMA decay is `0.99`, and the divisor is never smaller than
one.

The maximized actor objective is, schematically,

```text
J_actor = weighted_mean(
    log pi(a_t | z_t) * stop_gradient((clip(G_t^lambda) - V_bar(z_t)) / S),
    w_t,
) + 0.003 * tanh_normal_entropy.
```

The tanh-Normal entropy coefficient is fixed at `3e-3` for the complete live
run; it is not scheduled or decayed.

### 4.5 Critic Loss

The critic is trained from both imagined and real trajectories:

```text
L_critic = L_imagined
         + 1.0 * L_slow_value
         + 0.3 * L_real_replay.
```

- `L_imagined` is two-hot cross-entropy against clipped imagined lambda
  returns, weighted by predicted survival.
- `L_slow_value` regularizes the online critic toward the EMA target critic on
  imagined latents.
- `L_real_replay` uses batches of 16 real sequences with horizon 64. Lambda
  returns are computed from actual replay rewards and terminal flags, with the
  EMA critic providing the bootstrap. All 64 states in each sequence train the
  critic.

There is no separate critic warmup stage.

## 5. How the Components Work Together

The encoder is owned by the world model. Actor and critic optimization never
updates the encoder or transformer. Conversely, world-model optimization never
updates the actor or critic.

| Optimizer | Trainable parameters | Frozen parameters |
| --- | --- | --- |
| World model | encoder, action encoder, transformer, latent predictor, reward head, continue head | actor, critic |
| Actor | actor head | world model, critic |
| Critic | value head | world model, actor |

The coupling is nevertheless continuous:

1. the current policy collects real transitions;
2. the world model adapts its latent space and dynamics to replay;
3. actor and critic consume the updated latent representation;
4. imagined rollouts improve the policy without additional environment calls;
5. the improved stochastic policy collects the next real-data block.

Actor and critic parameters are initialized once after the initial world-model
fit and are then carried through every online phase. The algorithm always uses
the latest policy. It does not reset policy heads online, search checkpoints,
or maintain a selected champion.

## 6. Replay and Training Schedule

### 6.1 Reset-Rich Bootstrap

Initial data is collected with uniformly random actions in a separate set of 16
environments. Each environment contributes four independently reset segments
of 80 transitions:

```text
320 transitions per environment * 16 environments = 5,120 transitions.
```

Artificial segment ends are recorded as replay cuts, not environment terminal
targets. Training sequences cannot cross a cut. Using a separate bootstrap
adapter leaves the online environments at their initial reset for the first
policy collection phase.

A separate fixed-seed validation replay contains 80 random transitions per
environment, or 1,280 transitions total. It is used only for world-model
measurement and never for gradients, replay mixing, update gating, or policy
selection.

### 6.2 Initial Fit

After bootstrap collection, the algorithm performs:

```text
1,280 world-model updates
1,280 actor-critic updates.
```

The actor and critic heads are freshly initialized immediately before this
initial actor-critic fit.

### 6.3 Interleaved Online Learning

The run then executes 481 online phases. Every phase performs:

```text
collect 64 transitions per environment = 1,024 real transitions
perform 1,024 world-model updates
perform   512 actor updates and 512 critic updates.
```

This tight interleaving updates the model and policy after every 1,024 new real
transitions rather than collecting a large offline block first.

### 6.4 Global and Recent Replay

The full replay capacity is 1,000,000 transitions. The live 500k run therefore
retains all training data and samples valid contiguous sequences uniformly from
the full history.

A second rolling buffer stores the latest 320 transitions from each of the 16
environment streams:

```text
320 * 16 = 5,120 recent transitions.
```

Recent replay is routed as follows:

| Consumer | Requested recent fraction |
| --- | ---: |
| World-model batches | 0.50 |
| Actor imagination starts | 0.00 |
| Real replay-critic batches | 0.00 |

The world-model fraction is dynamically reduced so that a recent transition is
never more than 10 times as likely to be sampled as an older transition. If
`F` is full replay size and `R` is recent replay size per environment, the
effective mixture fraction is

```text
f_WM = min(0.50, 9R / (F + 9R)).
```

At the end of the run, `F = 31,104` and `R = 320`, giving an unrounded mixture
fraction of approximately `0.0847`; batch rounding places one recent sequence
in each 16-sequence world-model batch. Recent transitions can also be drawn
through the global portion because the recent buffer is a subset of full
replay.

This arrangement lets the dynamics track policy-induced distribution shift
while preserving globally distributed actor starts and value targets.

### 6.5 Exact Training Budget

| Quantity | Count |
| --- | ---: |
| Initial training transitions | 5,120 |
| Online phases | 481 |
| Online transitions | 492,544 |
| **Total training transitions** | **497,664** |
| Held-out validation transitions | 1,280 |
| World-model updates | 493,824 |
| Actor updates | 247,552 |
| Critic updates | 247,552 |

The phase size leaves the run 2,336 transitions below the nominal 500k training
budget.

## 7. Optimization and Numerical Stabilization

All three parameter groups use separate Adam optimizer states.

| Parameter | World model | Actor | Critic |
| --- | ---: | ---: | ---: |
| Learning rate | `4e-5` | `4e-5` | `4e-5` |
| Adam epsilon | `1e-8` | `1e-8` | `1e-8` |
| Linear warmup | 1,000 updates | 1,000 updates | 1,000 updates |
| Adaptive gradient clipping | `0.3` | `0.3` | `0.3` |
| Global gradient clipping | Disabled | `10` | `100` |

The implementation uses Adam's default momentum coefficients, no weight decay,
and no dropout.

The complete stabilization stack is:

| Mechanism | Role |
| --- | --- |
| Input symlog | Compresses observation magnitudes without hard clipping. |
| SiLU and RMSNorm | Keeps activation and normalization behavior consistent across the model and policy. |
| Stopped JEPA target | Prevents the predictor branch from moving its own target within one gradient calculation. |
| SIGReg | Prevents latent collapse while retaining a decoder-free objective. |
| Five-step recurrent supervision | Trains the exact autoregressive path used in imagination. |
| Done-aware masks and replay cuts | Prevent prediction and attention across unrelated episodes or forced resets. |
| Reward and value two-hot distributions | Makes regression robust across return scales and outliers. |
| Zero-initialized reward/value outputs | Starts imagined rewards and values conservatively. |
| Small actor output initialization | Prevents initially saturated actions. |
| Bounded actor standard deviation | Retains exploration without unbounded action noise. |
| Tanh-Normal entropy | Regularizes the actual bounded action distribution. |
| Lambda returns | Combines short model rewards with critic bootstrap. |
| EMA percentile return normalization | Stabilizes policy-gradient scale as returns improve. |
| Return clipping | Bounds extreme imagined targets at magnitude 100. |
| EMA target critic | Stabilizes actor baselines, return bootstrap, and critic targets. |
| Slow-value regularization | Limits rapid drift of the online critic. |
| Real replay-critic loss | Grounds values in observed rewards and terminal flags. |
| Squash-corrected REINFORCE | Avoids backpropagating actor gradients through potentially exploitable model derivatives. |
| Reset-rich bootstrap | Covers multiple initial-state regions with a small random dataset. |
| Bounded recent world-model replay | Tracks new policy data without starving the model of global history. |
| Optimizer warmup and adaptive clipping | Reduces early and parameter-relative gradient shocks. |

## 8. Parameter Count

The exact parameter count for the six-dimensional observation and
two-dimensional action spaces of `reacher/easy` is:

| Component | Trainable parameters |
| --- | ---: |
| Observation encoder | 34,048 |
| Latent projection | 16,512 |
| Continuous action encoder | 16,896 |
| Two transformer blocks | 527,360 |
| Dynamics RMSNorm | 128 |
| Horizon embedding | 768 |
| Latent predictor and RMSNorm | 33,152 |
| Reward head | 49,407 |
| Continue head | 16,641 |
| **JEPA world model subtotal** | **694,912** |
| Actor | 16,964 |
| Critic | 33,279 |
| **Total trainable parameters** | **745,155** |

The EMA target critic introduces no additional trainable parameters. It keeps a
delayed copy of the 33,279-parameter value head. Optimizer states are excluded
from the table.

## 9. Evaluation and Reproducibility

Performance during learning is reported from episodes completed naturally by
the stochastic collection policy. These returns require no environment calls
beyond the 497,664 training transitions.

The live experiments additionally run deterministic measurement-only
evaluations every 50,000 training transitions and a deterministic 100-episode
evaluation of the final latest policy. Evaluation transitions never enter
replay, update parameters, gate training, or select a checkpoint. They are not
part of the learning algorithm and are accounted separately from training
transitions.

The return thresholds 100 and 900 are reporting labels for Reacher failure and
success rates. They do not affect rewards, losses, replay sampling, or actions.

For reproducibility, the run uses:

- isolated named RNG streams for initialization, collection, world-model
  replay, policy replay, imagination, and validation;
- deterministic accelerator reductions and highest JAX matrix multiplication
  precision;
- resolved configuration, dependency, replay, parameter, and target-critic
  fingerprints;
- recovery checkpoints every 16 online phases;
- final checkpoint reload verification.

Recovery checkpoints exist only for fault tolerance. The reported final policy
is the latest policy after the final update.
