# JEPA Architecture

This document describes the single-agent vector-control JEPA algorithm and its
two maintained training tracks: the historical high-return Reacher path and an
experimental Dreamer control-parity path for fixed-budget comparisons.

The best current launcher preset is
`dreamer_ac_online_adaptive_hard_start` in
`src/world_marl/scripts/write_dmc_vector_launcher.py`. This is the Reacher/easy
configuration family that reached stable 920+ mean return with many episodes
near 950-1000. That result used extensive real-environment checkpoint selection,
so it is a capability result rather than a clean sample-efficiency baseline.

## Goal

The world model learns action-conditioned dynamics in latent space:

```text
p(z_next, reward, continue | z, continuous_action)
```

where `z = encoder(observation)`. The model does not reconstruct observations
or pixels. Control is learned by training actor and critic heads through
imagined rollouts inside the latent world model.

## Dreamer Control-Parity Track

The presets `jepa_dreamer_parity_100k` and `jepa_dreamer_parity_500k` keep the
JEPA world model but replace the earlier control and scheduling choices with the
main reusable mechanics from the official DreamerV3 implementation. This path
is experimental until full multi-seed runs validate it.

The central change is the actor gradient. The historical path differentiates
the actor directly through imagined world-model transitions. The parity path
stops that action-to-model gradient and uses a score-function objective:

```text
advantage = lambda_return - stopgrad(slow_value)
actor_loss = -log pi(action | latent) * stopgrad(advantage / return_scale)
```

This prevents the actor from improving its objective by following gradients
through model errors. The supporting controls are:

- stochastic tanh-normal actor with entropy coefficient `3e-4`;
- EMA-smoothed p95-p5 return scale with decay `0.99`;
- lambda returns with `lambda=0.95` and effective horizon `333`;
- EMA target critic with decay `0.98` and slow-value regularization `1.0`;
- replay critic loss `0.3`, using lambda targets at every state in a length-64
  replay sequence;
- value targets clipped at `100`; raising this to `400` destabilized the critic
  and reduced the seed-1 final evaluation to `125.75`, so the clip is retained
  as a control-stability bound rather than treated as a representational limit;
- symlog inputs and 255-bin symlog two-hot reward/value targets;
- SiLU, RMSNorm, small output initialization, 1000-update warmup, and AGC `0.3`;
- latest-policy reporting with no real-environment checkpoint search, champion
  selection, hard-start/CVaR objective, or candidate-refit gate;
- online encoder updates enabled so newly collected distributions can change the
  representation.
- atomically replaced recovery parameters every five online phases, with plot,
  video, and W&B artifact failures treated as nonfatal logging errors.

The world-model architecture remains JEPA: normalized latent targets, causal
action-conditioned transformer dynamics, reward/continue heads, and SIGReg. It
does not become an RSSM and does not add an observation decoder.

| Preset | Train replay | Held-out replay | Train + held-out | WM updates | Policy updates | WM replay ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `jepa_dreamer_parity_100k` | 99584 | 1280 | 100864 | 99584 | 99584 | 1024 |
| `jepa_dreamer_parity_500k` | 496896 | 1280 | 498176 | 496896 | 496896 | 1024 |

Both use a 16x64 world-model batch, 1024 imagination starts, horizon 15, a
128-dimensional latent/model trunk with two transformer blocks, and 3x64
actor/critic MLPs. Final 20-episode evaluation is reporting overhead and is
tracked separately from training replay.

### Reproducibility Contract

The parity presets use `--isolated-rng-streams`. Initialization, world-model
updates, policy updates, replay sampling, online action sampling, validation,
and evaluation own deterministic named streams. As a result, enabling an extra
evaluation or changing the number of world-model updates no longer silently
changes later collection noise or policy minibatches. Historical runs can be
replayed with `--no-isolated-rng-streams`.

The presets also use `--deterministic-compute`: deterministic XLA GPU
reductions, a deterministic cuBLAS workspace, disabled TF32 overrides, and
highest-precision JAX matrix multiplication. GPU GEMM autotuning and Triton
GEMM selection are disabled so separate devices cannot choose different fused
kernels. This costs throughput but prevents tiny same-seed numerical differences
from being amplified by the online policy/data feedback loop.

Each run records `rng_streams.json`, stable initial/validation replay digests,
and parameter plus recent-replay digests after every online phase. Full replay
digests are recorded at recovery-checkpoint phases and at the end. Before a
configuration is compared across seeds, two identical same-seed jobs must
produce matching replay digests and either identical parameter digests or a
documented numerical tolerance. Cross-seed spread is treated as algorithmic
robustness only after that same-seed gate passes. The parity presets evaluate
every training seed on the same final environment seed (`9000000`) so policy
variance is not mixed with a different set of reporting episodes.

The `*_interleaved` parity presets are controlled robustness experiments. They
halve each online collection, world-model update, and policy update burst while
doubling the phase count. Total real transitions and optimizer updates remain
identical to the corresponding base preset, and checkpoint/video cadence stays
aligned in environment steps. This isolates feedback cadence from data budget.

W&B learning curves use `budget/train_env_steps` rather than W&B's internal
`_step` row counter. `report/episode_return` is logged at the exact cumulative
training transition where each episode finishes, while `report/return_mean` and
tail metrics summarize each collection phase. The 500k parity preset therefore
ends at 496,896 training transitions (498,176 including held-out validation),
with final evaluation interactions reported separately and excluded from the
learning-curve x-axis.

## Current Best Preset

The current reference preset is `dreamer_ac_online_adaptive_hard_start`:

| Area | Current setting |
| --- | --- |
| Environment track | DMC vector/proprioceptive control |
| Parallel envs | 16 |
| Initial random collect | 8192 vector steps, 131072 transitions |
| Validation collect | 256 vector steps, 4096 transitions |
| Offline WM updates | 12000 |
| Online iterations | 8 |
| Online collect per iteration | 6144 vector steps, 98304 transitions |
| Online validation per iteration | 256 vector steps |
| Online WM updates per iteration | 3000 |
| Online actor updates per iteration | 750 |
| WM batch size | 16 sequences |
| WM sequence length | 64 |
| Actor batch size | 1024 start states |
| Context window | 8 |
| Imagination horizon | 16 |
| Critic real-return horizon | 32 |
| Latent dim | 512 |
| Transformer dim | 512 |
| Transformer layers | 2 |
| Transformer heads | 8 |
| Dynamics ensemble heads | 5 |
| Actor MLP | 3 hidden layers, width 512, LayerNorm |
| Critic MLP | 3 hidden layers, width 512, LayerNorm |
| WM learning rate | 1e-4 |
| Actor and critic learning rate | 3e-5 |
| WM grad clip | 300 |
| Actor grad clip | 10 |
| Critic grad clip | 30 |
| Train replay steps | 917504 transitions |
| Train + validation replay steps | 954368 transitions |

The preset keeps the algorithm within the same broad DMC sample-efficiency
regime, but it is not a strict 500k-step run. For paper-style comparisons,
report the exact step accounting next to each result.

## 500k Sample-Efficiency Search

The current sample-efficiency target is:

```text
DMC reacher/easy mean return >= 920
within <= 500k training-replay environment steps
```

With `16` parallel environments, a 500k training-replay budget is about `31250`
vector steps. The launcher provides three named presets that keep the
hard-start algorithm fixed and vary only the initial-vs-online data split:

| Preset | Initial vector steps | Online phases | Online vector steps/phase | Train replay env steps | Initial share |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dreamer_ac_500k_hard_start_lean` | 1024 | 12 | 2496 | 495616 | 3.3% |
| `dreamer_ac_500k_hard_start_balanced` | 2048 | 12 | 2432 | 499712 | 6.6% |
| `dreamer_ac_500k_hard_start_coverage` | 4096 | 12 | 2256 | 498688 | 13.1% |

The first config to try is `dreamer_ac_500k_hard_start_balanced`: it avoids a
large random bootstrap, gives the world model enough initial support to start
learning, and spends most real data on the actor-induced online distribution.

For each generated launcher, `manifest.json` includes a `step_accounting`
section with:

- training-replay vector and environment steps;
- validation-replay vector and environment steps;
- train+validation totals.

Policy selection, confirmation, and final evaluation episodes are still real
environment interactions, but they are tracked separately from training replay
so we can report both Dreamer-style training curves and stricter audit totals.

## World Model

The world model has these components:

1. vector observation encoder;
2. continuous action encoder;
3. causal latent dynamics transformer;
4. residual latent predictor;
5. reward head;
6. continuation head;
7. dynamics ensemble heads.

The observation encoder is an MLP that maps vector observations to normalized
latents:

```text
z_t = E(o_t)
```

For each timestep, the dynamics stack builds a token from the current latent and
the encoded continuous action:

```text
x_t = W_z z_t + A(a_t)
```

The dynamics model is a causal transformer over latent/action history. It uses
RoPE attention, pre-norm transformer blocks, and GEGLU feed-forward blocks.
Attention is causal and masked across episode boundaries.

The latent transition is residual:

```text
z_hat_next = normalize(z_t + delta_theta(history, action))
```

The reward head uses symlog two-hot targets. Continuation uses binary cross
entropy. During imagination, predicted rewards are clipped to the valid DMC
range `[0, 1]` before actor-critic targets are built.

## JEPA Objective

The target for the latent predictor is the encoder output on the next
observation:

```text
target_z_next = stopgrad(E(o_next))
```

The latent prediction loss is cosine distance between the predicted next latent
and the target next latent. The full world-model loss is:

```text
JEPA cosine loss
+ reward prediction loss
+ continuation loss
+ SIGReg latent regularization
```

SIGReg is the anti-collapse regularizer. There is no observation decoder and no
EMA target encoder. The current implementation uses one encoder with
stop-gradient targets.

## Actor-Critic

The actor and critic operate on the same latent state used by the world model.
The current actor-critic path is Dreamer/STORM-style:

- stochastic tanh-normal actor during training;
- deterministic mean action at evaluation;
- entropy bonus with coefficient `1e-4`;
- actor MLP: 3 hidden layers, width 512, LayerNorm;
- critic MLP: 3 hidden layers, width 512, LayerNorm;
- value head trained with symlog two-hot targets;
- target critic EMA with decay `0.98`;
- lambda-return actor objective;
- value baseline for actor advantages;
- percentile normalization of imagined returns/advantages;
- lower-tail actor CVaR term with fraction `0.25` and coefficient `0.5`;
- action-bound penalty with coefficient `2.0` and bound `0.85`;
- real-replay critic auxiliary loss with coefficient `0.1`.

Actor and critic are trained through imagined latent rollouts. The world model
is frozen during actor-critic updates.

## Policy Objective

The actor objective uses imagined lambda returns from the frozen world model:

```text
G_lambda = lambda_return(reward_hat, continue_hat, V(z_hat))
advantage = percentile_normalize(G_lambda - V(z_start))
```

The actor maximizes the normalized imagined advantage plus entropy, with a
lower-tail term that puts extra pressure on poor imagined starts. The critic
learns the same lambda-return target. A soft real-replay critic auxiliary loss
is mixed into the critic update:

```text
critic_loss = imagined_value_loss
            + replay_critic_coef * real_replay_value_loss
```

## Hard-Start Replay

The current stability fix is a small failure-focused replay buffer, not a
task-specific reward hack.

After actor replay collection, episodes below the hard-start return cutoff are
stored as early-episode prefixes. During policy training:

- `50%` of actor imagination starts come from hard-start prefixes when enough
  hard starts are available;
- `50%` of replay-critic batches come from the same hard-start buffer;
- the hard-start cutoff is the bottom `30%` of collected episode returns;
- prefixes are capped at `64` transitions;
- the buffer stores up to `65536` transitions.

This targets the failure mode where the policy solves most starts nearly
perfectly but catastrophically fails a small subset of initial geometries.

## Online Learning

The online loop is:

1. collect replay with the current actor;
2. update the hard-start buffer from low-return actor episodes;
3. keep held-out recent-policy validation replay;
4. train candidate world-model refits with the encoder frozen;
5. sample each refit batch from a fixed anchor/recent mixture;
6. accept only candidate checkpoints that improve recent validation while
   keeping anchor validation degradation within tolerance;
7. continue actor-critic training in the accepted world model;
8. keep a champion actor selected by real-environment evaluation.

The encoder is frozen during online refits so the actor and critic do not lose
their latent coordinate system. The trainable online refit components are the
action encoder, transformer dynamics, latent predictor, reward head, and
continuation head.

The anchor/recent candidate-refit batch is:

```text
batch = rho * initial_random_anchor + (1 - rho) * latest_actor_replay
```

with `rho = 0.5`.

The current online acceptance setup uses:

- candidate checkpoint evaluation every 250 WM updates;
- recent validation improvement gate;
- anchor degradation gate with max degradation `0.08`;
- anchor penalty `2.0`;
- control-value consistency weight `0.1`;
- online policy trust penalty `3.0`.

## What Is Not In The Current Algorithm

These were experimental or diagnostic branches and are no longer part of the
current JEPA path:

- candidate-distillation policy objective;
- action-contrast auxiliary world-model loss;
- latent-delta scale auxiliary world-model loss;
- stratified reset-start sampler;
- sampled final-evaluation diagnostic;
- Optuna HPO launcher;
- rollout/actor-critic diagnostic CLIs;
- predictor-bottleneck diagnostic CLIs.

The only remaining "candidate" terminology refers to online candidate
world-model refits, which are part of the current algorithm.

## Step Accounting

For sample-efficiency comparisons, the most important number is real
environment interaction. The script separates:

- training replay steps;
- held-out validation replay steps;
- real policy selection/evaluation/confirmation episodes;
- strict total real environment steps.

Imagined rollouts, world-model optimizer updates, and actor-critic optimizer
updates are not real environment steps.

For paper-style comparisons, report both:

1. training-replay-only real steps, for optimistic Dreamer/STORM-style curves;
2. strict real steps including validation and policy evaluation, for full audit
   transparency.
