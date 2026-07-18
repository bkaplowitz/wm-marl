# Canonical JEPA Algorithm Audit

This audit examines the maintained `jepa_500k` algorithm after the
behavior-preserving cleanup. It asks two separate questions:

1. Is the scientific model simple and coherent?
2. Which remaining mechanisms are essential, and which are unproven protocol
   complexity?

The reference result is the fixed five-seed `dmc:reacher/easy` baseline:

- mean of seed means: `913.506`;
- standard deviation of seed means: `37.825`;
- mean failure rate: `3.4%`;
- mean success rate: `89.0%`;
- best seed: `954.09`;
- weakest seed: `848.00`.

Those results establish that the complete algorithm works. They do not, by
themselves, establish that every mechanism in the complete algorithm is
necessary.

## 1. Minimal Scientific Description

The algorithm is a decoder-free, model-based reinforcement-learning agent:

1. encode real observations into deterministic latent states;
2. predict future latent states, rewards, and continuation from latent/action
   histories;
3. train a stochastic actor and distributional critic on imagined latent
   trajectories;
4. continually refit the world model and actor-critic as the current policy
   gathers new real transitions.

The JEPA world model is trained by a stopped-target cosine loss with SIGReg.
It does not reconstruct observations and has no pixel decoder, stochastic
latent posterior, KL loss, or dynamics ensemble.

The actor is trained with squash-corrected REINFORCE over model-generated
lambda returns. The critic is trained from both imagined returns and real
replay returns. The reported policy is always the latest policy; real
evaluations never select checkpoints.

This core is conceptually simple. Most remaining complexity is in training
stabilization rather than the model architecture.

## 2. Complexity Ledger

### 2.1 Trainable model

| Component | Parameters | Share |
| --- | ---: | ---: |
| JEPA world model | 694,912 | 93.3% |
| Actor | 16,964 | 2.3% |
| Critic | 33,279 | 4.5% |
| **Total** | **745,155** | **100%** |

The EMA critic is a non-trainable copy of the 33,279-parameter value head.
There is no target encoder or second world model.

### 2.2 Learned objectives

The maintained implementation has four world-model loss terms:

1. latent cosine prediction;
2. SIGReg representation regularization;
3. symlog two-hot reward prediction;
4. continuation binary cross-entropy.

It has one actor objective with three terms:

1. normalized lambda-return REINFORCE;
2. tanh-Normal entropy;
3. full-distribution KL budget.

It has three critic terms:

1. imagined lambda-return prediction;
2. slow-value regularization to the EMA critic;
3. real-replay lambda-return prediction.

### 2.3 Step-dependent rules

Five rules change behavior at fixed training steps:

| Rule | Boundary |
| --- | ---: |
| Recent WM replay turns off | 50,000 |
| Actor cadence changes from 1:1 to 1:2 | 50,000 |
| Observation encoder freezes | 101,376 |
| Value clip expands from 100 to 333 | 150,528-250,880 |
| Reset-aligned actor starts activate at 10% | 201,728 |

This is the largest source of algorithmic complexity. The architecture is one
model; the protocol is effectively an early, middle, and mature training
curriculum.

## 3. Component Verdicts

The labels mean:

- **Core**: removing it changes the scientific identity of the algorithm.
- **Operational**: required for execution, measurement, or recovery but not
  part of learning.
- **Supported stabilization**: general mechanism with a clear failure mode and
  evidence from the development sequence.
- **Unproven complexity**: present in the successful baseline, but not isolated
  well enough to claim necessity.
- **Correctness work**: should be fixed before broad publication claims.

| Component | Verdict | Reason |
| --- | --- | --- |
| Symlog MLP observation encoder | Core | Defines the compact vector-observation latent interface. |
| Action-conditioned causal transformer | Core | Supplies temporal, counterfactual latent dynamics. |
| Residual latent prediction | Core | The maintained deterministic dynamics formulation. |
| Stopped JEPA target | Core | Defines the asymmetric predictive objective. |
| SIGReg | Supported stabilization | Prevents collapse without a decoder or negatives. |
| Reward head | Core | Makes imagined control reward-aware. |
| Continuation head | Core | Supplies imagined survival and return discounting. |
| Five-step recurrent WM supervision | Core | Trains the recurrent path used by imagination. |
| Stochastic tanh-Normal actor | Core | Current continuous-control policy. |
| Distributional critic | Core | Supplies long-horizon bootstrap beyond model rewards. |
| Fifteen-step latent imagination | Core configuration | Successful baseline horizon; its exact value is a hyperparameter, not a separate algorithm. |
| Lambda returns | Core | Joins model rewards and critic bootstrap. |
| Squash-corrected REINFORCE | Core | Avoids actor exploitation of model derivatives. |
| EMA percentile return scale | Supported stabilization | Controls score-function gradient scale across learning. |
| Tanh-Normal entropy | Supported stabilization | Preserves bounded-action exploration. |
| Full-distribution KL budget | Supported stabilization | Limits abrupt mean and variance changes. |
| EMA target critic | Supported stabilization | Stabilizes bootstrap and actor baseline. |
| Real-replay critic loss | Supported stabilization | Grounds critic values in observed reward sequences. |
| Slow-value regularization | Unproven complexity | Plausible and active, but overlaps conceptually with the EMA target and real critic anchor. |
| Reset-rich bootstrap | Supported stabilization | General reset-state coverage with only 5,120 random transitions. |
| Uniform full replay | Core data path | Preserves all training history under the 500k budget. |
| Early recent WM replay | Unproven complexity | Plausible early adaptation benefit, but necessity is not isolated in the five-seed baseline. |
| Actor cadence switch | Supported stabilization | Development runs associated slower late actor updates with fewer policy collapses. |
| Hard encoder freeze | Supported but coarse | Addresses moving latent coordinates, but the exact threshold is a hand-set time rule. |
| Scheduled value clipping | Supported but coarse | Trades stable early updates for later value resolution; exact thresholds are hand-set. |
| Delayed reset-aligned starts | Supported but coarse | Reward-agnostic defense against forgetting episode starts; exact activation time is hand-set. |
| Held-out random validation replay | Operational | Measures WM quality and never affects gradients or selection. |
| Periodic deterministic policy eval | Operational | Produces a learning curve and never affects training. |
| Recovery snapshots and hashes | Operational | Fault tolerance and reproducibility only. |
| Failure/success thresholds | Operational | Reporting labels only; they never affect learning. |

## 4. What Is Already Simple

The maintained algorithm does **not** contain:

- checkpoint or champion search;
- real-evaluation policy selection;
- failure-conditioned replay;
- reward-conditioned hard-start buffers;
- Reacher geometry features;
- task-specific reward shaping;
- CVaR actor objectives;
- a dynamics ensemble;
- a target encoder;
- a decoder or reconstruction objective;
- pathwise actor gradients through the world model;
- multiple selectable actor or world-model algorithms.

This matters for the publication claim. The current policy is produced by one
online learning path, and the lower-tail mechanisms use only generic episode
boundaries and training progress.

## 5. Remaining Concerns

### 5.1 Replay still conflates episode boundaries and terminal bootstrap

Replay stores `dones` and collector `cuts`. Cuts correctly prevent sampling
across forced resets without creating a terminal target. However, `dones`
still serves two distinct roles:

1. sequence/attention boundary (`is_last`);
2. zero-bootstrap target (`is_terminal`).

DMC time-limit endings can be episode boundaries without being environmental
terminal states. The adapter currently turns every `LAST` step into
`done = 1`, so the continuation head and real critic both receive a
zero-continuation target at such boundaries.

This is a general replay-contract issue, not Reacher reward engineering. The
publication implementation should represent at least:

```text
is_last      # sequence ends or environment resets
is_terminal  # Bellman continuation must be zero
cut          # collector-imposed boundary with no terminal target
```

Sequence masks should use `is_last OR cut`; continuation and Bellman targets
should use `is_terminal`.

### 5.2 Schedule thresholds are tied to the 500k budget

The active boundaries correspond approximately to 10%, 20%, 30-50%, and 40%
of the 500k run. The 100k and 200k presets currently inherit the same absolute
steps, so they do not execute a proportionally equivalent algorithm.

This weakens the "one general algorithm" story more than the number of model
parameters does. A cleaner protocol would express schedule boundaries as
training-budget fractions or define a small number of named training stages.
The 500k behavior can remain exactly unchanged while shorter budgets become
well-defined.

### 5.3 The actor standard-deviation bound has dead gradients

Actor `log_std` is bounded with a hard `clip`. Outputs outside the interval
receive zero local gradient. A smooth bounded parameterization can preserve
the exact `[0.1, 1.0]` standard-deviation range without dead regions. This is
a general numerical correction and a plausible source of seed sensitivity.

It should still be treated as an algorithmic change and tested against the
fixed baseline.

### 5.4 Three critic stabilizers may overlap

The critic simultaneously uses:

- an EMA target critic;
- slow-value regularization toward that EMA critic;
- a real-replay lambda-return loss.

All are defensible, but the complete baseline does not establish that all
three are needed together. Removing one may simplify the algorithm without
losing stability. This should be tested only after the replay terminal contract
is corrected, because incorrect terminal targets can make critic ablations
misleading.

### 5.5 Capacity is not the first explanation for the lower tail

The same 745k-parameter model reaches `954.09` with `97%` success on seed 1.
That proves it has enough capacity to solve Reacher/easy. Seed 2's `848.00`
mean and `9%` failure rate are therefore stronger evidence for optimization,
data-distribution, or boundary-target sensitivity than for a universal
capacity ceiling.

Larger networks remain relevant for harder tasks, but model scaling is not the
first intervention for the Reacher lower tail.

## 6. Simplicity Verdict

The scientific architecture is simple enough to explain cleanly:

```text
decoder-free JEPA dynamics + reward/continue prediction
                         +
distributional actor-critic in latent imagination
```

The implementation no longer contains a maze of alternate algorithms. It is
not yet a minimal training protocol, because five timed rules and three critic
stabilizers remain active. Those mechanisms are generic and reward-agnostic,
but several have not been isolated strongly enough to claim that they are
necessary.

The correct next move is not another broad hyperparameter sweep. It is:

1. fix the terminal/boundary replay contract;
2. express the existing schedule coherently relative to budget;
3. test the smallest number of high-value simplifications against the fixed
   five-seed baseline;
4. only then pursue performance changes.
