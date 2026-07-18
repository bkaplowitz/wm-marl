# JEPA Code Inventory

This inventory defines the maintained surface after freezing the five-seed
`reacher/easy` baseline at commit
`a73f577e06a040caed7257880f49dc7875f6d12d`.

The cleanup rule is strict: code belongs in the canonical runner only when it
is exercised by that baseline, required to reproduce or inspect it, or needed
for fault-tolerant execution. A scientifically interesting alternative is not
kept merely because it could become useful in a future ablation; it can be
reintroduced in a focused change when that experiment is approved.

## Canonical Algorithm

The maintained learning path is:

1. reset-rich random bootstrap collection;
2. recurrent five-step JEPA world-model fitting;
3. stochastic tanh-Normal actor and distributional critic training in
   15-step latent imagination;
4. interleaved real collection, world-model updates, critic updates, and actor
   updates;
5. deterministic evaluation of the latest policy, without checkpoint search.

The canonical model choices are:

| Component | Maintained choice |
| --- | --- |
| Dynamics | One deterministic residual latent predictor |
| JEPA target | Shared encoder with stopped target gradient |
| Representation regularizer | SIGReg |
| Reward and value heads | Symlog two-hot distributions |
| Hidden activation | SiLU |
| Normalization | RMSNorm |
| Actor | Stochastic tanh-Normal |
| Collection | Stochastic latest-policy actions |
| Actor estimator | Squash-corrected REINFORCE |
| Return target | Lambda return with value baseline |
| Return scaling | EMA 95th-to-5th percentile range |
| Entropy | Tanh-Normal entropy |
| Critic target | EMA target critic |
| Real critic target | Lambda return over all sampled replay steps |

These choices should be represented directly in the maintained implementation,
not selected through historical mode switches.

## Current Schedule Rules

Five step-dependent rules are active in the baseline:

| Rule | Boundary | Purpose |
| --- | ---: | --- |
| Recent world-model replay ends | 50,000 | Fast early adaptation, then uniform coverage |
| Actor cadence changes from 1:1 to 1:2 | 50,000 | Reduce late policy-update variance |
| Observation encoder freezes | 101,376 | Stabilize the actor/critic latent coordinates |
| Value clip grows from 100 to 333 | 150,528-250,880 | Preserve early stability and later value resolution |
| Reset-aligned actor starts activate at 10% | 201,728 | Retain reset-state competence late in training |

These are genuine parts of the measured baseline. They are the primary
complexity target for the later component audit, but removing or combining
them before a controlled test would silently change the reference algorithm.

## Maintained Operational Surface

| File or feature | Reason retained |
| --- | --- |
| `train_dmc_jepa_bootstrap.py` | Configures deterministic accelerator behavior before JAX import |
| `train_dmc_jepa.py` | Canonical training orchestration |
| `write_dmc_vector_launcher.py` | Reproducible task, seed, GPU, and budget launch manifests |
| `eval_dmc_jepa.py` | Fixed-seed latest-policy evaluation |
| `eval_jepa_wm.py` | World-model-only predictive verification |
| `training_snapshot.py` | Exact optimizer, replay, RNG, target-critic, and simulator resume |
| W&B logging and videos | Measurement only; never influences learning |
| Periodic and final evaluations | Measurement only; never enter replay or select a policy |

## Retired Public Algorithm Branches

The following branches are absent from the five-seed baseline and should be
removed from the canonical CLI and training path:

| Retired branch | Canonical replacement |
| --- | --- |
| Multi-head dynamics ensemble | Single deterministic predictor |
| Symmetric JEPA target gradients | Stopped target gradient |
| Absolute next-latent prediction | Residual dynamics |
| No representation regularizer | SIGReg |
| Scalar MSE reward/value heads | Symlog two-hot heads |
| GELU or LayerNorm model variants | SiLU and RMSNorm |
| Deterministic actor or collection | Stochastic tanh-Normal actor and collection |
| Unsquashed Gaussian entropy | Tanh-Normal entropy |
| Reward-only actor objective | Lambda returns |
| No actor value baseline | EMA-critic value baseline |
| Batch, percentile-only, or no return scaling | EMA percentile scaling |
| Pathwise dynamics actor gradients | REINFORCE |
| Reward-only real critic target | Lambda return |
| Last-step-only real critic loss | All replay steps |

Low-level numerical helpers may remain when shared by the canonical
implementation, but the runner must not advertise these retired algorithms as
equivalent maintained modes.

## Already Removed

The first cleanup pass removed:

- five dated diagnostic report suites;
- four one-off analysis and plotting scripts;
- latent-interface and legacy world-model pass/fail diagnostics;
- diagnostic control modes and compatibility aliases;
- inactive encoder scaling, critic warmup, advantage winsorization, entropy
  scheduling, and periodic reset controls;
- the dormant cross-phase slow-policy controller;
- duplicate actor, critic, and bootstrap replay paths.

Every cleanup commit was checked against the pre-cleanup canonical step. The
latest comparison covered 196 arrays and 56 parameter leaves with maximum
absolute difference zero. The maintained CPU test suite passed 95 tests.

## Next Structural Work

1. Move the canonical configuration into one importable source of truth.
2. Make both the direct trainer and launcher consume that configuration.
3. Remove the retired public algorithm branches listed above.
4. Split collection, update scheduling, evaluation, and orchestration out of
   the monolithic trainer without changing numerical behavior.
5. Re-run exact one-step equivalence, snapshot-resume checks, the CPU suite,
   and a local smoke run.

