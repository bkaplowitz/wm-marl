# JEPA Reacher/Easy Five-Seed Baseline

This directory freezes the maintained baseline before code cleanup.

## Identity

- Commit: `a73f577e06a040caed7257880f49dc7875f6d12d`
- Preset: `jepa_500k`
- Environment: `dmc:reacher/easy`
- Seeds: 0, 1, 2, 3, 4
- Protocol: `reset_rich_interleaved_latest_policy`
- Policy selection: latest policy; no checkpoint search
- Final evaluation: 100 deterministic episodes per training seed with evaluation seed `9000000`

The exact resolved configuration and update accounting are preserved in
`manifest.json`. The five final evaluations and aggregate are preserved in
`results.json`.

## Final Results

| Seed | Mean | Episode std | p10 | CVaR10 | Failure rate | Success rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 895.94 | 223.37 | 888.70 | 331.70 | 5% | 89% |
| 1 | 954.09 | 64.50 | 925.00 | 858.40 | 0% | 97% |
| 2 | 848.00 | 277.93 | 260.00 | 47.00 | 9% | 78% |
| 3 | 936.23 | 111.89 | 899.10 | 719.30 | 1% | 90% |
| 4 | 933.27 | 162.35 | 906.40 | 586.90 | 2% | 91% |

Across the five seed-level means, the mean is `913.506` and the population
standard deviation is `37.825`. Mean failure rate is `3.4%`; mean success rate
is `89.0%`.

## Step Accounting

- Training replay: `499,712` real environment transitions
- Held-out validation replay: `1,280` real environment transitions
- Training plus validation: `500,992` transitions
- Periodic policy evaluation: `288,000` transitions
- Final policy evaluation: `112,000` transitions
- Total evaluation: `400,000` transitions

Evaluation transitions are reporting-only. They are excluded from the learning
budget and never enter replay or select a checkpoint.

## Cleanup Contract

The cleanup may change organization, naming, and implementation structure, but
must preserve:

1. the resolved `jepa_500k` algorithm configuration;
2. latest-policy training and evaluation semantics;
3. training, validation, and evaluation step accounting;
4. deterministic smoke-test trajectories and parameter initialization;
5. checkpoint and training-snapshot compatibility until an explicit migration
   is introduced.

Algorithmic simplification is evaluated only after behavior-preserving cleanup
has passed this contract.
