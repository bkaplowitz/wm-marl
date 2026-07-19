# Streamlined JEPA Reacher/Easy 500k Baseline

## Status

This report seals the first two-seed baseline produced by the streamlined JEPA
implementation. The code-cleanup and baseline goal is complete. The experiment
does not establish cross-seed robustness as solved: seed 2 still has a weak
lower tail, which is intentionally left as the next research objective.

## Protocol

- Source branch: `jepa-streamlined-baseline-20260719`
- Source commit: `15f6230b171d8c00fcd0f34e275830eb9cf52e6d`
- Source archive SHA-256:
  `8abbd77aaf58b42818f503f11383b2c0bd7095cb21d0d3b98e5faf6843086f1d`
- Environment: DMC `reacher/easy`, proprioceptive observations
- Training seeds: 1 and 2
- Vector environments: 16
- Policy evaluation: latest deterministic policy; no checkpoint search
- Final evaluation: 100 episodes per seed with fixed evaluation seed 9,000,000
- Failure threshold: return below 100
- Success threshold: return at least 900
- Model parameters: 745,155 total
  - World model: 694,912
  - Actor: 16,964
  - Critic: 33,279

Each seed used 499,712 train-replay transitions and 1,280 held-out validation
transitions, or 500,992 train-plus-validation transitions. Curve and final
evaluations consumed another 400,000 environment transitions per seed. Those
evaluation transitions were never added to replay and were not used for policy
selection. Therefore, this is a 500k training-interaction result under the
standard convention that evaluation is excluded, and a 900,992-transition run
if every diagnostic environment interaction is counted.

The exact resolved configuration and step accounting are in
[`manifest.json`](manifest.json).

## Final Results

| Seed | Mean | Std | Failure | Success | P10 | CVaR10 | Nonfailure mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 957.18 | 34.93 | 0% | 95% | 921.90 | 880.70 | 957.18 |
| 2 | 862.52 | 270.76 | 4% | 86% | 170.00 | 90.90 | 897.36 |
| Aggregate | 909.85 | 47.33 across seed means | 2% | 90.5% | - | - | - |

The final 10k training-window means were 953.94 for seed 1 and 931.69 for
seed 2. Seed 1 cleanly reached the solved regime. Seed 2 learned a strong
policy but retained four catastrophic episodes in the fixed 100-episode
evaluation, which explains most of the remaining aggregate gap.

## Learning Curve

All entries use the latest deterministic policy and 20 fixed-seed episodes.
The 199,680 row is the separately executed prefix gate; the other rows are the
scheduled curve evaluations.

| Train steps | Seed 1 mean | Seed 1 failure | Seed 1 success | Seed 2 mean | Seed 2 failure | Seed 2 success |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50,176 | 634.70 | 10% | 30% | 317.75 | 40% | 0% |
| 100,352 | 680.65 | 25% | 50% | 741.75 | 15% | 35% |
| 150,528 | 926.80 | 0% | 75% | 769.85 | 15% | 45% |
| 199,680 | 899.08 | 1% | 62% | 793.08 | 12% | 59% |
| 200,704 | 909.60 | 0% | 70% | 697.05 | 20% | 45% |
| 250,880 | 943.05 | 0% | 85% | 719.35 | 25% | 65% |
| 300,032 | 858.55 | 5% | 85% | 895.55 | 5% | 75% |
| 350,208 | 952.50 | 0% | 95% | 879.65 | 5% | 85% |
| 400,384 | 913.15 | 5% | 90% | 893.60 | 5% | 85% |
| 450,560 | 915.55 | 5% | 95% | 826.30 | 10% | 75% |
| 499,712 final | 957.18 | 0% | 95% | 862.52 | 4% | 86% |

## Prefix Gate And Continuation

The predeclared 199,680-step gate failed. Its pair mean was 846.08 against a
required 855.135; all other checks passed. This result remains recorded as a
failure in [`prefix_gate.json`](prefix_gate.json).

Both jobs were stopped by the gate supervisor and then continued from their
complete 199,680-step training snapshots. The continuation restored model and
optimizer state, replay buffers, simulator state, and isolated RNG streams. It
used the same source, configuration, seeds, and W&B run IDs. Both continuation
processes exited with status 0, as recorded in
[`continuation_status.json`](continuation_status.json).

## Verification

- Both final checkpoints reload with maximum prediction difference `0.0`.
- Both W&B runs are marked `finished`.
- Maintained test suite: 95 passed, 1 warning.
- Ruff: clean.
- Source worktree was clean when the run manifest was produced.
- Video capture was enabled, but the headless pod lacked a usable OpenGL
  context. No video artifact was produced; numerical evaluation was unaffected.

W&B runs:

- [Seed 1](https://wandb.ai/osaze-obahor/world-marl/runs/fu0o44p0)
- [Seed 2](https://wandb.ai/osaze-obahor/world-marl/runs/vw4zalxj)

Machine-readable final metrics are in [`results.json`](results.json).
