# JEPA actor objective diagnostic

Date: 2026-07-12

## Interventions

This diagnostic isolates two task-agnostic changes on the exact same seed-1
reset-rich replay:

1. `tanh-normal` entropy includes the action-squashing Jacobian, whereas the
   historical `gaussian` mode measures only pre-squash entropy;
2. online actor/critic updates are reduced from 4,096 to 2,048 per phase.

All other model, replay, RNG, online-data, and locked final-evaluation settings
are fixed. No real-environment policy selection or task-specific intervention
is enabled.

## Mechanism

At the initial 1,280-update actor checkpoint, corrected entropy reduces action
saturation from 24.5% to 16.2% without materially changing imagined reward or
return. However, long online policy blocks can still overcome that pressure:
the corrected-entropy policy reaches 87.7% saturation and imagined return 54.7
by the end of its first 4,096-update block.

## Locked final results

| Entropy | Online policy updates | Mean | Std | p10 | CVaR10 | Failure | Success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gaussian | 4,096 | 114.10 | 248.67 | 0 | 0 | 80% | 0% |
| Gaussian | 2,048 | 4.25 | 14.77 | 0 | 0 | 100% | 0% |
| Tanh-Normal | 4,096 | 123.45 | 239.94 | 0 | 0 | 80% | 0% |
| Tanh-Normal | 2,048 | 136.45 | 321.38 | 0 | 0 | 85% | 10% |

## Conclusion

Squash-aware entropy is the mathematically consistent estimator and modestly
improves the initial action distribution, so it remains available as an
explicit configuration. It is not sufficient to solve lower-tail failures.
Reducing total policy updates alone is also not sufficient and can worsen the
latest policy.

The next isolated mechanism is update granularity: refresh online data and the
world model before the actor can take thousands of consecutive steps against a
frozen model.
