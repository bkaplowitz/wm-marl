# JEPA online interleaving diagnostic

Date: 2026-07-12

## Question

Do long actor/critic optimization blocks on a frozen world model cause the
late-policy collapse seen with the reset-rich bootstrap?

## Design

The compared schedules use the same initial replay, total online data, total
world-model updates, total policy updates, h15 control objective, and locked
20-episode final seed pool. Only update granularity changes:

| Schedule | Phases | Collect / phase | Model updates / phase | Policy updates / phase |
| --- | ---: | ---: | ---: | ---: |
| Four-block | 4 | 256 | 4,096 | 2,048 |
| Eight-block | 8 | 128 | 2,048 | 1,024 |
| Sixteen-block | 16 | 64 | 1,024 | 512 |

All schedules use corrected tanh-Normal entropy. Training-time real policy
evaluation, checkpoint selection, championing, hard-start replay, CVaR, action
penalties, uncertainty penalties, and task-specific reward logic are disabled.

## Mechanism

Smaller blocks clearly constrain model exploitation. In the eight-block seed-1
run, action saturation stays between roughly 13% and 64% and imagined return
ends below 20, instead of saturation above 80% and imagined returns near
90-95. The sixteen-block independent runs similarly remain far below the
earlier runaway objective for most phases.

## Locked final results

| Schedule | Seed | Mean | Std | Minimum | p10 | CVaR10 | Failure | Success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 blocks | 0 | 89.35 | 182.36 | 0 | 0 | 0 | 80% | 0% |
| 8 blocks | 1 | 73.85 | 82.53 | 10 | 12.9 | 11.0 | 80% | 0% |
| 8 blocks | 2 | 72.90 | 143.49 | 0 | 0 | 0 | 80% | 0% |
| 16 blocks | 0 | 199.85 | 318.95 | 0 | 0 | 0 | 65% | 10% |
| 16 blocks | 2 | 30.70 | 39.89 | 0 | 0 | 0 | 90% | 0% |

The sixteen-block phase-16 collection means were 85.0 and 121.6 for seeds 0
and 2, but both still contained zero-return episodes before the final update.

## Conclusion

Frequent interleaving fixes runaway imagined optimization, but it does not by
itself fix cross-seed control at only 22,784 train-plus-validation transitions.
Increasing interleaving frequency again would not isolate the remaining cause.

The next gate is a fixed 99,584-transition bridge on the weak seeds using the
512-update block schedule. If returns and lower tails improve consistently with
additional online data, the remaining issue is early data scarcity and the
same schedule can be promoted to a fixed 500k run. If they do not, actor/value
calibration requires further diagnosis before spending the full budget.
