# Reset-rich JEPA bootstrap diagnostic

Date: 2026-07-12

## Intervention

The initial random replay can optionally reset all environments at a fixed
interval. Collector cuts prevent replay sequences from crossing these resets,
without treating the artificial boundary as an environment terminal target.

The diagnostic compares:

- contiguous: 320 random vector steps per environment;
- reset-rich: four independent 80-step segments per environment;
- 5,120 initial training transitions in both conditions;
- shared held-out replay and evaluation RNG;
- otherwise identical h15 architecture and optimization.

## Initial world-model screen

Held-out reward loss:

| Seed | Contiguous | Reset-rich |
| --- | ---: | ---: |
| 0 | 0.085042 | 0.031677 |
| 1 | 0.028594 | 0.027400 |
| 2 | 0.072431 | 0.014350 |

Aggregate metrics:

| Metric | Contiguous | Reset-rich |
| --- | ---: | ---: |
| Reward loss mean | 0.062022 | 0.024476 |
| Reward loss std | 0.024192 | 0.007369 |
| Worst reward loss | 0.085042 | 0.031677 |
| JEPA cosine mean | 0.863497 | 0.883796 |
| Open-loop cosine mean | 0.872668 | 0.893795 |

Reset-rich collection improves every seed, reduces mean reward loss by about
61%, and reduces its cross-seed standard deviation by about 70%.

## Four-phase control test

Reset-rich results after 22,784 training-plus-validation transitions:

| Seed | Phase-4 collection mean | Collection failure | Fixed final mean | Final failure | Final success |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 135.375 | 56.25% | 86.05 | 80% | 0% |
| 1 | 95.188 | 56.25% | 114.10 | 80% | 0% |
| 2 | 176.938 | 50.00% | 70.80 | 90% | 5% |

Across seeds, the phase-4 collection mean is 135.83 with a standard deviation
of 33.38. The fixed final mean is 90.32 with a standard deviation of 17.93.

The model remains calibrated after phase 4, with reward losses of 0.0044,
0.0110, and 0.0375 for seeds 0, 1, and 2. The final policy nevertheless
degrades during the last actor/critic update.

## Matched contiguous control

Seed 1 with contiguous 320-step replay also collapses:

- phase-4 collection mean: 120.375;
- fixed final mean: 75.40;
- fixed final failure rate: 80%;
- held-out reward loss: 0.000305.

Thus the control failure is not caused by artificial reset boundaries. A
four-times-broader bootstrap replay with the unchanged actor/critic schedule is
the shared factor. For reference, the earlier 80-step seed-1 replay reached a
fixed four-phase mean of 481.25 despite worse initial reward calibration.

## Conclusion

Reset-rich collection is a successful world-model robustness improvement, but
it is not yet an end-to-end control improvement. It should remain opt-in until
the actor/critic can train reliably across the broader replay distribution.

The next diagnostic should separate actor undertraining from gradient conflict:

1. keep the reset-rich world model and data fixed;
2. sweep only initial policy updates over 1,280, 2,560, and 5,120;
3. compare a uniform replay-start sampler with an easy-to-broad curriculum;
4. evaluate checkpoints with one fixed reporting set, not policy selection.
