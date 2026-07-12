# JEPA bootstrap crossover diagnostic

Date: 2026-07-12

## Question

Does the early Reacher instability follow the initial replay data or the model
initialization seed?

## Controls

- DMC `reacher/easy`
- deterministic compute and isolated RNG streams
- 1,280 initial world-model updates
- one shared validation environment seed, replay, sequence sample, and JAX key
- identical architecture and optimizer settings from the h15 100k run
- initial replay is the only variable in the online comparison

## Initial-fit crossover

Held-out reward loss:

| Initialization | Seed-0 replay | Seed-1 replay |
| --- | ---: | ---: |
| Seed 0 | 0.545916 | 0.244546 |
| Seed 1 | 0.643324 | 0.125946 |

Open-loop cosine remained between 0.849 and 0.858 in all four cells. The
seed-1 replay improved reward generalization under both initializations.

## Replay structure

| Replay | Positive steps | Positive environments | Reward-boundary flips | Longest positive occupancy |
| --- | ---: | ---: | ---: | ---: |
| Seed 0 | 106 | 2 | 7 | 71 / 80 |
| Seed 1 | 51 | 2 | 13 | 26 / 80 |
| Seed 2 | 136 | 3 | 8 | 79 / 80 |

The weak replays contain more positive rewards, but those rewards are clustered
inside nearly stationary positive trajectories. The strong replay contains
almost twice as many transitions into or out of reward, which better identifies
the action-conditioned reward boundary.

## Four-phase online crossover

Both runs use seed 1 for initialization, optimization, online collection, and
final evaluation. Only the loaded initial replay differs.

| Metric | Seed-0 replay | Seed-1 replay |
| --- | ---: | ---: |
| Phase-4 collection mean | 87.6875 | 168.6250 |
| Phase-4 collection failure rate | 75% | 62.5% |
| Phase-4 held-out reward loss | 0.135038 | 0.006774 |
| Phase-4 open-loop cosine | 0.958944 | 0.967432 |
| Fixed final evaluation mean | 59.40 | 481.25 |
| Fixed final evaluation std | 111.24 | 414.20 |
| Fixed final failure rate | 85% | 35% |
| Fixed final success rate | 0% | 40% |

Training plus validation used 18,944 real transitions per run. The fixed final
evaluation was not used for training or selection.

## Conclusion

The early robustness failure follows bootstrap replay composition, not hidden
GPU nondeterminism and not a universally good or bad parameter initialization.
Latent dynamics learn in every cell, while reward calibration depends strongly
on whether the bootstrap contains diverse reward-boundary transitions. The
resulting reward model changes the actor's early exploration regime, and the
online feedback loop amplifies that difference.

## Next targeted test

Use a reset-rich random bootstrap that exposes more independent initial
geometries and reward transitions while preserving the total data budget as
closely as possible. Compare it against the current contiguous 80-step
bootstrap over several collection seeds before changing the model or RL loss.
