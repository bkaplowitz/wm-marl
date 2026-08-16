# DreaMARL Visual Lock-In

## Maintained configuration

The maintained visual representation path is:

- 64 x 64 RGB observations;
- DreamerV3 convolutional encoder;
- causal Transformer dynamics;
- decoder-free EMA-target JEPA training;
- posterior, action-conditioned dynamics, and masked-spatial prediction;
- cosine representation losses and SIGReg at 0.05;
- exactly 50% fixed-count spatial masking;
- the unchanged DreamerV3 actor-critic with imagination horizon 15.

The faithful 224 x 224 V-JEPA encoder and multi-block masking recipe remain an
explicit research variant. They are not part of the maintained configuration.

## Screening results at 100k environment steps

| Task | DreaMARL mean | DreaMARL median | DreaMARL std | DreaMARL AUC | DreamerV3 mean | DreamerV3 AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reacher Easy | 721.7 | 960.0 | 389.8 | 555.7 | 376.2 | 279.3 |
| Hopper Hop | 98.4 | 98.8 | 4.2 | 34.8 | 10.1 | 1.2 |
| Walker Walk | 627.6 | 655.3 | 78.7 | 475.3 | 604.6 | 350.4 |

DreaMARL values are 10-episode deterministic evaluations from seed 0 at fixed
20k checkpoints. DreamerV3 values are computed from the ten official visual-DMC
runs in `external/dreamerv3/scores/dmc_vision-dreamerv3.json.gz`; the endpoint
is the mean episode return in the 90k-100k bin and AUC uses 10k online-return
bins. The table is therefore a promotion screen, not a protocol-identical paper
comparison.

## DreamerV3 longer-budget reference

| Task | 500k mean | 1M mean |
| --- | ---: | ---: |
| Reacher Easy | 820.5 | 930.4 |
| Hopper Hop | 157.3 | 230.0 |
| Walker Walk | 945.0 | 961.7 |

## Decision

The fixed-count CNN configuration is promoted because it combines the strongest
early aggregate performance with substantially lower compute cost and better
cross-task stability than the tested ViT variants. At 100k it is competitive
with the official DreamerV3 visual-DMC archive on all three screening tasks.
The next scientific gate is a protocol-matched multi-seed run, not another
encoder or masking-topology search.
