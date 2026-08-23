# DreaMARL Single-Agent Results

## Maintained configuration

The single-agent visual representation path is fixed to:

- 64 x 64 RGB observations;
- compact DreamerV3-style convolutional encoder;
- two-layer causal Transformer dynamics;
- 32 categorical variables with 64 classes;
- EMA-target posterior and action-conditioned dynamics JEPA losses;
- 50% fixed-count masked-spatial prediction;
- cosine representation losses and SIGReg at scale `0.05`;
- Dreamer-style actor-critic learning with imagination horizon 15.

Larger visual Transformer recipes were substantially more expensive and are
not shipped as supported configurations. The public ablations retain only the
compact controls needed to isolate the maintained model's design choices.

## Final 500k evidence

The final lock-in uses three DreaMARL seeds per task and 500,000 environment
transitions. Training curves are divided into 30 equal-count environment-step
bins. Within each bin, episode returns are averaged per seed and then aggregated
across seeds with population standard deviation.

The DreamerV3 reference uses the ten official visual-DMC runs in the pinned
`external/dreamerv3` score archive with the same 30-bin, 500k truncation.

| Task | DreaMARL AUC | DreamerV3 AUC | DreaMARL final training bin | DreamerV3 final training bin | DreaMARL fixed evaluation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cheetah Run | 384.7 | 459.2 | 499.3 | 670.3 | 577.8 +/- 94.8 |
| Hopper Hop | 162.6 | 71.7 | 269.6 | 157.2 | 284.7 +/- 3.9 |
| Reacher Easy | 804.6 | 620.3 | 957.1 | 847.8 | 972.9 +/- 4.4 |
| Walker Walk | 751.4 | 793.2 | 883.9 | 942.8 | 937.3 +/- 2.1 |

The fixed-evaluation column is the mean and population standard deviation of
the three seed-level deterministic evaluation means. Each seed was evaluated
for 20 episodes at the final 500k checkpoint.

## Interpretation

The compact decoder-free model is competitive with the official DreamerV3
archive across this four-task panel. It is stronger on Hopper and Reacher under
the reported 500k AUC, while DreamerV3 is stronger on Cheetah and modestly
stronger on Walker. DreaMARL's final deterministic evaluation is high on
Reacher and Walker, stable across seeds on Hopper, and more variable on
Cheetah.

This table is strong configuration-selection evidence, not a protocol-identical
head-to-head benchmark claim. DreaMARL uses explicit deterministic final-policy
evaluation in addition to its training episodes; the DreamerV3 values come from
the upstream online-return archive. Claims about statistical superiority would
require matched evaluation, equal seeds, and uncertainty-aware significance
analysis.

## Decision

The compact CNN and fixed-count masking configuration is the authoritative
single-agent base for subsequent multi-agent work. It preserves the strongest
combination of control performance, cross-seed stability, and practical model
cost among the tested representation recipes. DreaMARL-CTDE builds on this local
execution model rather than replacing it with a larger visual backbone.
