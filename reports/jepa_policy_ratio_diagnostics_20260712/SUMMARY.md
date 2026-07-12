# JEPA policy update-ratio diagnostic

Date: 2026-07-12

## Question

Does the reset-rich 5,120-transition bootstrap fail because the actor and
critic receive too few initial updates for the broader replay distribution?

## Controls

All runs use the exact same seed-1 reset-rich replay (SHA-256
`94ed2126bb38d892f09661941242e86195200de207c9811a57502857381a3e00`),
world-model configuration, named RNG streams, four online phases, h15
imagination, and locked 20-episode final evaluation. Training-time real
environment evaluation, checkpoint selection, championing, hard-start replay,
CVaR weighting, and task-specific interventions are disabled.

Only the number of initial actor/critic updates changes.

## Results

| Initial policy updates | Phase-4 collection mean | Final mean | Final std | Final failure | Final success | Nonfailure mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,280 | 95.19 | 114.10 | 248.67 | 80% | 0% | 539.50 |
| 2,560 | 104.19 | 71.05 | 218.52 | 85% | 5% | 469.33 |
| 5,120 | 105.19 | 14.70 | 29.85 | 95% | 0% | 102.00 |

Every run has final return p10 and CVaR10 equal to zero. The 2,560- and
5,120-update policies reach phase-1 action saturation above 98% while their
imagined returns grow rapidly. At the end of training, their imagined returns
are about 88 and 95 despite fixed real means of only 71 and 15.

## Conclusion

The broader replay is not failing because the policy is undertrained. More
initial policy optimization monotonically worsens fixed final robustness and
increases the gap between imagined and real performance.

The next controlled diagnostic isolates two general mechanisms:

1. include the tanh action-squashing Jacobian in the stochastic actor entropy;
2. halve online actor/critic updates while preserving every data and model
   setting.

Neither change uses reward thresholds, environment geometry, adaptive real
evaluation, or task-specific replay weighting.
