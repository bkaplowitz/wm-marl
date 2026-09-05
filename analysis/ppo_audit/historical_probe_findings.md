# Historical frozen-representation probes — 5 September 2026

These are historical references, not outcomes of the new PPO correction screen.
Both checkpoints are at 50,000 environment transitions, training seed 234:

- 8m, commit 1aeb694, final 83/128 wins (64.84%).
- 2s3z, commit 976ce62, coupled multistep JEPA action scale 0.25, final 33/128 wins (25.78%).

The local dynamics module is byte-identical to the current source. The encoder
module differs only in its opening documentation string. Each probe rebuilds
raw complete episodes with frozen current checkpoint weights. Forty-eight
randomly selected complete episodes are split 60/20/20 by episode, with up to
eight sampled states per episode. Ridge regularization is chosen on validation
and evaluated once on test. No actor training or environment transitions occur.

| Map | Frozen input | H1 return R² | H5 return R² | H15 return R² |
| --- | --- | ---: | ---: | ---: |
| 8m | raw | 0.617 | 0.500 | 0.290 |
| 8m | encoder | 0.547 | 0.703 | 0.335 |
| 8m | posterior_sample | 0.567 | 0.782 | 0.576 |
| 8m | posterior_mean | 0.581 | 0.785 | 0.590 |
| 2s3z | raw | 0.173 | 0.482 | 0.562 |
| 2s3z | encoder | 0.128 | 0.532 | 0.629 |
| 2s3z | posterior_sample | -0.209 | 0.556 | 0.682 |
| 2s3z | posterior_mean | -0.171 | 0.564 | 0.688 |

The history-based posterior improves longer-horizon factual-return prediction on
both maps relative to raw observations. Encoder features also retain much of the
current observation: median nonconstant-coordinate R² is 0.842 on 8m and 0.864
on 2s3z. These results do not support a generic claim that the JEPA encoder has
collapsed or discarded all reward-relevant information.

They do not prove control sufficiency. Linear decodability does not validate
counterfactual action ranking, recurrent simulated states, policy readout, or
unobserved information. The models, maps, replay distributions, and policy
settings differ, and this is only one episode split. The one-step 2s3z posterior
probe performs worse than a constant predictor, despite good H5/H15 prediction.
That could reflect immediate reward uncertainty or poor decodability; it is not
by itself proof that the representation lacks information.

Use matched fresh correction checkpoints and the paired simulator/value audits
before promoting the optional value-supervision experiment. Full per-target
metrics, chosen regularization, checkpoint paths, and episode identities are
preserved in historical_8m_seed234_probe.json and
historical_2s3z_seed234_probe.json.
