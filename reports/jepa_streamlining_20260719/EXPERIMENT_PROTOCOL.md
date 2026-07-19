# JEPA Streamlining Protocol

## Objective

Starting from the frozen five-seed Reacher/easy baseline, retain only general
changes that either correct a demonstrable implementation defect or preserve
end-to-end control quality while simplifying the maintained algorithm.

No experiment in this protocol uses task geometry, reward shaping, failure
labels for training, checkpoint selection, or real-environment evaluation to
choose a policy.

## Frozen Reference

The behavior reference is the cleaned canonical algorithm at `5d55bd6`, whose
learning behavior is identical to the five-seed baseline at `a73f577`.

Five-seed 500k result:

| Metric | Value |
| --- | ---: |
| Mean of seed means | 913.506 |
| Population standard deviation of seed means | 37.825 |
| Mean failure rate | 3.4% |
| Mean success rate | 89.0% |
| Weakest seed mean | 848.00 |
| Best seed mean | 954.09 |

The fixed 200k diagnostic controls use seeds 1 and 2:

| Seed | Mean | Failure | Success | P10 | CVaR10 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 808.67 | 13% | 73% | 0.0 | 0.0 |
| 2 | 921.60 | 1% | 74% | 854.1 | 733.4 |

All candidate manifests preserve the same replay budget, update counts,
evaluation seeds, and latest deterministic policy protocol.

## Audit Disposition

The review findings were checked against the canonical path before defining
experiments:

- ensemble-loss normalization and ensemble uncertainty gradients do not apply,
  because the maintained model has no ensemble or uncertainty actor term;
- champion EMA restoration does not apply, because training always retains and
  reports the latest policy rather than selecting or restoring a champion;
- squash-corrected tanh-Normal entropy, a valid two-hot constant-prediction
  baseline, explicit collector cuts, actor/critic global gradient clips, and
  adaptive gradient clipping are already implemented and tested;
- actor return updates already use an EMA percentile scale and a full Gaussian
  KL budget;
- the encoder freeze keeps one optimizer state and masks gradients rather than
  replacing the optimizer.

These mechanisms are retained. Re-running removed alternatives would not
answer an open question about the current algorithm.

## Correctness Stages

### Explicit replay schema

Commit `5814d20` separates `is_last` and `is_terminal` in replay while
assigning both from the historical done signal. It is numerically identical to
the frozen implementation on 577 compared arrays, including optimizer
updates.

Decision: retain.

### Mandatory independent RNG streams

The maintained manifests already required independent JAX and NumPy streams
for initialization, model fitting, policy fitting, replay sampling,
collection, and evaluation. Commit `b7921b9` removes the unreachable shared
stream implementation while retaining the hidden positive CLI spelling needed
to replay existing resolved manifests. The default and canonical numerical
paths are unchanged.

Verification: all 90 maintained JEPA model, replay, DMC adapter, launcher,
snapshot, evaluator, W&B, and CLI tests pass.

Decision: retain.

### Physical reset-boundary successor

Commit `2c46ada` retains the historical finite-episode bootstrap convention but
uses the physical post-action observation as the immediate JEPA target when an
adapter auto-resets. All non-boundary calculations are numerically identical
to `5814d20`.

Diagnostic: fresh 200k seeds 1 and 2.

Promotion gate:

- no seed-level catastrophic regression;
- mean of seed means no more than 20 points below control;
- mean failure rate no more than 2 percentage points above control;
- prefer the candidate on a metric tie because it removes an invalid
  reset-observation target.

Result: rejected.

| Metric | Frozen control | Physical successor |
| --- | ---: | ---: |
| Mean of seed means | 865.14 | 839.30 |
| Mean failure rate | 7.0% | 8.5% |
| Mean P10 | 427.05 | 434.75 |
| Mean CVaR10 | 366.70 | 273.25 |
| Normalized curve area | 488.13 | 581.47 |

The candidate learned earlier on average, but its final mean was 25.84 points
below control, exceeding the allowed 20-point margin. Commit `2c46ada` was
therefore reverted from the canonical branch.

DMC time-limit bootstrapping is not bundled into this stage. The earlier
bundled terminal-contract candidate was rejected and cannot be promoted as
evidence for this isolated change.

### DMC time-limit bootstrap

The terminal contract is tested independently of the rejected physical
successor change. It changes only bootstrap semantics at environment time
limits:

- `is_last` remains true, so sequence and target histories stop;
- `is_terminal` follows the DMC discount, so a time limit with discount `1.0`
  does not force continuation or the real critic bootstrap to zero.

A direct `dm_control` rollout verified that `reacher/easy` reaches `LAST` at
step 1,000 with discount `1.0`. Commit `f5a1bc7` applies the isolated terminal
semantics directly to the accepted replay schema; 90 focused adapter, replay,
runner, and launcher tests pass.

Diagnostic: fresh 200k seeds 1 and 2.

Promotion gate:

- no seed-level catastrophic regression;
- mean of seed means no more than 20 points below control;
- mean failure rate no more than 2 percentage points above control;
- prefer the candidate on a metric tie because it implements the environment's
  explicit bootstrap contract.

Result: pending.

## General Numerical Fixes

### Budget-relative milestones

Commit `4e8b444` scales the existing 500k milestones by declared training
budget. The resolved 500k thresholds remain exactly unchanged. This is a
protocol consistency fix and does not alter the target 500k baseline.

Diagnostic: fresh 200k seeds 1 and 2 compare the historical absolute
milestones with the same milestones at their 500k-relative progress. This
diagnostic changes the milestone bundle as one protocol-level intervention;
individual schedule rules are not promoted or rejected from this comparison.

Promotion gate:

- the 500k resolved manifest remains exact;
- no seed-level catastrophic regression at 200k;
- improved or equal mean, lower tail, and area under the fixed-evaluation
  curve favor the proportional schedule;
- if proportional scheduling hurts, keep the exact 500k algorithm and remove
  the misleading maintained short-budget preset rather than reporting it as
  the same algorithm.

### Smooth actor scale

Commit `5a621c2` is an isolated diagnostic based on the explicit replay schema.
It replaces hard clipping with a smooth map to the same standard-deviation
range `[0.1, 1.0]`, without changing parameters or policy family.

Motivation from the frozen 200k controls:

- both seeds have final `action_log_std_max = 0.0`, exactly at the hard bound;
- outputs beyond that bound receive zero scale gradient in the frozen actor.

Diagnostic: fresh 200k seeds 1 and 2.

Promotion gate:

- no seed-level catastrophic regression;
- mean of seed means no more than 10 points below control;
- mean failure rate no worse than control;
- reduced exact-bound occupancy and finite actor metrics;
- prefer the candidate on a return tie because it removes gradient-dead
  regions.

Result: rejected.

| Metric | Frozen control | Smooth scale |
| --- | ---: | ---: |
| Mean of seed means | 865.14 | 826.43 |
| Seed-mean population std | 56.47 | 23.29 |
| Mean failure rate | 7.0% | 4.5% |
| Mean success rate | 73.5% | 71.0% |
| Mean P10 | 427.05 | 374.55 |
| Mean CVaR10 | 366.70 | 129.25 |
| Normalized curve area | 488.13 | 637.15 |

The smooth parameterization learned substantially earlier and reduced
cross-seed dispersion and failure rate. Its actor metrics stayed finite and
strictly inside both standard-deviation bounds. However, its final pair mean
was 38.71 points below control, exceeding the allowed 10-point margin, and its
final lower tail was worse. The fixed `+2` smooth-map offset changed the
effective exploration distribution rather than acting as a behavior-neutral
numerical correction. Commit `6ef6a3f` was therefore reverted from the
canonical branch.

## Critic Simplification

The canonical critic uses imagined return prediction, slow-value
regularization, and a real-replay critic loss. In the frozen 200k controls,
the final losses are:

| Seed | Imagined | Slow value | Real replay | Total |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.579 | 0.509 | 0.872 | 1.350 |
| 2 | 0.567 | 0.520 | 0.731 | 1.306 |

The slow-value term is therefore active and material, rather than a dormant
configuration field.

Diagnostic: set only `policy_slow_value_regularization_coef` from `1.0` to
`0.0`, then run fresh 200k seeds 1 and 2.

Promotion gate:

- no seed-level catastrophic regression;
- mean of seed means no more than 10 points below control;
- mean failure and lower-tail metrics no worse than control;
- a statistical tie removes the term because the EMA critic remains the actor
  baseline and lambda-return bootstrap.

The EMA critic is not removed in this stage.

Result: rejected.

| Metric | Frozen control | No slow-value term |
| --- | ---: | ---: |
| Mean of seed means | 865.14 | 626.69 |
| Seed-mean population std | 56.47 | 39.98 |
| Mean failure rate | 7.0% | 27.5% |
| Mean success rate | 73.5% | 52.5% |
| Mean P10 | 427.05 | 2.70 |
| Mean CVaR10 | 366.70 | 0.00 |
| Normalized curve area | 488.13 | 424.44 |

Removing the term reduced the pair mean by 238.45 points, increased failure
rate by 20.5 percentage points, and collapsed the lower tail. The slow-value
regularizer is therefore a necessary part of the maintained critic rather than
removable complexity.

## Early Replay Simplification

The 500k baseline mixes 50% recent data into world-model batches before 50k
training transitions, then uses uniform replay. This rule is plausible but was
not isolated in the five-seed baseline.

Diagnostic: set only `online_recent_world_model_fraction` from `0.5` to `0.0`
under the historical 200k schedule, then run fresh seeds 1 and 2.

Promotion gate:

- no seed-level catastrophic regression;
- equal or better mean, lower tail, and fixed-evaluation area under the curve;
- a statistical tie removes the recent replay and its activation threshold,
  because uniform replay is the simpler data path.

## Combination Gate

Only independently passing changes are combined. The combined candidate is
launched directly with the final 500k manifest, but both jobs are held at the
matched nominal-200k phase boundary of 199,680 training transitions. Their
fixed 50k/100k/150k evaluations and 199,680 endpoint are compared with the
frozen controls before the jobs are allowed to continue.

This makes the gate an exact prefix of the final experiment. It avoids a
throwaway 200k run whose budget-relative schedule would differ from the first
200k transitions of the 500k algorithm. No optimizer, replay, RNG, simulator,
or policy state is reset after the gate.

The 199,680-transition prefix must satisfy:

- minimum seed mean at least 500, ruling out a seed-level catastrophe;
- mean of seed means at least 855.135, exactly 10 points below the frozen
  200k control mean;
- mean failure rate no more than 7%;
- mean P10 at least 300 and mean CVaR10 at least 250;
- normalized fixed-evaluation curve area at least 439.317, 90% of the frozen
  control area;
- finite world-model, actor, and critic metrics;

If the prefix fails, both jobs are terminated and no 500k result is reported
for that candidate. If it passes, the same processes continue uninterrupted.

The final baseline then runs:

- `dmc:reacher/easy`;
- seeds 1 and 2, predeclared as the strong and weak diagnostic seeds;
- 499,712 training-replay transitions;
- latest deterministic policy;
- fixed evaluation seeds;
- no checkpoint search or policy selection;
- training snapshots at fixed reporting milestones for future exact
  continuation diagnostics.
