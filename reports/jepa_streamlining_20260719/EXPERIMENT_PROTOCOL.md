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

## Correctness Stages

### Explicit replay schema

Commit `5814d20` separates `is_last` and `is_terminal` in replay while
assigning both from the historical done signal. It is numerically identical to
the frozen implementation on 577 compared arrays, including optimizer
updates.

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

DMC time-limit bootstrapping is not bundled into this stage. The earlier
bundled terminal-contract candidate was rejected and cannot be promoted as
evidence for this isolated change.

### DMC time-limit bootstrap

This stage is conditional on the physical-successor stage passing. It changes
only bootstrap semantics at environment time limits:

- `is_last` remains true, so sequence and target histories stop;
- `is_terminal` follows the DMC discount, so a time limit with discount `1.0`
  does not force continuation or the real critic bootstrap to zero.

A direct `dm_control` rollout verified that `reacher/easy` reaches `LAST` at
step 1,000 with discount `1.0`. The isolated implementation is commit
`2cf6a01`; its adapter and runner tests pass.

Diagnostic: fresh 200k seeds 1 and 2, launched only after the preceding stage
passes.

Promotion gate:

- no seed-level catastrophic regression;
- mean of seed means no more than 20 points below its physical-successor
  parent;
- mean failure rate no more than 2 percentage points above the parent;
- prefer the candidate on a metric tie because it implements the environment's
  explicit bootstrap contract.

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

Only independently passing changes are combined. The combined candidate must
first pass a fresh two-seed short-budget run before any 500k launch.

The final baseline then runs:

- `dmc:reacher/easy`;
- seeds 1 and 2, predeclared as the strong and weak diagnostic seeds;
- 499,712 training-replay transitions;
- latest deterministic policy;
- fixed evaluation seeds;
- no checkpoint search or policy selection;
- training snapshots at fixed reporting milestones for future exact
  continuation diagnostics.
