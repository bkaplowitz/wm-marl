# Independent semantic review of optional recurrent CTDE training

Reviewed on 2026-09-05 against isolated implementation `d623d4d` and corrected
baseline `2663ae5`. The target recorded cross-source disabled-parity evidence in
`295aa3c`; its HEAD was `24e4dca` when this note was written. Review was read-only
for the implementation. No GPU job or policy evaluation was launched for this
review.

## Decision

No blocking loss or update-path defect was found for maintained SMAC semantics.
The optional, default-disabled implementation is suitable for integration and a
bounded production resource gate. This is not evidence that its enabled loss
improves the simulator or policy. Keep the enabled treatment isolated until
paired held-out simulator diagnostics and matched control runs establish that.

One metric correction was requested from the implementation owner:
`self_fed.py` reports `(next_alive - target_alive)^2` as `alive_brier`, but
`next_alive` is already thresholded. This is classification error. The paired
offline auditor's deployed Brier instead compares
`current_alive * alive_output.prob(1)` with the factual alive target. At predicted
probability 0.51 and target 1, these are 0 and 0.2401 respectively. Use the
probabilistic quantity for Brier, retaining a separately named classification
metric if useful. This is a reporting correction, not an objective change.

## Validated contracts

- Root state and action timing are consistent: the local cache has consumed
  observation `t`; the selected joint cache precedes the transition from `t`;
  `prevact[t+h]` drives the arrival at `t+h`. The initial joint cache followed by
  factual sequence snapshots supplies the required cache before each root.
- Joint snapshots and generated joint transitions run with `training=False`,
  matching PPO imagination. The local temporal path has no inference/training
  dropout difference. Generated categorical posteriors use `sample=True` as in
  the deployed hard-mask PPO path.
- Validity includes terminal arrival and its final reward, excludes the next
  outgoing transition, and cannot cross a reset, roster disappearance, or
  unknown batch boundary. Local endpoint losses preserve the root-live cohort
  after individual death. Team reward and continuation losses preserve every
  present root slot.
- Maintained SMAC uses `controllable_alive` for physical death while
  `agent_alive` stays true for present roster slots. The dead-head redistribution
  additionally handles generic metadata whose outer learner mask drops dead
  slots; it is not evidence of a remaining SMAC outer-mask omission.
- Generated states and factual targets are detached between steps. Auxiliary
  gradients reach the existing joint predictor and outcome heads; they do not
  reach the local dynamics, encoder, EMA encoder, actor, critics, or teammate
  modules. The optional frozen-consumer KL uses the same stopped deterministic
  state on both sides and actual unimix probabilities; its default scale is 0.
- Mask loss uses the maintained balanced reduction. Sparse loss placement
  cancels the learner's outer normalization and preserves the valid sampled
  head mean, including dead-root team heads. Each configured horizon keeps
  weight `1 / len(horizons)` even when another horizon has no valid endpoint.
- The owner's cross-source comparison found all 624 initialized tensors and
  all 922 active-PPO update/output tensors identical between the corrected
  baseline and the default-disabled optional implementation. This is stronger
  evidence than comparing two flags inside the modified implementation alone.

The existing tests cover action/reward arrival indexing, endpoint boundaries,
dead-head gradients, detached-step derivatives, parameter-group ownership,
actual learner JIT, and disabled/zero-scale parity. The nonzero-root joint-cache
gather is consistent by inspection, but the toy cache used in the alignment
tests is not a numerical independent reconstruction of an arbitrary nonzero
root. A unique-cache nonzero-root regression would strengthen future refactors.

## Material limits for interpreting an enabled run

**Roots differ from PPO roots.** The auxiliary inherits factual world-branch
features and the saved-code prefix burn-in. PPO rebuilds its behavior prefix
from current raw observations. Rebuilding the joint snapshots in inference mode
removes dropout differences, but not the local mixed-age code seam. The
auxiliary therefore matches the recurrence operator, not the full distribution
of fresh PPO roots. Require improvement in the offline auditor that rebuilds
the entire raw prefix with frozen current weights.

**Recorded actions remain factual after predicted support errors.** The
auxiliary continues using replay actions even if its current mask excludes them
or a hard false death would make deployed PPO choose no-op. Replacing those
actions would invalidate the factual labels. This produces useful corrupted
state supervision, but is not a current-policy counterfactual validation. Read
the recorded-action-outside-support metric alongside the held-out audit.

**Recurrence is detached.** H2/H4/H5 endpoint losses expose heads and the joint
predictor to their generated states, but do not propagate credit through an
earlier prediction. Hard false death remains irreversible within one rollout.
Later head supervision may improve the shared model on that input, but cannot
directly repair the preceding survival decision through a trajectory gradient.
H3 has no direct auxiliary endpoint under the initial configuration.

**Auxiliary ownership is not global freezing.** The auxiliary contributes zero
encoder/local-dynamics gradients; the ordinary world loss continues updating
those modules, and PPO remains active after its normal warmup. Interpret this
as an additional recurrent exposure objective, not a globally frozen
representation experiment.

**Eight roots do not bound the complete cost.** Before at most 40 generated
team transitions, each update rebuilds factual joint snapshots across the full
world sequence batch. At 16 teams and 64 states that is 1,008 additional factual
team transitions. Production timing and peak memory must establish cost; the
tiny compilation tests cannot do so.

**Learning metrics are not the promotion gate.** Aggregate mask disagreement
can conceal a false-positive/false-negative tradeoff. The observed failure was
strong H5 false-positive legality and liveness growth. Re-run the common-cohort
paired auditor with separate FP/FN, probabilistic alive Brier, and cumulative
return error at H1/H2/H4/H5/H8, preserving episode-clustered uncertainty and
teacher-factual baselines. Neither lower embedding loss nor lower posterior KL
alone establishes improvement in those errors or real policy return.
