# Review of the self-fed CTDE diagnostic in 2663ae5

The report exercises the simulator PPO consumes and is useful for detecting
deployment-path failures. Its action and cache alignment is correct. Its scalar
curves need the qualifications below before they justify a new training loss.

## Confirmed alignment and support

- A local root contains the posterior/cache after observing state t. The joint
  root uses the snapshot after source t-1, or the initial joint cache at t=0.
- `prevact[t+1]` is factual action a_t. The joint transition predicts embedding,
  reward, continuation, availability and liveness at t+1; local `advance`
  consumes a_t before completing the posterior from that embedding.
- Reset crossings invalidate all subsequent horizon metrics. Terminal-transition
  reward remains included. Shared outcomes cover the present roster, including
  dead focal slots; local metrics use the factual source-controllable mask.
- Predicted masks include the deployed threshold, empty-mask no-op fallback,
  and predicted-death no-op override. This is the correct hard-mask path for
  the supported PPO configuration. The legacy soft-mask route is not supported
  by the current PPO constructor and should not be interpreted as validated.

## Interpretation limits

1. **The report history is not the exact current PPO history.** Report replay
   `_apply_replay_context` rebuilds the prefix using saved latent pairs and
   stochastic samples from collection. The suffix is recomputed with current
   parameters, but its temporal context still contains those older codes. PPO's
   behavior replay path re-encodes raw observations throughout the prefix.
   A checkpoint-level audit should reconstruct complete raw episodes from reset.
2. **Sampling and approximation error mix after the first step.** The report's
   imagined posterior draws are independent of factual replay draws. Even an
   exact observation predictor can therefore produce a different deterministic
   history after H1. This is a deployment-path sample error, not purely model
   approximation error. Couple categorical draws by target timestep in the
   factual, self-fed, and oracle-observation paths for causal comparisons.
3. **The displayed KL is between raw logits.** The executable latent categorical
   distribution mixes in 0.01 uniform probability. Raw-softmax KL overstates the
   actual mixed-distribution KL, especially on small target probabilities. Use
   the mixed distributions and state whether KL is summed over the 32 variables.
4. **The compared cohorts change with horizon.** Four roots per report plus
   reset/death-dependent support can produce a changing selection of states.
   A zero count emits zero-valued metrics, not evidence of perfect prediction.
   Compare a common root cohort with full factual futures, report counts, and
   aggregate at least 64 roots before acting on a curve.
5. **Cosine and KL answer different questions.** Cosine uses EMA embeddings;
   the actor consumes online embeddings. The cumulative posterior KL compares
   predicted history with factual history and therefore combines interface and
   temporal drift. Add online-embedding similarity and teacher-forced posterior
   KL at the same factual next deterministic state.
6. **Liveness “Brier” is a deployed-path error.** After a thresholded predicted
   death, subsequent predicted survival is zero. It is not a calibrated marginal
   survival probability over stochastic death paths. Keep the label explicit.
7. **The actions are factual, not current-policy sampled.** This is a necessary
   controlled dynamics test, but it cannot establish counterfactual action-value
   accuracy or absence of policy exploitation away from replay support.

No parameters or loss terms were changed during this review.

## Offline audit outcome semantics and cumulative returns

`scripts/audit_causal_simulator.py` requires an explicit interpretation of the
deployed outcome path. Its `--outcome-semantics corrected_team` default pools
present roster slots, preserves final transition reward, and guards all-dead
absorbing tails. Use `--outcome-semantics original_slots` for the unmodified
`92014f1` control: that version consumed each slot's raw reward and continuation
head directly. Applying corrected pooling to its checkpoint would silently
audit a different simulator. Every JSON records the selected mode and formula.

The cumulative-return error measures a finite sum of predicted rewards weighted
by preceding predicted continuation products against the factual discounted
reward sum. It has no critic bootstrap, lambda mixing, GAE, or focal-death actor
mask. In particular, selecting `original_slots` reproduces original simulator
outcomes but does not reproduce old GAE's additional credit cutoff at focal unit
death. These return metrics diagnose simulator accuracy; they do not reconstruct
the advantages on which the original PPO actor trained.

## Historical evidence that should prevent repeated experiments

The retained local files
`/Users/osaze/MARL/world_marl_ctde_diagnostics/results/ctde_diagnostics/2s3z_seed0_100k.json`
and the adjacent 3m file contain four-batch, older REINFORCE diagnostics. They
already measured recursive joint/posterior drift, although their rollout used
`sample=False` and their root support differs from the new report.

| Older 2s3z metric | H1 | H4 | H8 | H15 |
| --- | ---: | ---: | ---: | ---: |
| Reward MAE | 0.03565 | 0.07060 | 0.06940 | 0.10137 |
| Embedding cosine | 0.92614 | 0.82933 | 0.70388 | 0.49234 |
| Raw posterior KL | 7.414 | 12.520 | 15.764 | 26.871 |
| Hard-mask excluded policy mass | 0.00306 | 0.07230 | 0.10980 | 0.31026 |
| Policy argmax agreement, oracle mask | 0.99167 | 0.90625 | 0.91067 | 0.87037 |

The collected historical 128-episode evaluations also include direct multistep
JEPA on 2s3z seed234 at 0% win rate, and the c23 return-calibration treatment at
12.5%, compared with c19's 43.75% seed0 result. These are historical comparisons,
not matched causal ablations of the corrected PPO implementation. They rule out
claiming that those additions are untried; they do not rule out a targeted fix
after identifying a specific simulator failure.

## Minimal follow-up if common-cohort H1-to-H5 error grows

First use an offline, frozen-checkpoint paired audit with current raw-episode
history and shared categorical randomness. Separate teacher-forced error from
the extra self-fed error; examine cumulative-return bias and action-mask errors,
not only representation cosine.

If H1/H2 outcomes are accurate and H5 is materially worse, the smallest first
intervention is a matched H2 imagined-PPO arm with the already-added factual
critic grounding retained. It changes one configuration value, keeps a value
bootstrap for longer credit, and tests whether policy updates are exploiting
the extra unreliable transitions. This is an experiment, not a recommendation
to discard long-term credit or a guarantee of improved performance.

If short-horizon truncation helps and the paired audit identifies a reproducible
self-feed failure, the next focused model intervention is sampled last-step
supervision at H2/H4 on the actual joint-to-local-posterior path. Freeze actor and
local representation initially; correct the joint predictor/heads on current
raw-history roots, and require held-out improvement in outcome/return error.
The existing two-step implementation uses deterministic completion and is
incompatible with the maintained dual-view guard, so simply enabling its old
configuration would neither run nor reproduce the PPO state distribution.
