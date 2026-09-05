# Draft: recurrent world-model training and consumer consistency screen

Status: planning only. No source revision or loss scale is selected and no runs
are authorized by this document. The six current correction runs must retain
their original 50k/final128 protocol. New work needs a tested, immutable source
snapshot and the fresh representation/simulator diagnostics before deployment.

## Question and six-run allocation

Does training the world model on its recurrently generated inputs improve real
control, and does an additional consumer KL objective improve on that change?
The current correction screen isolates cooperative death semantics and factual
critic grounding; this screen addresses the remaining training/imagination
distribution mismatch. Consumer KL remains a hypothesis, not an established fix.

| Slot | Treatment | Task | Training seed | Comparison |
| --- | --- | --- | ---: | --- |
| 0 | Recurrent loss only | 2s3z | 0 | Corrected baseline with same seed |
| 1 | Recurrent loss + consumer KL | 2s3z | 0 | Slot 0 at fixed seed/budget |
| 2 | Recurrent loss only | 2s3z | 1 | Corrected baseline with same seed |
| 3 | Recurrent loss + consumer KL | 2s3z | 1 | Slot 2 at fixed seed/budget |
| 4 | Recurrent loss only | 3m | 0 | Current corrected 3m regression run |
| 5 | Corrected baseline, selected anchor setting | 2s3z | 123 | Additional baseline training-seed replication |

“Recurrent loss only” means adding the recurrent training objective to the
existing corrected model, while keeping the current factual training objectives.
It does not mean discarding factual teacher-forced training. Slots 0–4 use the
same selected factual critic anchor setting and all share one recurrent-loss
configuration. Slots 1 and 3 differ only in consumer KL. Slot 5 has both new
objectives disabled and must reproduce the selected corrected baseline exactly.

## Select and freeze before launching

The anchor choice, exact objective definitions, gradient paths, scales, horizons,
source commit, package SHA256, and configuration keys are **TBD**. Record them
in a dated final protocol before starting any of these runs. Do not select them
from an early curve peak or tune them independently for each task or seed.

Use the current screen's separate final128 results and mechanistic diagnostics
to choose the anchor setting once. A proposed deterministic development rule is
the higher mean final win rate over 2s3z seeds 0 and 1, then mean final return for
an exact win-rate tie; retain the maintained setting if both remain tied. Any
departure because of numerical instability or contradictory diagnostics must
be documented before testing seed 123. Report both losing and winning anchor
results. The same selected setting applies to all six new runs.

Before deployment, resolve and test these contracts:

- Recurrent inputs must match the executable imagination path: action encoding,
  history length, terminal/presence masks, peer-action distribution including
  unimix, and recurrent use of predicted features. Record which predictions are
  sampled versus averaged and which stochastic draws are coupled.
- Factual targets come from the same replay trajectories. Episode boundaries,
  truncation bootstrap behavior, and per-horizon valid masks must remain correct.
  Document target-encoder stop gradients and whether gradients cross recurrent
  steps or sampled peer actions.
- Specify the consumer KL distributions, KL direction, legal-action support,
  target branch, detach rules, and which parameters receive gradients. An actor
  update hidden inside world-model fitting would change the PPO comparison and
  must not be silently introduced.
- A zero scale must recover the prior corrected training path. Positive scales
  must produce finite losses and the intended nonzero gradients through an
  actual learner update. Validate the full resolved configurations and a bounded
  SMAC/GPU integration before assigning a new immutable source fingerprint.

The implementation under review proposes the isolated configuration group
`agent.marl.ctde.self_fed`, disabled by default, with candidate horizons
`[2, 4, 5]` and eight anchors. The final protocol must fill in its common
recurrent `scale` and separate `consumer_kl_scale`; neither is selected here.
The common recurrent scale multiplies the existing embedding, interface,
reward, continuation, action-mask, and alive loss weights. Record all resolved
weights rather than quoting the common scale alone.

Its proposed gradient contract uses recorded joint actions, samples the frozen
local posterior, and detaches state between recurrent steps. Gradients reach
the current joint model and prediction heads, not the actor, encoder, or local
dynamics through this auxiliary objective. Thus this tests fitting at self-fed
inputs rather than full backpropagation through the entire imagined sequence.
The proposed consumer KL compares frozen local-posterior distributions from
the factual online embedding and the predicted embedding using the same stopped
self-fed deterministic state. It is an interface-consistency objective, not a
direct actor or critic KL. Verify these details against the final implementation
and tests; changes to this contract must be reflected in the run manifest.

Fresh diagnostics should compare factual teacher-forced and recurrent self-fed
predictions from identical raw-history roots, with coupled samples and matched
horizon support. Record reward/continuation calibration, availability-mask
errors, executable peer-action accuracy, and latent error. Pair value predictions
with actual discounted stochastic-policy outcomes at the same states; aggregate
imagined means versus full-episode returns are not paired calibration evidence.

## Fixed training and evaluation protocol

Preserve the current screen's original base hyperparameters unless the final
protocol explicitly identifies a necessary change: one collection environment,
50,000 joint environment transitions including prefill, PPO start at 5,000,
world-model start at 0, 1,000-update world optimizer warmups, train ratio 128,
imagination horizon 5, actor 3×1024, actor/critic learning rates 3e-5, collection
unimix 0.05, and fixed entropy coefficient 0.01. Do not bundle critic EMA changes,
replay-age changes, new capacity, optimizer changes, or entropy annealing into
these arms. Record extra compute caused by recurrent fitting and consumer KL.

Use the same SMAC maps and SC2 4.10/Base75689 runtime as the active screen.
Evaluate 32 greedy episodes every 5,000 transitions with seed offset 50,000;
evaluate the exact 50k checkpoint on a separate fixed 128 episodes with seed
offset 100,000 and four evaluation environments. Keep both seed sets unchanged.
Report all ten curve points and final128 separately; do not select a best
checkpoint. A failed run remains part of the result table.

## Evidence, decisions, and limitations

Primary comparisons are fixed-final128 win rate for recurrent versus corrected
baseline on 2s3z seeds 0 and 1, then recurrent+consumer KL versus recurrent-only
at those same seeds. Report each seed, mean, spread, return, timeout rate,
learning-curve area, and wall/GPU time. Diagnostics must establish whether the
new objective reduces the measured mismatch without numerical instability.
Improved latent error alone is insufficient if real policy performance declines.

The third baseline seed estimates some remaining baseline variability; it is
not a third recurrent or consumer replication. The consumer setting remains a
development hypothesis tested on two seeds. Original PPO controls cover only
two training seeds, with the preserved seed-0 experiment coming from the earlier
runtime whose SC2 build was not recoverable. Current-host seed-1 control gives
the cleaner runtime match. Anchor selection on seeds 0 and 1 is itself a
development choice and does not provide an unbiased independent benchmark.

Recurrent fitting on recorded joint actions tests exposure to predicted state
inputs. It does not demonstrate counterfactual accuracy on actions absent from
replay or remove the behavior-policy action-distribution gap. Local-posterior KL
agreement also does not by itself establish reward/value sufficiency.

The single 3m run screens for a regression; it cannot establish robustness.
Moreover, if the selected anchor differs from the active corrected 3m setting,
the available comparison changes both anchor and recurrent training and must
be labeled as such. A matched 3m control would then be required before making a
causal claim about a 3m change. This six-run plan tests mechanisms on two tasks
and does not establish superiority over DMAWM or MATWM; use the broader
protocol in `benchmark_targets.md` for that claim.

Use unique source/output directories and a distinct W&B group with deterministic
train/final IDs. Start the compact local metric/config mirror before training;
archive each successful final checkpoint, configurations, raw replay chunks,
source provenance, and later diagnostics independently of unsuccessful peers.
Verify SHA256 of transferred artifacts. Keep all source data until local
backups have been verified. Do not commit live mirrors or large checkpoints.
