# Interface and coverage verification

The user authorized reallocating all six A100s to verify the measured simulator
failure and test its repair. This supersedes the unfinished value sweep and the
last matrix training job if still running at handoff. Preserve every completed
result, interrupted attempt, latest completed checkpoint, and original replay.
Do not treat interrupted jobs as zero wins or infrastructure failures.

We identified a failure in frozen simulators, particularly on stronger-policy
trajectories. We have not established that this is the original cause of seed
divergence or that the proposed repair improves real wins. This experiment tests
both the narrower model-learning claim and the actual online outcome.

## Fixed-data factorial: 12 jobs

Starting checkpoints: `3s_vs_3z` weak seeds 0 and 2, and `3s_vs_5z` weak seed 0.
Use the original frozen matrix checkpoint in every cell for a given case.
Each case has two objectives crossed with two experience conditions:

| Factor | Control | Treatment |
| --- | --- | --- |
| Objective | Existing BPTT2 self-fed auxiliary | BPTT2 + factual-history posterior alignment |
| Experience | 16 episodes from the weak policy | 8 weak-policy + 8 stronger-policy episodes |

The preserved 32-episode bank from each policy is split into 16 candidate training
episodes and 16 held-out episodes using seed 314159. The mixed cell uses the
first eight candidate episodes from each source. Both experience conditions have
16 unique training episodes. Alternate paired episode slots for exactly 250
updates per half/source; use the same episode-slot draws and model RNG keys in
both objective cells. Episode lengths can differ; retain exact transition counts
and masks. Coverage changes the trajectory distribution, not optimizer budget.

Every cell gets 500 updates, eight recurrent roots per update, H2/H4/H5 endpoint
losses and two-step BPTT. Only joint predictor and reward/continuation/mask/alive
heads are optimized. The encoder, local recurrent model, actor, critic and their
targets stay fixed. All cells start with the same fresh Adam optimizer, learning
rate 4e-5, beta1 .9, beta2 .999, epsilon 1e-8, global gradient clip 1000.
This is an isolated diagnostic, not a full resumed learner or benchmark result.

Audit each before/after model on identical 96 held-out roots pooled from the two
policies, H1/H2/H4/H5/H8, seed 2718. Reconstruct raw histories from episode reset.
Report false deaths separately from missed deaths, action-support errors,
posterior KL, reward error, and source-policy strata. Use episode-level paired
uncertainty; individual agents/roots are not independent training seeds.

No imported trajectory or modified diagnostic checkpoint enters the fresh
online stage. Dataset manifests distinguish the real extra experience used by
this diagnostic from the standard online environment budget.

## Fresh online comparison: 24 jobs

Maps, in priority order: `3s_vs_3z`, `3s_vs_4z`, `2s3z`, `MMM`.
Seeds 0, 1, 2 for both base and align: six simultaneous jobs per map.
Each starts from scratch and collects 50k joint environment transitions,
including 5k replay prefill. Final evaluation is 100 greedy episodes; curve
evaluation is 32 episodes every 5k transitions. Extra diagnostic episodes are
excluded from training. The planned online budget is 1.2M transitions total.

Both arms retain one collection environment, imagination H5, actor 3 x 1024,
world/joint LR 4e-5, actor/critic LR 3e-5, five actor and critic epochs, entropy
.01, batch-standardized advantages, replay-value scale .3, critic target full
copy, encoder EMA .01, zero optimizer warmup, train ratio 128, collection mix .05,
BPTT2, eight anchors, recurrent loss .1, and independent reporting RNG.
Factual-value sweep controls remain off. The only algorithm difference is the
alignment auxiliary; no liveness threshold, reward or map-specific rule changes.

New `trajectory_kl_scale=.1` multiplies the existing recurrent scale `.1`.
It computes KL from the stop-gradient factual local posterior to the posterior
induced by the imagined embedding and imagined recurrent history, including the
configured categorical unimix. The local posterior parameters remain frozen;
the live input/history Jacobian credits the joint predictor within the existing
two-step BPTT window. This differs from the older `consumer_kl_scale`, which
compares embeddings on the same predicted deterministic history. That older
flag remains zero and its existing BPTT2 restriction remains in force.

Preserve every 5k checkpoint for both arms/all seeds on `3s_vs_3z`, including
early states before divergence. Other maps retain the normal latest/final
checkpoint. All checkpoint selection for reported wins is fixed final, never
the largest curve value. Paired replay inputs must be frozen explicitly for
later forks; checkpoint retention alone does not preserve historical replay.

## Decision and monitoring

Use all three training seeds and report per-seed final wins, mean, minimum and
spread on every map. A simulator-metric improvement alone is insufficient.
Evidence for improved stability should include improved weak-seed performance,
without sacrificing mean performance or the retention map. Treat three seeds
as limited evidence; do not infer a universal fix or DMAWM competitiveness from
one selected run. Report the fixed-data coverage interaction separately from
fresh online performance.

Stop new claims at the original 45-hour cutoff; decide by the original 48-hour
deadline: 2026-09-08 10:42:23 UTC / 12:42:23 Zurich. Do not reset the clock, add
unbounded arms, or extend the existing compute authorization. Incomplete data
remain incomplete. Completed jobs are never restarted to seek a favorable seed.

Active queue: `/workspace/majepa_interface_verify_20260906/queue.json`.
Frozen source: `/workspace/ma_jepa_interface_verify_20260906`.
W&B group: https://wandb.ai/osaze-obahor/majepa-ppo-treatments/groups/ma-jepa-interface-verify-20260906
Launcher: `scripts/run_interface_verification.py`.
Offline training: `scripts/verify_interface_offline.py`.
Online IDs: `ifv-ARM-MAP-sSEED-train` and `...-final100`.
Offline IDs: `ifv-off-MAP-sSEED-ARM-COVERAGE`.

Monitor workers, finite update metrics, W&B synchronization, before/after audits,
checkpoints and queue transitions. Keep source snapshots immutable. On a
technical fault preserve the failed attempt and repair the launcher/runtime in
a versioned way; do not silently change the algorithm or substitute a result.
Do not restart the superseded value sweep or old checkpoint keeper. Notify on
material findings, failures, completion or a required decision; unchanged state
does not require a user update.
