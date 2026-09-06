# Bounded replay-coverage follow-up

The user asked what to queue behind the active interface verification. Complete
all 36 logical jobs in that queue first (12 fixed-data diagnostics and 24 fresh
online runs). Six follow-up workers then claim the 24 jobs below. No current
learner, seed, objective, or result is replaced. This is a bounded addition of
1.2M joint environment transitions within the original 48-hour decision window.

## Evidence and hypothesis

The completed fixed-data factorial favored mixed weak/strong-policy experience
over weak-policy-only experience more consistently than it favored posterior
alignment: posterior KL and legal-action false negatives improved in all six
paired coverage comparisons, liveness Brier in five, and five-step return RMSE
in four. These are directional diagnostic comparisons, not six independent
online seeds. Imported strong trajectories never enter the online experiments.

Current world-model training samples sequence starts with exponential recency
weighting, while PPO imagination roots are uniform over replay. Mixing recent
and uniform world-model samples may reduce this distribution mismatch and
retain competence on older or rare behavior. It cannot create successful
experience that the online policy never collected. This is a testable coverage
hypothesis, not an established cure for seed instability.

The primary-map final100 scores also require attribution discipline. Current
base scored 83%, 94%, 47%; align scored 79%, 71%, 74%. Both average 74.67%.
Alignment has a narrower observed range on three seeds, with no mean gain.
The old matrix scored 15%, 75%, 9%. The current base already includes the shared
report-RNG isolation change. That fix prevents diagnostic reporting from
advancing the behavior replay sampler and training batch seed counter, but
its causal contribution to the win-rate difference is unmeasured.

## Predeclared jobs and controls

| Jobs | Maps, in queue order | Training seeds | Change |
| --- | --- | --- | --- |
| 18 | 3s_vs_4z, 2s3z, 3s_vs_3z | 0, 1, 2 for base and align | World replay: 50% current recency selector, 50% independent uniform selector |
| 6 | 3s_vs_3z | 0, 1, 2 for base and align | Report RNG isolation off; original world replay |

Each cell's paired control is the same map, seed and objective in the current
interface verification. The coverage jobs complete a recency/mixed-replay by
base/alignment comparison using the already-running controls. The report-off
jobs compare with current report-on controls for both objectives. They test
the reporting coupling; favorable results would not imply every randomized
training trajectory improves when reporting is isolated.

Keep BPTT2, one collection environment, H5 imagination, eight recurrent roots,
recurrent loss scale .1, world/joint LR 4e-5, actor/critic LR 3e-5, actor 3x1024,
five actor/critic epochs, full-copy critic target, encoder EMA .01, 5k prefill,
train ratio 128, entropy .01 and all other algorithm settings unchanged.
Alignment remains trajectory_kl_scale .1 for align, zero for base. Factual
value controls remain off. Uniform world sampling has its own selector/RNG;
it does not consume behavior or report draws. A zero mixture preserves the
original sampler calls. Logs include the realized uniform-world fraction and
the existing sampled-age and update-ratio diagnostics.

All jobs start fresh and train to 50k, including prefill. Curve evaluations
remain 32 greedy episodes every 5k; fixed-final evaluation remains 100 greedy
episodes. The full resolved training/evaluation configuration is compared
against the saved paired control, allowing only the declared replay/report
flags and logging filter. Checkpoint saving cadence also matches the controls.
Same seeds are paired statistically; timing-driven reports and asynchronous
execution do not guarantee bitwise matched training trajectories.

## Retention, decisions and limits

Retain the latest full checkpoint throughout each new training/final100 job.
Only strictly older completed checkpoint files are removed; an in-progress
save or newly completed save whose latest pointer has not advanced is safe.
After final100 completes, preserve every model tensor exactly in agent.pkl.gz,
with tensor equality and SHA256 verification. Optimizer state is omitted.
This applies to every new run regardless of score. The new jobs do not keep
5k weight histories. Current verification histories and seed-0 resumable
finals follow their existing retention policy.

Report every completed per-seed final100 score and each map's mean, minimum,
and range. Prefer a coverage change only if the weak-seed improvement is
repeatable across maps without a material mean regression. Three seeds are
limited evidence; evaluate uncertainty and mechanisms before promoting a
default. Compare held-out simulator error, sample-age coverage and legal/death
errors to explain any win-rate change. A successful candidate next warrants
retention testing on MMM and other strong maps, not immediate benchmark claims.

The claim cutoff stays 2026-09-08 07:42:23 UTC; decide by 2026-09-08
10:42:23 UTC (12:42:23 Zurich). No added arms or deadline extension. Partial
curves are not final results; overnight completion provides evidence and does
not guarantee an improvement. Preserve failed attempts and repair only technical
faults in versioned source. Do not rerun completed seeds to select winners.

Queue: /workspace/majepa_coverage_followup_20260906/queue.json.
Dependency: /workspace/majepa_interface_verify_20260906/queue.json.
Launcher: scripts/run_coverage_followup.py.
W&B group: https://wandb.ai/osaze-obahor/majepa-ppo-treatments/groups/ma-jepa-coverage-followup-20260906.

## Deployment and storage record

Frozen source: /workspace/ma_jepa_coverage_followup_3d4aeda_20260906,
commit 3d4aedadab7db7c118345c88c6d030788cf5fc19. Package SHA256:
101751eba8d04bb6f4db7348760703fc15e97d25d6ac00c80bd1ab289c227eb8.
Only main.py, configs.yaml and replay.py differ from the current learner
package. One actual replay check verified broader coverage, eviction and
unchanged behavior draws; all 48 train/final100 configuration comparisons
passed. Six worker processes were verified waiting for the current matrix.

To make room without discarding model evidence, four byte-identical old
diagnostic checkpoint copies were replaced by hard links to the preserved
canonical checkpoint, reclaiming 7,020,856,384 bytes with all original paths
and full checkpoint contents retained. Under the user's prior unsuccessful
checkpoint cleanup authorization, ten completed weak/abandoned configurations
also retain all model tensors exactly while their optimizer state is released.
The strong references and original diagnostic input checkpoints stay full.
The already-installed SC2 download archive is redundant and can be removed;
the installed game, dependencies, replay and metrics stay intact. Per-path
scores, equality checks, sizes and hashes are recorded in the deployment and
the pod's storage_followup_cleanup.json. No local checkpoint archive is made.
