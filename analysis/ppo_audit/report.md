# PPO experiment audit — 4 September 2026

Read-only evidence was extracted from JNIN and Jinn. The last treatment snapshot
is approximately 20:37 UTC. No existing experiment, source tree, checkpoint, or
remote process was changed. Four compact JSON files preserve run provenance,
selected diagnostics, historical outcomes, and evaluation curves;
`outcomes.csv` is the sortable summary. Training-window entries are means of
logged observations in the preceding 5,000 environment steps, not means over
individual decisions. Latest metrics retain their actual logging step.

## What the existing experiments establish

The first PPO wave solves 3m consistently but remains weak on 2s3z. All rows
below used seed 0, 50,000 total environment transitions, H=5, ten evaluations
of 32 episodes, and a separate final 128-episode evaluation. The final evaluation
uses four workers with offset 100,000, and curves use offset 50,000.

| Treatment | 3m final wins | 2s3z final wins | 2s3z final return | 2s3z training hours |
| --- | ---: | ---: | ---: | ---: |
| Original PPO base | 120/128 (93.75%) | 14/128 (10.94%) | 12.63 | 1.63 |
| Actor 2×256, no peer residual | 121/128 (94.53%) | 26/128 (20.31%) | 13.45 | 1.56 |
| 16 environments, same total steps | 117/128 (91.41%) | 23/128 (17.97%) | 13.12 | 1.40 |
| Start world model at 5k, no optimizer warmup | 124/128 (96.88%) | 35/128 (27.34%) | 13.35 | 1.37 |
| Annealed normalized entropy, no collection mixing | 123/128 (96.09%) | 2/128 (1.56%) | 10.45 | 1.47 |
| One-layer, eight-head critic | 113/128 (88.28%) | 1/128 (0.78%) | 10.83 | 1.59 |

The base, small-actor, critic, and entropy treatments all improve and regress
between 2s3z checkpoints. The delayed-world-model treatment is the clearest
first-wave improvement, with 18.75–37.5% curve wins throughout 20k–50k. These
are single training seeds; 128 evaluation episodes do not measure training-seed
variance. No row proves that an architecture or entropy coefficient is optimal.

Seventeen of the eighteen treatments have an early deterioration after PPO
starts at 5k. For example, base 2s3z goes from 4.33 return at 5k to 1.76 at 10k,
then recovers to 10.52 at 15k; base 3m falls from 1.75 to zero before recovering.
The same shape appears with 16 environments and delayed world-model
training. This is a repeatable startup problem, rather than simply slow final
convergence. Starting model learning and actor learning together with zero
model updates is not an evidence-based general solution even though one such
treatment improved the final score.

## The diagnostics do not validate the learning target

In base 2s3z at 45k–50k, the PPO critic reports explained variance 0.984 and its
batch target return and target value are 9.116 and 9.115. Final actor KL is only
about 0.0015. Those metrics show successful fitting of short imagined,
self-bootstrapped targets, not calibrated prediction of real returns. Poor real
wins coexist with very high reported critic fit.

One combined treatment is particularly informative: at approximately 28k,
delayed-world-model + 16 environments + simple critic reported imagined return
24.29, target value 24.18, and explained variance 0.990, while its latest full
episode evaluation return was 10.91. This is a warning of value drift, although
the replay-root and initial-episode state distributions differ and the numbers
must not be interpreted as a paired estimate of bias.

At 45k–50k the base predicted imagination reward averages 0.213, compared with
0.135 in the behavior replay view and 0.164 in the world replay view. Different
actions and state sampling contribute to that gap, so a paired factual rollout
diagnostic is needed before attributing it entirely to model optimism.

The code audit by the implementation agents identifies mechanisms that match
these symptoms: cooperative returns were cut by individual death, dead-agent
reward/continuation rows were excluded from world-head supervision, and PPO
removed the pre-existing factual replay-value objective. The correction screen
must test those mechanisms directly. A low imagined value loss cannot adjudicate
them.

## World and teammate fit is uneven

The base 2s3z late diagnostics include:

| Diagnostic | Training view | Report view |
| --- | ---: | ---: |
| Active teammate action top-1 | 67.9% | 39.3% |
| Active teammate plan q4 top-1 | 59.4% | 31.5% |
| Embedding cosine | — | 0.816 |
| Legal attack recall | — | 90.0% |

Training uses a recent world view; reporting does not. Behavior roots are drawn
from a uniform buffer and average approximately 23k steps old near the end of a
50k run, whereas world roots average approximately 5k steps old. Thus the actor
is optimized from states whose dynamics and teammate policy are less well fit
than the headline training numbers suggest. This is a plausible secondary
distribution mismatch; it should be isolated after the return corrections.

The delayed-world-model 2s3z treatment has fewer timeouts (3.1% versus base
21.1%), fewer movement actions (12.5% versus 19.6%), and more attack actions
(53.6% versus 38.8%). This is consistent with engagement/commitment being an
important part of the behavior gap. Overall action fractions include dead-agent
noops and should not be treated as active-agent-only rates. Previous repository
handoffs also report target allocation and temporal commitment failures; those
claims should be verified with paired action/role diagnostics on the corrected
model before adding another actor mechanism.

## Historical context and distinctions

Earlier REINFORCE 2s3z results at 50k and fixed final128:

| Experiment | Seed 0 | Seed 1 | Seed 123 | Mean |
| --- | ---: | ---: | ---: | ---: |
| c32 legal collection mixing 5% | 52.34% | 34.38% | 35.16% | 40.63% |
| c32 legal collection mixing 1% | 31.25% | 48.44% | 16.41% | 32.03% |
| c31 finite horizon | 13.28% | 46.09% | 37.50% | 32.29% |

The newer entropy screen at commit `3583b59` achieved 14.84%/6.25% for seeds
0/1 with collection mixing disabled, and 46.88%/34.38% when policy action mixing
was disabled as well. Thus entropy/mixing changes have already shown large,
inconsistent effects. They do not supply a general fix for the current PPO
failure. Historical and current PPO runs differ in optimizer schedules and
other settings, so these comparisons do not isolate PPO versus REINFORCE.

Commit `e3fd3e5` (“Separate death masking from team pooling”) is an ancestor of
that entropy screen. Its change removed live-agent pooling of rewards,
continuations, and value logits while retaining individual death masking in
imagination and replay-value losses. It did not implement the current correction
of keeping cooperative GAE returns alive after focal-agent death or supervising
dead roster rows for team reward and continuation. The old treatment names
must not be mistaken for an earlier experiment on the same correction.

c23 “return calibration” achieved only 12.5% for seed 0 versus 43.75% in the
earlier exact-c4/all-legal reference. Its mechanism added a complete-tail Monte
Carlo target with a ramp and an extra coefficient of 0.25 **on top of an already
present factual replay-value objective**. Restoring the factual TD/λ-return
objective deleted during the PPO switch is a different intervention.

Historical source roots and exact evaluation paths are recorded in
`jnin_historical.json` and `jinn_historical.json`; current manifests and selected
metrics are in `jnin_treatments.json` and `jinn_treatments.json`.

## Correction screen and operational plan

`scripts/run_ppo_correction_screen.py` defines this fixed six-run screen:

| Slot / intended host GPU | Map | Seed | Changes from original PPO |
| --- | --- | ---: | --- |
| 0 / JNIN 0 | 2s3z | 0 | Cooperative death semantics; anchor scale 0 |
| 1 / JNIN 1 | 2s3z | 0 | Cooperative death semantics; factual critic anchor 0.3 |
| 2 / JNIN 2 | 2s3z | 1 | Cooperative death semantics; anchor scale 0 |
| 3 / JNIN 3 | 2s3z | 1 | Cooperative death semantics; factual critic anchor 0.3 |
| 4 / Jinn 0 | 3m | 0 | Cooperative death semantics; factual critic anchor 0.3 |
| 5 / Jinn 1 | 2s3z | 1 | Original source, original PPO reference |

All use original architecture, LR, entropy, 1 environment, 5k PPO start,
1000-update world optimizer warmups, 50k total steps, curve32 every5k, and
fixed128 at the final checkpoint. JSONL logging avoids dependencies on external
logging services. There is no early checkpoint selection. Training should take
roughly 1.5–1.8 hours once a GPU is available, plus final evaluation; the critic
anchor may add overhead.

The launcher validates configuration and exact package SHA256 before waiting,
and verifies the hash again before training. Supply predecessor supervisor PIDs
with repeated `--wait-pid`; it requires both those PIDs to have exited and the
GPU to have no compute processes for 60 continuous seconds. This protects the
gap between a predecessor's training and final evaluation. It only cleans up
processes it created, preserves all checkpoints, and refuses existing result
directories. Source and output roots must be new timestamp/hash directories.
Local launcher tests passed (5 tests), actual configuration resolution passed
for corrected slots 0/1/4, and Ruff passed. Full algorithm validation and parent
confirmation are prerequisites to source staging and launch.

At 20:37, current JNIN treatments were at 35k–40k and the two Jinn treatments at
25k–26k. JNIN looked approximately 15–30 minutes from training completion;
Jinn was slower around evaluation and subsequently resumed, so its completion
time was less certain. These are observations of existing jobs, not promises
of availability.

## Deployment update — 20:45 UTC

The approved tested commit `2663ae57ab80482ad201f9c3bdaa9fe2c732dbe9` was archived and staged to `/workspace/ma_jepa_ppo_2663ae5` on both hosts. Both package fingerprints match `9db9598e0da7cced1eab43036845b9ff593bb9f61713720f4654460a5c49b4e7`. All six profiles validated on their host runtimes; all six supervisors are queued and waiting for predecessor jobs to finish. Existing jobs were not interrupted. `deployment.json` contains exact commands, PIDs, hashes and observed status. New output root: `/workspace/majepa_ppo_correction_2663ae5_20260904`. Original control source is pinned to `92014f1dd35f946816c350906af85370a434de40`; its package code is identical to the first-wave `cf97200` snapshot.

## Recovery after old pods stopped — 5 September

The last local correction-screen observation is 4 September 20:47 UTC: all six
supervisors were waiting, with no training metrics yet. Their later completion
status is **unknown**, rather than known to be failed or still at step zero.
The new pod's workspace is empty of old experiment storage. The original
correction runs used JSONL-only logging, so their outcomes cannot be recovered
from W&B. `recovery_status.json` identifies all six missing result directories
and the precise manifests, metrics, checkpoint pointers, and final-evaluation
files that would need to be recovered from the old persistent storage.

The second-wave **baseline** treatments did log to W&B, and their outcomes and
all ten curve points were recovered read-only using existing account access.
Every one reached 50k and completed its fixed final128 evaluation:

| Second-wave 2s3z treatment, seed 0 | Final wins | Final return |
| --- | ---: | ---: |
| Delayed world model + 16 envs | 15/128 (11.72%) | 12.39 |
| Above + small actor | 0/128 (0%) | 9.79 |
| Above + annealed entropy | 7/128 (5.47%) | 10.93 |
| Above + simple critic | 15/128 (11.72%) | 13.06 |
| Above + entropy + small actor | 9/128 (7.03%) | 11.73 |
| Above + entropy + simple critic | 0/128 (0%) | 8.46 |

These combinations did not improve on the first-wave delayed-world-model
result (35/128, 27.34%). Exact W&B URLs, completion states, final metrics, and
curves are in `second_wave_recovered.json`; `completed_baselines.csv` contains
all 18 verified completed baseline outcomes.

The correction screen is being rerun on the new six-A100 pod with the same
verified algorithm packages. The launcher now supports explicit W&B logging
with unique train/final IDs and a distinct date-stamped group. An independent
local watcher mirrors complete JSONL records, manifests, outcomes, and final
evaluation summaries every two minutes. It copies checkpoint inventory only,
excluding weight payloads and credential files. The expanded launcher/mirror
tests passed, and all six profiles and W&B authentication validated
on the new runtime. `deployment_20260905.json` records the new deployment;
`correction_20260905_mirror/` holds the independent local evidence.

The new screen launched at 07:19 UTC on 5 September with slots 0–5 mapped to
GPUs 0–5 of the same pod. At 07:35:54 UTC every run had real learner metrics:
environment steps were 3820, 2470, 3830, 2450, 4110, and 3800 in slot order.
PPO update counters were all zero before the configured 5k start. Initial
precompilation took approximately 10–13 minutes and completed successfully.
Training speed and outcomes should be assessed from subsequent measurements.
The original baseline SC2 build could not be recovered from its W&B metadata;
the new pod uses verified SC2 4.10/Base75689 for all six arms.

`summarize_mirror.py` reads local evidence without SSH and retains the step of
each individual metric. Both saved configuration files are mirrored. Separately,
`watch_final_backups.py` waits for each successful final128 outcome and invokes
`backup_final_artifacts.py` to preserve that run's checkpoint, configuration,
manifest, evaluation summary, and raw replay NPZ chunks. It transfers one run at
a time at 8 MB/s, verifies SHA256 for every file, and preserves remote sources.
It does not require peers to succeed or diagnostic exports to exist. The archive
root is `/Users/osaze/MARL/remote_archives/majepa_ppo_correction_20260905`; it had
175.6 GiB free at watcher start. Later diagnostic exports use `--exports-only`.
The launcher, incremental mirror, local summary, and per-run backup behavior
have 18 passing tests; Ruff passes. Deployment records include watcher commands
and process IDs. No final outcomes are available at this startup observation.
