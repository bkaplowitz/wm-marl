# World/behavior replay burn-in review — 5 September 2026

Scope: read-only review of the maintained implementation at `2663ae5` and its
history. No source changes, training updates, or GPU jobs were made for this
review. The mismatch is real at the interface level, but its performance cost
has not been established. Collection age is **not** latent age.

## What each path actually does

- `src/majepa/marl/core.py:2355` overrides world/report `_apply_replay_context`.
  It replays saved `dyn/pair`, `dyn/stoch`, `dyn/reset`, `dyn/position`, and
  activity. `ParallelTransformerDynamics.replay_sequence` at
  `src/majepa/world_model/transformer.py:375` rebuilds temporal hidden state with
  current Transformer weights and the saved pairs, then takes saved stochastic
  samples as its state. This is not reuse of an old KV cache. The joint burn-in
  then consumes these reconstructed local features with current joint weights.
- `src/majepa/training/learner.py:227` re-encodes behavior prefix raw observations
  with the current encoder and history-conditioned local posterior. The resulting
  carry and joint burn-in features are stopped. PPO does this **after** the world
  optimizer step. The world forward and its returned replay entries were computed
  before that step. Even a just-refreshed entry is not an exact same-weight sample.
- `src/majepa/training/reporting.py:20` uses the world stored-prefix path. Report
  sampling is uniform; world training sampling is recent. Thus comparisons of
  aggregate training, report, and PPO curves change both history construction and
  state distribution. They are not a paired test of either effect.

## Refresh cadence and why archive age can mislead

`LearnerMixin.train` returns only the optimized **world suffix** entries and
step IDs (`learner.py:127-136`). `TeamAxis.unfold_replay_updates` keeps one common
step ID per team and all agent features (`marl/axes.py:172`). The maintained
runtime returns its previous call's asynchronous outputs
(`external/dreamerv3/embodied/jax/agent.py:285-291`), and `train.py:193-194` applies
`replay.update` whenever those outputs arrive. Consequently:

1. Every successful training call normally leads to a suffix refresh one call
   later. The codes were generated before that call's world weight update.
2. The prefix of the sampled window is not refreshed by that call. Behavior
   inference and reports never refresh replay entries.
3. Overlapping world suffix windows can refresh rows later used as prefixes.
   The world and behavior selectors share one physical replay store.
4. `Replay.update` counts attempted rows, not unique or successful refreshes;
   updates to evicted chunks are silently skipped. There is no per-row model
   version or last-refresh timestamp. The one-item prefetch can also precede a
   refresh that happens before the prefetched batch is consumed.

For the current ratio128, length64, batch16 configuration, the nominal cadence is
one learner call per8 inserted team transitions, 1024 suffix rows per call, or
128 attempted refreshed rows per inserted transition. This does **not** imply
128 refreshes of every row. With exponential selector decay0.9998, a rough
stationary single-environment interior approximation gives a refresh hazard
`128 * (1-decay) * decay**age` per inserted transition. At age5000 this suggests
roughly106 environment steps between refreshes, not5000. The approximation
ignores overlap, sequence edges, changing replay population, prefetch, startup,
and eviction. Only instrumented actual refresh versions could measure this.

There is a second, stronger provenance limitation: pinned
`embodied/core/replay.py:296-307` saves each chunk UUID only once.
`embodied/core/chunk.py:54-59` updates its in-memory arrays but does not invalidate
the replay's saved set. Archived NPZ codes therefore reflect **first
serialization**, not necessarily the live replay at the final checkpoint.
Frozen checkpoint + archived prefix comparisons measure archived-state drift;
they cannot be presented as the online learner's current latent staleness.

## Other differences that must not be mislabeled weight drift

The inherited `Replay._annotate_batch` sets `is_first[:,0]=True` for every sampled
sequence (`embodied/core/replay.py:280-288`). Saved `dyn/reset` is unchanged. The
behavior prefix therefore begins at an artificial reset, whereas world replay
uses the saved actual reset and position. At true resets both temporal paths
use a learned start token, so the unmasked previous stochastic code inside the
stored pair does **not** leak through a reset.

The 192-step burn-in covers the local Transformer cache's nominal128-step and
joint cache's192-step receptive fields. Existing
`tests/test_majepa_transformer_world_model.py:241` proves reconstruction from the
**same saved pairs** matches the online cache. It does not prove equivalence of
freshly sampled histories: the posterior is history-conditioned, and categorical
states recursively carry information beyond a fixed Transformer window. A new
artificial reset or different prefix samples can therefore affect the suffix
even with frozen weights. Within-window new categorical draws are a source of
sampling variance, not by themselves approximation failure.

`_temporal_pair` is simply flattened previous categorical state concatenated with
the action features (`transformer.py:665`). Overlapping suffix updates can replace
`stoch[t-1]` without replacing a neighboring `pair[t]`, or the reverse. Replaying
such rows produces a mixture of different posterior trajectories even when their
ages are short. This is an independently measurable consistency issue.

## Small archived replay check

Four evenly spaced1024-step files from
`/Users/osaze/MARL/remote_archives/jinn_multistep_jepa_20260828/runs/generalist-d12w256-h15-dual-maskbalanced-multistepjepa-8m-seed234-50k-r1/run/replay`
were inspected with NumPy only. For non-reset adjacent agent states, compare
the stochastic part of `pair[t]` with saved `stoch[t-1]` exactly:

| Chunk creation timestamp | Eligible pairs | Mismatches | Fraction |
| --- | ---: | ---: | ---: |
| 20260828T002543F862684 | 7936 | 1112 | 14.01% |
| 20260828T012012F800335 | 8008 | 1656 | 20.68% |
| 20260828T021052F120798 | 7912 | 496 | 6.27% |
| 20260828T030510F465171 | 7936 | 832 | 10.48% |

These prove partly refreshed sequences existed when those chunks were first
saved. They do not measure the final checkpoint's live replay, nor demonstrate
that removing the seams improves policy performance.

## History: what has already been tried

- `0120476360d6cf8db718c909d5e6d7297f90d2fd` (26August) introduced the current
  separate behavior raw-prefix burn-in in the REINFORCE dual-view learner. The
  world stored-prefix method already existed in `20407ab`. This asymmetry was
  **not introduced by the PPO migration**. `d1cd115` retained it in the MA-JEPA
  cleanup, and `c86451e` retained the behavior helper during migration.
- `0120476` tests verify world/behavior optimizer isolation and that behavior
  changes do not affect world replay refresh outputs. They do not test matching
  histories, refresh ages, or prefix consistency.
- Recent/world versus uniform/behavior sampling was introduced earlier in
  `c3ebc16`; ratio/warm-start schedules (`2ead378`) and replay retention
  (`93401a1`, `4254efb`) were also explored. Those are not clean tests of fresh
  **world** prefix inference and should not be repeated under a new name.
- The historical rejected uniform-replay record `34dcf59` concerns a July DMC
  JEPA implementation, not a matched SMAC prefix ablation. Its endpoint result
  cannot establish whether this maintained CTDE prefix treatment helps.
- A history search across both `src/dreamarl` and `src/majepa` found no isolated
  fresh-world-prefix treatment. Absence from the searched commit/code records is
  not evidence that no unpublished run ever attempted it.

## Paired diagnostic before a training intervention

Freeze one checkpoint and snapshot actual **live sampled batches** with their
raw observations, saved codes, real `dyn/reset`, sampler-added `is_first`, step
IDs, source checkpoint/counter, and sample mode. Capture at least64 synchronized
roots across multiple complete episodes; stratify recent-world and uniform-report
cohorts. Do not change the running sampler or record collection age as latent
age. Archived data can be used only with its serialization qualification.

Run each identical batch with no gradients or parameter updates:

1. **Stored, operational:** invoke the current world `_apply_replay_context`.
2. **Fresh, operational:** invoke the current behavior raw-prefix helper on that
   *same world batch*, at the same frozen weights. This directly measures the
   training/consumer history gap without a replay-distribution confound.
3. **Fresh-repeat floor:** repeat fresh inference with independent prefix draws,
   coupling suffix categorical uniforms/Gumbels by absolute timestep across
   comparisons. Report the fresh-versus-fresh sampling floor. Giving two whole
   Ninjax calls the same seed is insufficient if their call/scan structure differs.
4. **Boundary reference:** on complete raw episodes, infer from the true reset
   with timestep-coupled draws, then cut the same roots. Compare fresh prefix
   lengths192/384 where history is available. This separates artificial-reset
   error from saved-code mismatch. Do not relabel the forced `is_first[0]` as a
   real environment reset.
5. Optionally replay pairs regenerated from freshly inferred prefix entries;
   with the *same* categorical draws and adequate cache coverage it should match
   direct fresh-prefix carry. This is the implementation-equivalence gate.

Report root and suffix offsets0/1/2/4/8/16/32: deterministic-state normalized
RMSE/cosine, actual-unimix posterior KL, policy categorical KL and masked action
agreement, live/slow critic difference, and joint teacher-forced reward,
continuation, mask, alive, embedding/interface error. Then run a paired actual
self-fed H5 factual-action rollout and cumulative-return error on each root
construction. Keep targets, suffix actions, root cohort, and categorical noise
fixed. Include counts, seam density, distance to last true reset, and episode
cluster uncertainty. A root representation gap that disappears in suffix losses
or is comparable to the fresh-repeat floor is weak grounds for a new treatment.

## Minimal experiment and required tests

No existing configuration isolates current-weight world burn-in. Increasing
recency or changing the buffer changes the state distribution and has already
been explored; increasing prefix length does not repair stale codes. Neither is
a clean first test of the identified interface.

If the paired gap is larger than sampling/boundary effects **and** harms
consumer-relevant predictions, the smallest code experiment is a default-off
world/report prefix option that calls the existing stopped raw-prefix helper on
the world batch. Preserve the one-step suffix loss, replay selection/ratio,
suffix-only update outputs, and actor path. Initially require `consec_train=1`
and `consec_report=1` (the current settings), or explicitly preserve later-chunk
carry semantics. Treat this separately from recurrent self-fed model training.

Required tests: same saved-pair cache parity; fresh entries→replayed prefix
parity with coupled draws; true reset isolation; artificial reset distinguished
from `dyn/reset`; overlapping refresh seam reproduction; delayed outer-runtime
refresh and suffix-only write extent; no prefix gradient; unchanged actor/critic
and default-off parameter/RNG behavior; full tiny JIT with terminal and unit-death
boundaries. A persistence test must show that ordinary checkpoint replay files
are not silently treated as a latest-live-code snapshot.
