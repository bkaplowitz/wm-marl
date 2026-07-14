# Native DreamerV3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL:
> superpowers:subagent-driven-development. Use one fresh implementer per task,
> followed by a fresh task reviewer for spec compliance and code quality.
> Critical and Important findings must be fixed and re-reviewed before advancing.
> Every behavior change follows strict red-green-refactor TDD.

**Goal:** Replace the approximate baseline with a native Flax port that is
numerically conformant with Danijar’s DreamerV3, defaults to the published paper
profile, and runs complete online DMC Vision and Proprio training.

**Architecture:** ARCHITECTURE.md is binding and must be read completely before
every task. The official implementation on dreamerv3+functional_online_jepa is
the oracle; this branch owns the native in-process implementation. Paper and
upstream-current profiles share code only when oracle behavior is identical.

**Tech stack:** Python 3.11, JAX/JAXlib 0.4.36, Flax, custom Optax-compatible
gradient transforms, NumPy fixtures, pytest, dm_control.

## Global constraints

- Default profile: paper. upstream-current is always explicit.
- Authorities: bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 and
  e3f02248693a79dc8b0ebd62c93683888ddaccfe.
- No anti-simplification rule in ARCHITECTURE.md may be weakened.
- Write each failing test and capture expected RED output before production code.
- Oracle fixtures require source hashes and an explicit official checkout.
- No submodule, Git-internal, other-worktree, process, or experiment mutation.
- Per-task conventional commits are authorized. Never push, merge, or launch
  paid compute without explicit authorization.
- Focused tests and reports must have pristine output. Full pytest runs at
  integration checkpoints and completion.

## Subagent controller protocol

For Task N:

1. Read .superpowers/sdd/progress.md; completed tasks are never repeated.
2. Record BASE_SHA before dispatch.
3. Run the skill task-brief script against this file and Task N.
4. Dispatch one fresh implementer with brief/report paths and only relevant
   prior interfaces.
5. Require status, commit SHA, RED/GREEN evidence, focused-test command/output,
   files changed, self-review, and concerns in the report.
6. Generate a review package from BASE_SHA through task HEAD.
7. Dispatch a fresh read-only reviewer with brief, report, diff package, and
   Global constraints.
8. Send all Critical/Important findings to one fresh fixer; require covering
   tests and re-review.
9. Append the clean commit range/review result to the progress ledger.
10. Continue without user interruption unless an architectural decision or new
    external authorization is required.

## File ownership

| File | Sole responsibility |
| --- | --- |
| config.py | Immutable profiles and configuration |
| distributions.py | Output distributions and scalar transforms |
| networks.py | Initializers, normalization, layers, encoder/decoder/heads |
| rssm.py | Latent state, prior/posterior transitions, KL terms |
| normalization.py | Return percentile and slow-value state |
| agent.py | Preprocess, policy, imagination, unified objective, report |
| optimizer.py | AGC, LaProp, train state, train_step |
| replay.py | Chunks, writers, online queue, uniform sampling, writeback |
| dmc.py | Exact DMC environment/vector contracts |
| driver.py | Interleaved collection, update limiter, evaluation |
| artifacts.py | Append-only metrics and run manifests/summaries |
| checkpoint.py | Atomic complete-state save/restore |
| oracle.py | Fixture provenance, parameter mapping, fixture generation |
| scripts/train_dreamer_v3_baseline.py | Public CLI and nothing else |

Old losses.py, models.py, imagination.py, training.py, and validation.py are
removed only after replacements are green and all imports migrate.

---

## Task 1: Freeze authority, profile, and oracle schemas

**Files**

- Replace config.py.
- Create oracle.py.
- Create tests/test_dreamer_v3_config.py.
- Create tests/test_dreamer_v3_oracle_manifest.py.
- Create tests/fixtures/dreamer_v3/README.md.

**Interfaces produced**

- DreamerProfile, ObservationMode, ModelSize, NetworkSize.
- All configuration dataclasses and resolve_dreamer_config.
- OracleManifest load/save/hash validation, OracleHarness process boundary and
  fixture writer, and the strict ParameterTranslator registry/consumption
  checks that later tasks populate with concrete mappings.

**TDD sequence**

1. Assert exact paper/current snapshots, model-size table, DMC mode defaults,
   immutable canonical hashes, and invalid combinations.
2. Assert manifests reject wrong source/config/file hashes, override maps, and
   schema versions; assert PAPER applies its declarative overrides without
   modifying the official checkout while UPSTREAM_CURRENT applies none.
3. Assert the harness names cases, writes deterministic NPZ/manifests, rejects
   unconsumed source/destination parameters, and can execute a config-only case.
4. Run focused tests; RED must be missing/new API or mismatched old values.
5. Implement config, manifest, harness boundary, and translator registry only.
6. Run focused tests, Ruff, and current Dreamer tests.

**Acceptance**

Every numeric value matches ARCHITECTURE.md. Profiles are constructed
independently rather than current config patched into paper config.

**Commit:** feat(dreamer): freeze conformance profiles

## Task 2: Port scalar transforms and output distributions

**Files**

- Create distributions.py.
- Create tests/test_dreamer_v3_distributions.py.
- Add distribution oracle NPZ/manifest fixtures.

**Interfaces produced**

- symlog, symexp, MSEOutput, BinaryOutput, NormalOutput, CategoricalOutput,
  OneHotOutput, TwoHotOutput, AggregateOutput.

**TDD sequence**

1. Generate official cases for extreme scalars, logits, supplied categorical
   noise, bounded normal actions, and 255-bin targets.
2. Write one failing behavior/oracle test per public method and event reduction.
3. Cover 1% probability mixing, hard/straight-through samples, clamping, ordered
   expectation, unreduced squared error, exact event aggregation, tanh mean plus
   sigmoid standard deviation for bounded-normal heads, and zero-logit
   expectation.
4. Implement minimal objects, then refactor shared reductions while green.

**Acceptance**

Scalar tolerance 1e-7; float32 outputs 1e-5. No shape-only/finite-only tests.

**Commit:** feat(dreamer): port output distributions

## Task 3: Port exact network primitives

**Files**

- Create networks.py.
- Create tests/test_dreamer_v3_networks.py.
- Add network oracle fixtures and parameter maps.

**Interfaces produced**

- Initializer, RMSNorm, Linear, BlockLinear, Conv2D, MLP, BlockGRU,
  DictEncoder, DictDecoder, MLPHead.

**TDD sequence**

1. Generate official parameters/outputs at deterministic small dimensions for
   paper stride and current pooling image paths.
2. Test parameter names/shapes/initial values, forward outputs, block
   isolation/mixing, image resolutions, key ordering, vector symlog, and head
   output families.
3. Implement classes in dependency order with a RED/GREEN cycle per class.
4. Require ParameterTranslator to consume every official and Flax parameter
   exactly once.

**Acceptance**

Dense recurrent weights, Flax default initializers, or a single image path for
both profiles are Critical failures. Float32 parity tolerance is 1e-5.

**Commit:** feat(dreamer): port network primitives

## Task 4: Port RSSM transitions and KL objectives

**Files**

- Replace rssm.py.
- Create tests/test_dreamer_v3_rssm_parity.py.
- Add RSSM oracle fixtures.

**Interfaces produced**

- RSSMState, RSSMTrajectory, RSSM methods in ARCHITECTURE.md.

**TDD sequence**

1. Test initial/reset states, supplied-noise categorical samples, prior and
   posterior steps, mid-sequence resets, scans, open-loop imagination, KL
   stop-gradient boundaries, free nats, and gradients.
2. Confirm old RSSM fails sampling/block-GRU/oracle assertions.
3. Implement state and single-step methods before scans.
4. Compare float32 states/logits/gradients at 1e-5 and bfloat16 at 2e-2.
5. Migrate package exports only when parity is green.

**Commit:** feat(dreamer): port discrete rssm

## Task 5: Implement online replay and latent writeback

**Files**

- Create replay.py.
- Create tests/test_dreamer_v3_replay.py.

**Interfaces produced**

- ReplayKey, ReplayBatch, ReplayChunk, ReplayWriter, OnlineQueue,
  UniformSelector, ConsecutiveStream, DreamerReplay.

**TDD sequence**

1. Test chronology, chunk sealing, non-overlapping online sequences,
   online-first fill, uniform valid starts including sequences crossing natural
   episode boundaries, forced leading is_first and next-is_first is_last
   annotation, capacity eviction, exact context-prefix/consecutive slicing,
   independent persistent train/report/eval streams and locks with no retained
   cross-mode slices, exact writeback, and complete save/restore.
2. Implement writer/chunk, then queue, selector, aggregate replay, persistence.
3. Add deterministic selection fixture and seeded statistical uniformity test.

**Acceptance**

State restore includes writers, queue, selector RNG, eviction order, latent
context, and each mode's current consecutive batch/index—not merely transition
arrays. Train, report, and eval streams resume independently; only train may
drain the online queue.

Persistence schema version 2 records every writer's complete lifetime
`chunk_history`, including evicted ids. Acceptance requires exact contiguous and
disjoint allocation provenance, per-writer counter/history/current-chunk
cadence (including the empty successor for `chunk_size=1`), retained-chain
suffix ownership and chronology, canonical sealed/open geometry, and unique
item and queue starts. In global item-id order, every writer's nonempty retained
item projection is the exact step-one suffix of its emitted starts ending at
`row_count - raw_length`; empty projections after global eviction are valid and
cross-writer interleaving is unconstrained. FIFO state is a canonical list of
exact Python integers. The live item map is exactly `dict[int, ReplayKey]` with
exact Python-integer keys; eviction and public validation reject bool and NumPy
integer aliases before equality or membership. Outer writer identities and
selector ids/indices likewise require exact Python integers.
Eviction preflights the complete FIFO/item/selector/ref-decrement operation
before its first mutation. Public validation shares the exact live selector
container, Python-integer key/index, uniqueness, bijection, and item-agreement
checks without mutation. A valid idle writer whose predecessor was evicted must
restore and continue bit-exactly.

For online replay, each writer's nonempty persisted queue projection is the
exact `raw_length`-spaced suffix of phase `1 % raw_length` through the latest
eligible start. Empty projections, stale/live entries, `raw_length=1`, and
arbitrary cross-writer interleaving remain valid; offline persisted queues must
be empty. Impossible direct offline queue injection is ignored without draining
it or changing uniform selector RNG, metrics, or reported queue size.

Persisted consecutive batches are validated operationally even after their
backing chunks are evicted: annotations must obey leading-first,
terminal-to-last, and adjacent last/first equality, while all step ids must be
allocated and logically consecutive. Every live retained position additionally
binds all declared immutable transition leaves to backing storage after
reconstructing leading-first and next-first-is-last annotations. Mutable latent
leaves are exempt because latent writeback may legitimately diverge from the
sampled copy; fully evicted positions retain ledger/annotation validation only.
Restore also enforces
`sampled_sequences == batch_size * sample_calls == online_samples +
uniform_samples`, with the latter equality read as a separate identity, and
accepts the all-zero metrics state after reset. Every rejection is
transactional. Task 5 is accepted only after the focused corruption/control
matrix, the complete replay file, accepted Dreamer regression matrix, oracle
provenance/regeneration checks, fixture hash check, and JEPA-scope diff all pass.

**Commit:** feat(dreamer): add online latent replay

## Task 6: Port the unified agent objective

**Files**

- Create normalization.py and agent.py.
- Create tests/test_dreamer_v3_agent.py.
- Add policy/loss oracle fixtures.

**Interfaces produced**

- PercentileNormalizer, SlowValue, AgentCarry, AgentLoss, DreamerAgent.

**TDD sequence**

1. Generate official fixtures for preprocessing, policy train/eval, exact
   replay-action/previous-action alignment, replay-context normal/reconstructed
   carry paths, all world losses, all-state imagination, lambda returns,
   REINFORCE, critic, replay critic, EMA regularizer, entropy, masks, and the
   weighted total.
2. Write a failing value and gradient-boundary test for every loss component.
3. Implement preprocess/policy, world loss, imagination/returns, actor,
   critics, aggregation in separate cycles.
4. Assert component names/reductions, values, and gradients at 1e-5.
5. Remove losses.py, models.py, imagination.py only after imports migrate.

**Acceptance**

Imagination starts at every valid replay state. Actor is score-function
REINFORCE. Reward/value/reconstruction switches detach only specified terms.

**Commit:** feat(dreamer): port unified agent objective

## Task 7: Port AGC, LaProp, and train_step

**Files**

- Create optimizer.py.
- Create tests/test_dreamer_v3_optimizer.py.
- Create tests/test_dreamer_v3_train_step_parity.py.
- Add optimizer and five-update fixtures.

**Interfaces produced**

- DreamerOptimizerState, DreamerOptimizer, DreamerTrainState, train_step.

**TDD sequence**

1. Test per-tensor AGC/floor, RMS-before-momentum, RMS and momentum bias
   correction, both beta2 values, epsilon, warmup/schedule, slow critic,
   normalizer, RNG split, counters.
2. Confirm old Adam path fails ordering/oracle tests.
3. Implement transforms in documented order.
4. Run five fixed replay updates; require parameters/state/metrics/writeback
   within 1e-4.
5. Remove training.py after import migration.

**Commit:** feat(dreamer): port laprop training updates

## Task 8: Implement DMC20 environment parity

**Files**

- Create dmc.py.
- Modify world_marl/envs/dmc_pixel_adapter.py.
- Create tests/test_dreamer_v3_dmc.py.

**Interfaces produced**

- DMCSpec, DMCEnvironment, DMCVectorEnvironment.

**TDD sequence**

1. Test 20-task validation, camera mapping, 64x64 uint8 images, modality
   exclusion, clip-before-scale action handling/repeat, deterministic instance
   seeds, terminal versus truncation, and auto-reset record ordering.
2. Use real dm_control specs. Small dm_env fakes are allowed only to inject
   boundary cases, never as end-to-end evidence.
3. Run renderer-independent focused tests and mark real render smoke integration.

**Commit:** feat(dreamer): add canonical dmc20 environments

## Task 9: Implement driver, artifacts, and exact resume

**Files**

- Create driver.py, artifacts.py, checkpoint.py.
- Create tests/test_dreamer_v3_driver.py.
- Create tests/test_dreamer_v3_checkpoint.py.

**Interfaces produced**

- DriverState, DreamerRunner, RunManifest, RunSummary, ArtifactWriter,
  CheckpointPayload, CheckpointManager.

**TDD sequence**

1. Test reset-step-policy-mask-transition/add/train order, same-row versus
   previous-action alignment, samples-per-insert math, online-first sampling,
   evaluation isolation, stop budget, atomic artifacts, and interruption.
2. Compare uninterrupted versus resumed train state, replay, queues, writers,
   driver, counters, and JSONL cursors.
3. Implement driver boundaries first, compose runner last.
4. Reject profile/config/schema mismatch on restore.

**Commit:** feat(dreamer): add online runner and exact resume

## Task 10: Replace CLI and benchmark contracts

**Files**

- Replace scripts/train_dreamer_v3_baseline.py.
- Modify scripts/compare_visual_wm.py and scripts/benchmark_dmc_pixels.py.
- Modify pyproject.toml only if entrypoint/extra changes are required.
- Create tests/test_dreamer_v3_cli.py and update comparison tests.

**TDD sequence**

1. Test every public flag, default paper profile, explicit current profile,
   legacy rejection, canonical override protection, resume, and dry-run without
   environment/model creation.
2. Replace precollect/offline phases with DreamerRunner.
3. Make comparison arm-aware for real steps/replay ratio.
4. Remove point_mass and synthetic image-grid data from canonical reports.

**Commit:** feat(dreamer): expose native paper baseline

## Task 11: Full parity and runtime gate

**Files**

- Rewrite tests/test_dreamer_v3_baseline.py as package conformance gate.
- Create tests/test_dreamer_v3_integration.py.
- Update ARCHITECTURE.md only with verified notes; never weaken contracts.

**TDD and verification**

1. Run all oracle cases for both profiles and verify provenance.
2. Run debug-model online collection, replay, update, eval, checkpoint, resume.
3. Run focused Dreamer/DMC/comparison tests, Ruff, full pytest.
4. If an already-authorized Linux EGL/CUDA runtime is available, run short
   Vision and Proprio real DMC smokes and inspect metrics, achieved ratio,
   checkpoint, and resumed progress. Otherwise emit the exact commands and mark
   this external-runtime check pending without launching paid compute.
5. Produce—but do not launch—the 20-task x 2-mode x 5-seed dry-run manifest.

**Implementation acceptance**

All local parity and online-system tests pass, both real-smoke commands are
materialized, and the dry-run scientific manifest is complete. Lack of an
authorized Linux GPU does not permit replacing those checks with fake evidence
and does not block code completion; it is reported explicitly.

**Deferred runtime/scientific acceptance (requires separate compute approval)**

The 200M Vision model must fit one 80GB A100 and reach at least 80% of official
throughput on equal hardware. The aggregate non-inferiority gate after full
runs is a lower 95% bootstrap bound of native-official IQM at least -0.05,
separately for Vision and Proprio. Task 11 implements these gates and their
reports but must not launch the runs without explicit authorization.

**Commit:** test(dreamer): verify native source parity

## Task 12: Whole-branch review and completion

1. Generate a review package from merge base through HEAD.
2. Dispatch a fresh highest-capability reviewer against ARCHITECTURE.md,
   PLAN.md, and accumulated Minor findings.
3. Send the complete finding list to one fixer, rerun covering tests, re-review.
4. Invoke verification-before-completion: Git status, focused tests, Ruff, full
   pytest, package import, CLI help, artifact inspection.
5. Invoke finishing-a-development-branch and present keep/merge/PR options.
   Never push or merge without explicit selection.
