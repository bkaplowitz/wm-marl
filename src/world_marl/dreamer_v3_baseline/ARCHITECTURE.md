# Native DreamerV3 Architecture

This document is the binding implementation contract for the native Flax port.
It is deliberately operational: every production class, state transition,
gradient boundary, and persistence boundary is defined here. An implementation
that produces finite losses or similar-looking networks but violates these
contracts is not DreamerV3.

## 1. Authority and conformance profiles

The default profile is **paper**. Explicit values and algorithms in Hafner et
al., “Mastering diverse control tasks through world models” take precedence.
Unspecified implementation details come from publication source revision
bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01. The secondary
**upstream-current** profile matches revision
e3f02248693a79dc8b0ebd62c93683888ddaccfe.

Profiles share code only where operations are identical. Differences are typed
configuration, have separate oracle fixtures, and appear in every run manifest.

| Setting | paper | upstream-current |
| --- | --- | --- |
| Image downsampling | stride-2 convolutions | convolution plus pooling |
| LaProp beta2 | 0.99 | 0.999 |
| DMC budget | 1,000,000 real transitions | 1,100,000 configured transitions |
| DMC Vision model | 200M | 200M |
| DMC Proprio model | 1M | 1M |

Unsupported combinations fail before initialization. There is no approximate
fallback.

## 2. Tensor and transition conventions

- B: replay batch, paper default 16.
- T: sequence length, paper default 64.
- L: stochastic latents, fixed at 32.
- C: categorical classes per latent, derived from model size.
- D: recurrent width, eight times model dimension H.
- A: flattened action width.

Sequences are [B,T,...]; single transitions are [B,...]. RSSM stochastic state
is [B,L,C] and flattens only when forming head features. Images are uint8 in
replay and converted once at preprocessing. Every transition contains
observations, actions, reward, is_first, is_last, and is_terminal. is_last marks
boundaries including truncation; is_terminal alone defines continuation target.

Replay follows the official temporal convention exactly. Stored row `t`
contains observation and reward received after executing `action[t-1]`, plus
the newly sampled `action[t]` for the next environment step. The world-model
transition into latent state `t` therefore consumes `prev_action[t]`, formed as
the training carry action prepended to `data[action][:, :-1]`; reconstruction,
reward, and continuation heads are evaluated at latent state `t`. The replay
action in the same row is never incorrectly treated as the action that produced
that row's reward. `is_first[t]` resets recurrent state before observing row `t`
and makes the cross-episode prepended action irrelevant.

## 3. Configuration classes

### DreamerProfile

Enum PAPER and UPSTREAM_CURRENT. Serialized by value, selected before config
resolution, immutable during a run.

### ObservationMode

Enum VISION and PROPRIO. VISION excludes proprio state; PROPRIO excludes pixels.
Mixed observations are noncanonical.

### ModelSize

Enum 1M, 12M, 25M, 50M, 100M, 200M, 400M. resolve returns NetworkSize: model
dimension H, recurrent D=8H, convolution base channels H/16, categorical
classes C=H/16. The 1M profile has four base channels and four classes.

### RSSMConfig

Fields: deter, hidden, stoch, classes, blocks, free_nats, unimix, activation,
normalization, image_layers, observation_layers, dynamics_layers, absolute,
initializer, output_scale. Validate 32 stochastics, eight blocks, D divisible
by eight, positive classes, unimix in [0,1), and nonnegative free nats.

### EncoderConfig

Convolution depth/multipliers/kernel/stride behavior, vector MLP layers/units,
activation, normalization, initializer, symlog, and outer-dimension handling.
It decides architecture only and never selects a cheaper runtime path.

### DecoderConfig

Encoder mirror plus image output and bias spaces. Paper uses transposed stride-2
convolutions. Output activations/likelihoods belong to decoder outputs.

### HeadConfig

Layers, units, activation, normalization, output family, output scale,
initializer, bins. Reward/value use 255-bin symexp two-hot and zero output
weights. Continue uses binary output.

### PolicyConfig

Layers, units, activation, normalization, min/max std, output scale, unimix,
initializer, and discrete/continuous distribution families. DMC uses bounded
normal actions.

### OptimizerConfig

Learning rate, AGC threshold/floor, epsilon, beta1/beta2, momentum, weight
decay, schedule, warmup, anneal. Profile supplies beta2; callers cannot patch it.

### ReplayConfig

Capacity, chunk size, online flag, selector fractions, context, sequence length.
Canonical capacity is 5,000,000 with uniform fraction one and no priority or
recency fraction.

### RunConfig

Real-step budget, replay ratio, environment counts, evaluation settings,
cadences, batch shape, device policy. Desired gradient updates per real
transition are replay_ratio/(batch_size*batch_length*action_repeat).

### DreamerV3Config

Immutable aggregate plus profile/mode. resolve_dreamer_config builds from
authority tables; validate enforces cross-component invariants; canonical_hash
losslessly hashes sorted values. Nothing mutates after hashing.

## 4. Distribution classes

All implement the official output protocol: pred, sample(seed), logp, prob,
entropy, kl, and loss where the operation is defined. Batch dimensions remain;
event dimensions reduce exactly once. Native convenience aliases may exist, but
agent code and parity tests use these official method names.

### MSEOutput

Unit-variance continuous reconstruction. pred is mean; loss returns the
elementwise squared error with a stop-gradient target. It does not apply a 0.5
factor or reduce event dimensions.

### AggregateOutput

Wraps another output and a count of trailing event dimensions. pred and sample
pass through. loss, entropy, and KL apply the configured reduction over exactly
those trailing axes; logp sums them. RSSM categorical KL uses one trailing
axis, vector heads use the space rank, and image reconstruction sums its three
HWC axes.

### BinaryOutput

Bernoulli logits. pred thresholds sigmoid at 0.5; loss is stable BCE. Continue
target is 1-is_terminal, never 1-is_last.

### NormalOutput

Plain diagonal normal with elementwise mean/std. For the “bounded_normal” policy
head, mean is tanh(raw_mean) and std is
(maxstd-minstd)*sigmoid(raw_std+2)+minstd before constructing NormalOutput.
Samples themselves are not tanh-transformed and logp has no transform
Jacobian; environment action handling is tested separately.

### CategoricalOutput

Converts logits to probabilities, optionally mixes 1% uniform, renormalizes,
then performs every operation. It owns no straight-through rule.

### OneHotOutput

Hard one-hot values with straight-through probability gradients. Supplied-noise
oracle cases require exact samples.

### TwoHotOutput

Builds 255 bins by symexp of uniform symlog coordinates. loss clamps targets and
interpolates adjacent bins. pred sums negative and positive expectations
separately from small to large magnitude before adding.

## 5. Network classes

### Initializer

Maps source names to Flax initializers, including truncated-normal fan-in and
exact zero output weights. Unknown names fail.

### RMSNorm

Divides by root mean square over final dimension, then learned scale/optional
bias. It never subtracts mean.

### Linear

Affine projection followed by configured norm/activation. Parameter names and
initializers are stable for oracle translation.

### BlockLinear

Independent matrices for eight blocks using block-diagonal multiplication. A
dense recurrent matrix is forbidden.

### Conv2D

Implements paper stride/transposed-stride and current pooling/upsampling paths.
Kernel, padding, channel order, init, norm, activation match profile fixtures.

### MLP

Exactly configured hidden count plus output projection. No implicit flattening
or undeclared sharing.

### BlockGRU

Flattens stochastic state and divides each action component by a stop-gradient
max(1,abs(action)). Separate dense+RMSNorm+SiLU projections encode deter,
stoch, and action. Their concatenation is repeated into eight groups and
concatenated with grouped deter. Configured BlockLinear hidden layers run, then
one BlockLinear emits reset, candidate, update in that order. The exact update
is reset=sigmoid(reset), candidate=tanh(reset*candidate),
update=sigmoid(update-1), deter=update*candidate+(1-update)*old_deter.
is_first zeroes deter, stoch, and action before this operation.

### DictEncoder

Partitions declared image/vector keys. Images use Conv2D. Vectors are cast,
symlogged, sorted by key, concatenated, then vector MLP. Metadata is rejected.
Sorted uint8 image keys concatenate on channels and convert to float/255-0.5
before the convolution stack. Current pooling is 2x2 max pooling, not averaging.

### DictDecoder

Returns one output object per key. Images use conv decoder, vectors MLP heads.
Image logits pass through sigmoid and are wrapped as MSEOutput then
AggregateOutput over HWC. Agent reconstruction targets are float(image)/255,
without the encoder’s -0.5 shift. Vector continuous outputs use symlog-MSE and
discrete vector outputs categorical heads. Decoder never reduces batch/time.

### MLPHead

Creates reward, continue, policy, or value output from features. Family comes
only from typed config.

## 6. RSSM classes

### RSSMState

Immutable Flax struct deter [B,D], stoch [B,L,C]. initial zeros; feature
concatenates deter and flattened stoch; reset zeroes both where is_first.

### RSSMTrajectory

Posterior/prior states/logits, features for [B,T], and final state for replay
writeback. Constructor validates all shapes.

### RSSM

- initial: zero RSSMState.
- img_step: embed state/action, BlockGRU, prior logits, mixed sample.
- obs_step: reset, img_step, posterior logits from deter+embedding, sample.
- observe: scan obs_step over T without crossing resets.
- imagine: scan img_step without observations.
- dyn_loss: KL(stop_gradient(post)||prior), clamped by free_nats.
- rep_loss: KL(post||stop_gradient(prior)), clamped by free_nats.

Sampling is never argmax. Posterior information is never injected after
imagination starts.

## 7. Agent and optimizer classes

### PercentileNormalizer

EMA 5th/95th return percentiles from stop-gradient returns. Calling it returns
`offset=low` and `scale=max(high-low,1)` but does not itself transform inputs.
Paper update rate is 0.01 (decay 0.99), limit is 1, and debias is false. The
stored state is low/high; a correction accumulator exists only when a
noncanonical profile explicitly enables debias. Canonical return normalization
uses the scale but not the offset for actor advantage:
`adv=(return-target_value)/return_scale`. The configured value and advantage
normalizers are identity implementations, so their offset is zero and scale is
one. The return offset is used only for normalized diagnostics.

### SlowValue

EMA critic parameters at update rate 0.02. Never gradient-updated.

### AgentCarry

Encoder carry, RSSM dynamics carry, decoder carry, and previous model action
tree. `initial(batch_size)` obtains all three component carries and creates a
zero action of each action-space dtype/shape. Both policy inference and replay
training use this exact structure. Individual component scans reset their
state where is_first is true; the aggregate carry is not replaced wholesale.

### AgentLoss

Total scalar; one reconstruction component per decoder observation key plus
rew, con, dyn, rep, policy, value, and repval; per-step writeback tensors;
metrics; updated normalizers. The configured scalar `rec` scale is expanded to
every decoder key and those key losses are summed independently—there is no
single pre-averaged reconstruction loss. Metrics never feed loss.

### DreamerAgent

- preprocess converts values once and retains reconstruction targets.
- initial returns AgentCarry.
- policy encodes, updates posterior, samples the policy in both training and
  evaluation exactly as upstream, returns action, carry, latent context, and
  per-tree finite diagnostics.
- apply_replay_context first constructs previous actions by prepending the
  action stored in AgentCarry to replay actions excluding the last row. With
  context K, both candidate paths drop the first K observations and step ids.
  The normal path also drops the first K constructed previous actions. The
  replay path reconstructs each component carry by truncating its stored
  encoder/dynamics/decoder entries over the K-row prefix and uses replay actions
  `[:, K-1:-1]` as previous actions. Per batch item, replay path is selected
  exactly when `consec[:, 0] == 0`; later consecutive slices select normal path
  and continue from the incoming training carry. Training writes newly inferred
  component entries back at the returned step ids and stores the final replay
  action in the outgoing AgentCarry.
- loss observes replay, computes world model, imagines from every valid
  posterior state, computes policy/value/replay-value, combines scales once.
- imagine runs prior+policy exactly 15 steps retaining rewards, continues,
  values, log_probs, entropies.
- lambda_return is reverse recurrence with continuation discounts/bootstrap.
- report is diagnostic and state-pure.

World model is the sum of every observation-key reconstruction loss plus
rew+con+dyn+0.1*rep. Complete objective adds policy+value+0.3*repval. Actor is
REINFORCE `logp(stop_gradient(action))` times stop-gradient normalized advantage
plus entropy, never a pathwise action gradient. Imagination weights are also
stop-gradient. Canonical `ac_grads=false` stops all imagined features before
actor/value losses; `reward_grad=true` allows reward reconstruction gradients
into the representation; `repval_grad=true` allows replay-value gradients into
the representation; `slowtar=false` uses the online value head as the lambda
return bootstrap/target-value source. Critic targets and slow-value predictions
inside the regularizer are stop-gradient. The slow critic regularizes value but
does not supply canonical lambda-return targets.
When continuation discounting is enabled, continue target is
(1-is_terminal)*(1-1/horizon). Imagination weights are the cumulative product
of discount*predicted_continue divided by the leading discount. Replay-value
weights are ~is_last. lambda_return uses live=(1-is_terminal[t+1])*discount and
cont=(1-is_last[t+1])*lambda in its reverse recurrence, so truncation and
termination affect different factors.

### DreamerOptimizerState

Step, RMS tree, momentum tree, schedule state matching parameter tree.

### DreamerOptimizer

Mandatory order: per-tensor norms; AGC 0.3 with 1e-3 parameter floor; RMS update
with profile beta2/epsilon 1e-20 and step-count bias correction; normalize
gradient; momentum beta1 0.9 with its own step-count bias correction;
warmup/schedule and learning rate 4e-5; apply delta. RMS-before-momentum is
LaProp; omitting either bias correction or reordering the stages is a
conformance failure.

### DreamerTrainState

Parameters, optimizer, slow critic, return normalizer, RNG, real/update counters,
config hash. train_step splits RNG deterministically, computes unified gradient,
applies optimizer, updates slow value/normalizer once, returns writeback/metrics.

## 8. Replay classes

### ReplayKey

Chunk id plus offset identity for exact latent writeback.

### ReplayBatch

[B,T,...] transitions and [B,T] step ids. Validates shapes and linked-storage
chronology but deliberately permits natural episode boundaries inside a
sequence. Sampling copies the data, forces the first returned row's is_first to
true, and sets `is_last[t] |= is_first[t+1]` so abandoned/reset boundaries are
visible without changing is_terminal. RSSM scans reset at each is_first.

### ReplayChunk

Fixed-capacity transition and latent storage. `append` first validates the
complete exact transition-plus-latent row, including the policy-produced
float32 latent shapes, and only then allocates or mutates storage; a rejected
row cannot leave zero-filled arrays or advance length. The accepted row stores
the policy's initial latent entries exactly. A full chunk is sealed to exactly
one successor and its transition prefix becomes read-only, while latent context
remains replaceable by key. Restore accepts only full sealed chunks and nonfull
open chunks, both at the configured aggregate chunk size.

### ReplayWriter

One per environment. It owns an ordered lifetime `chunk_history` containing
every chunk id allocated for that worker, including chunks later evicted. It
enforces chronology, opens/seals chunks, retains the last `R-1` keys, emits
every valid uniform start, and enqueues non-overlapping online starts satisfying
`row % R == 1 % R`, where `R=context+sequence_length*consecutive`. This is
`1+nR` for `R>1` and every row for `R=1`. The row and emitted
counters, pending suffix, current empty successor, and lifetime history are one
cadence invariant. Episode boundaries are data flags, not replay
discontinuities. On restore an empty current chunk may legitimately have no
retained predecessor when capacity eviction removed it; the persisted final
`is_last` flag still controls the next append.

### OnlineQueue

FIFO fresh sequences, drained before uniform replay fill, fully checkpointed.
For each writer, a nonempty persisted queue projection must be the exact
`R`-spaced suffix of all emitted online starts, with phase `1 % R`, ending at
that writer's latest eligible start. Entries from different writers may remain
interleaved in any collection order. An empty per-writer projection is valid
after service or eviction. Offline replay persists no queue; even an impossible
direct in-memory queue injection is ignored by sampling and statistics without
changing the queue, selector RNG, or sample metrics.

### UniformSelector

Uniform over every inserted start with enough linked successor transitions,
including starts whose sequence crosses one or more is_first boundaries.
Incomplete sequences are not inserted. Canonical profiles permit no
priority/recency.

### ConsecutiveStream

Stateful wrapper around one replay sampling mode. The raw replay sequence has
exactly `context + sequence_length * consecutive` transitions. For consecutive
index `i`, it returns the slice starting at `i * sequence_length` and ending
after `context + sequence_length` transitions, annotates every time step with
`consec=i`, and only fetches a new raw replay sequence after all consecutive
slices are consumed. Each instance retains its own current raw batch and index;
save/restore includes both. The canonical profiles use one consecutive slice,
but this class is still mandatory because replay-context restoration keys off
`consec[:, 0] == 0`.

### DreamerReplay

Owns chunks, writers, queue, selector, eviction/context, and one persistent
ConsecutiveStream plus serialization lock for each of train, report, and eval.
`sample(mode)` advances only that mode's stream. The train stream drains online
then uniformly fills its raw sequence; report and eval sample uniformly and
cannot consume the online queue. A retained raw batch or later consecutive slice
can never cross from one mode to another. `update_context` writes inferred
encoder/dynamics/decoder entries to the exact sequence beginning identified by
the first step id. Agent replay-context application may replace the context
prefix only for the first consecutive slice; each component's truncate method
consumes whatever stored prefix entries its official counterpart consumes and
falls back to the incoming carry exactly as its counterpart does. Later slices
perform the ordinary continuation scan. save/restore preserves every chunk
link/reference, current writer, queue, selector RNG, FIFO item, latent entry,
and the independent current raw batch and consecutive index for all three mode
streams. Restore rebinds blocked samplers to the restored stream for their own
mode without allowing cross-mode reuse.

`add` validates and normalizes a complete row before mutation, requires the
exact active writer, preflights every needed id, appends the transition and
policy latent entries, updates pending/item references, seals and allocates a
successor when full, emits the valid start, applies the online phase, and only
then advances writer and aggregate counters. New chunk allocation verifies its
owner before changing the id counter, chunk map, references, or lifetime
history.

Persistence schema version 2 treats writer histories as the allocation ledger.
Their exact positive 16-byte ids must be strictly increasing per writer,
disjoint across writers, and together cover precisely
`1..next_chunk_id-1`; each retained writer chain is the suffix of its history.
Restore builds a candidate and rejects it before live-state replacement unless
chunk geometry/ownership, per-writer chronology and cadence, item-key
uniqueness, refcounts, selector types/order, FIFO counters, and the exact sample
metric identities all agree. Raw persisted chunk size and length are checked
against the configured bounds before chunk construction or NumPy allocation.
The live item map is an exact dictionary from exact Python integers to exact
`ReplayKey` values; both eviction preflight and public validation reject
Python-equal aliases before equality or membership checks. The FIFO is a list
of exact Python integers and equals retained item-id order.
Each writer's nonempty item-key projection, read in global item-id order, is
the exact step-one suffix of every emitted valid start and ends at
`row_count-R`; a writer may have an empty projection after global capacity
eviction, and no global ordering is imposed across writers. Eviction validates
the complete FIFO/items and selector key/index bijection plus every recursively
triggered reference decrement before any selector, FIFO, item, chunk, or
refcount mutation begins. Public validation applies the same exact selector
list/dictionary types, Python-integer key/index types, uniqueness, bijection,
and selector/items agreement without mutating state.

Stale online keys remain valid only when their id was allocated by that writer,
their offset and logical row exist, and their nonempty per-writer projection is
the exact suffix of the `1+kR` cadence ending at the latest eligible start;
live keys must additionally resolve `R` rows. Cross-writer interleaving remains
unconstrained, while an offline persisted queue must be empty.

Each persisted consecutive current batch must begin with `is_first`, satisfy
`is_terminal => is_last` and
`is_last[:,:-1] == is_first[:,1:]`, and contain a fully consecutive logical
step-id sequence in the allocation ledger. These checks still apply when its
backing chunks were evicted. At every position whose backing chunk remains
live, every declared immutable transition leaf must equal linked storage after
reconstructing sampling annotations (`is_first[:,0]=True` and
`is_last[t] |= backing_is_first[t+1]`). Persisted latent leaves are deliberately
not compared because `update_context` may validly replace them after sampling;
fully evicted prefixes retain ledger-and-annotation validation only. The exact
metric identities are
`sampled_sequences == batch_size * sample_calls` and
`sampled_sequences == online_samples + uniform_samples`, including the all-zero
state produced by `stats(reset=True)`.

## 9. Environment, driver, artifacts, checkpoints

### DMCSpec

Immutable task/mode/seed/image/camera/repeat/time-limit. Canonical validates the
published 20 tasks and action repeat one.

### DMCEnvironment

One dm_control env. reset emits is_first. step first clips the policy action to
[-1, 1], then linearly maps it to the dm_control action bounds, applies action
repeat, keeps terminal distinct from truncation, and returns uint8 64x64 RGB or
vectors. This reproduces official `ClipAction(NormalizeAction(DMC))` ordering.
The clipping/scaling is an environment-side view: replay retains the raw sampled
policy action, except that the driver stores zeros on is_last rows. VISION never
leaks state.

### DMCVectorEnvironment

Owns 16 independent envs by default, batches transitions, and uses
deterministic per-instance seeds. Reset is an explicit boolean action: reset
true, or an already-finished underlying dm_env, causes reset and ignores the
continuous action. The returned reset observation has is_first true; terminal
and truncation flags from the preceding row remain in that preceding replay
row rather than being overwritten.

### DriverState

Current action tree (including reset), AgentCarry, episode returns/lengths,
real-step count. Initialization uses zero continuous actions and reset=true for
every environment.

### DreamerRunner

1. Restore/initialize agent, replay, envs, driver, writers, zero actions, and
   reset=true actions.
2. Step each environment with the driver's previously selected action/reset.
3. Pass the returned observation and AgentCarry to policy; sample the next
   action and latent context.
4. Where returned is_last is true, replace the next continuous action by zero
   and set its reset action true.
5. Assemble replay row as returned observation/reward plus the newly selected
   action and latent entries, then count, append, and log it in that order.
6. Advance the ratio limiter by the number of inserted rows; after the replay
   warmup, run every update owed, sampling, training, writeback, and aggregating
   metrics per update.
7. Evaluate on separate seeds at cadence without changing training carry,
   replay, limiter, counters, normalizer, or RNG.
8. Atomically checkpoint at cadence/interruption and stop at the exact profile
   real-transition budget.

Collection and learning remain interleaved; fixed offline training is forbidden.

### RunManifest

Immutable profile, authority hashes, resolved config/hash, task/mode/seed,
dependencies, devices, command, canonical flag written before training.

### RunSummary

Final/interrupted status, real steps, updates, achieved ratio, returns, runtime,
checkpoint, failure. No invented counters.

### ArtifactWriter

Append-only metrics/scores JSONL and atomic manifest/summary. Flushes before
checkpoint completion and resumes without duplicate committed steps.

### CheckpointPayload

Versioned train state, replay, DriverState, environment seed/reset state,
manifest hash, artifact cursors.

### CheckpointManager

Writes temporary sibling and atomically renames. restore rejects
schema/config/profile mismatch. Resume continues same run, never warm-start.

## 10. Oracle classes

### OracleSourceSpec

Pins the exact official file hashes for both authority revisions and the
allowed execution dtypes for one oracle family. A source may attach a pure
generator-provenance validator and a live invocation resolver. Each callback
has a stable contract id, and each may be marked required. Construction and
registration independently reject a required source without its callable.
Registry equality compares hashes, dtypes, required flags, and callback ids,
not Python callable identity: re-importing a module refreshes equal callbacks,
while a changed id or contract fails closed. Reloading `oracle.py` rehydrates
every preserved entry into the new `OracleSourceSpec` class while retaining
its callbacks and ids. An object created before reload is accepted by creation
APIs only when its name and complete structural signature equal the current
registered entry; the current entry performs all subsequent work.

### OracleInvocation

The resolved, one-shot execution value. It contains an executable command,
resolved working directory, and canonical JSON stdin. It is never serialized
into an oracle manifest. A source resolver constructs it from the stable
manifest recipe plus caller-supplied runtime coordinates.

### OracleManifest

Official commit/file hashes, profile hash, JAX/dtype/device, seed, tensor schema,
generator command and canonical generator request. Invalid manifest invalidates
fixture. Validation first resolves the named `OracleSourceSpec`, completes the
generic authority/config/dtype/device checks, parses the request as one JSON
object, and requires a nonempty command and nonnegative seed. It then invokes
the source validator before any supplied-checkout Git read, and only afterward
checks official object bytes and fixture bytes/schema.

`resolve_generator_invocation()` first repeats ordinary source-free manifest
validation, so a manually constructed or subsequently replaced manifest cannot
bypass generic authority/config/device/command checks. It then dispatches the
registered source resolver. Sources without a resolver retain their recorded
command/request behavior; a source that requires resolution fails if its
resolver is unavailable.

The replay source validator requires the exact request key set and binds
case name `replay`, seed 7, profile, observation mode, revision, overrides,
source name, and float32 execution dtype to the manifest. Cases, row schema,
debug UUID mode, Elements helper hashes, frozen CPython/NumPy versions, local
shim hashes, and isolated-worker mode must equal the source-owned contracts.
The persisted command is the location-independent descriptor
`[python:current, module:world_marl.dreamer_v3_baseline.replay_oracle,
_worker]`. The persisted request contains no checkout, interpreter, Elements
package, or distribution path. Instead it records raw SHA256 digests for the
complete local generator closure: `replay_oracle.py`, `oracle.py`, and
`config.py`. `replay_oracle_contract.py` stores the frozen descriptor, runtime,
shim, and closure contracts outside that closure, avoiding a self-hash cycle.
Public load compares only persisted data to these literals: it never reads live
source, calls `inspect.getsource`, or inspects the current interpreter.

The replay resolver rechecks all three live raw file digests, all shim digests,
the frozen Python/NumPy identity, and the pinned Elements version/helper files.
It resolves the current absolute interpreter and worker, plus the supplied
official checkout and current Elements paths, into an exact one-shot envelope
with keys `request` and `execution`. `execution` contains only
`official_checkout`, `python_executable`, `elements_package_dir`, and
`elements_dist_info`; these coordinates are never saved. The isolated worker
requires that exact envelope and independently repeats stable-request, live
closure, interpreter, Elements, official-source, case, and row-schema checks
before executing official source. Copying the package therefore preserves
manifest validity while resolving the copied worker, and any byte change in
any of the three local generator files prevents resolution and execution.

### ParameterTranslator

Explicit Ninjax-to-Flax mapping with required transpose/reshape. Every source
and destination parameter is used exactly once; missing/duplicate/unused fails.

### OracleHarness

Runs named cases against an explicitly supplied official checkout and writes
NPZ plus manifest. Cases cover config, distributions, networks, RSSM, loss,
optimizer, replay, five train steps. Ordinary tests never launch wrapper.
PAPER uses the publication checkout plus declarative, in-memory overrides for
the paper's explicit stride-convolution, beta2=0.99, and 1,000,000-step values;
it never edits the checkout. UPSTREAM_CURRENT uses the current checkout with no
paper overrides. The manifest records the base revision and complete override
map, and the harness constructs DMC identically for both profiles. A paper case
is invalid if it silently inherits a conflicting current default.

## 11. Mandatory anti-simplification rules

Forbidden: argmax categorical state; Adam; dense GRU; linear symlog bins; global
MSE; pathwise actor gradient; final-state-only imagination; fixed offline batch;
is_last continuation target; replay without online queue/latent writeback;
parameter-only checkpoint; shape/finite tests claimed as parity.

Every class requires behavior tests. Every official counterpart requires oracle
parity before completion.
