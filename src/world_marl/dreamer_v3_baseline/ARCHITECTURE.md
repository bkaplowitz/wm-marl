# DreamerV3 production architecture

This is the implementation contract for the native JAX/Flax DreamerV3
baseline. It describes the system that must exist at completion, not the
behavior of the obsolete baseline or the acceptance status of partial ports.
Passing component tests is not evidence that the composed agent or online
system is conformant.

## 1. Authorities and profiles

Authority precedence is:

1. explicit repository safety and integration requirements;
2. the published experiment protocol for values selected by the `paper`
   profile;
3. Danijar Hafner's official implementation for operational and mathematical
   details not fixed by that protocol;
4. native wm-marl adapters, provided they preserve the preceding contracts.

The official source revisions are
`bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01` (published-source pin) and
`e3f02248693a79dc8b0ebd62c93683888ddaccfe` (upstream-current pin). The
Dreamer, configuration, replay, and DMC files have identical Git blobs at both
revisions. Revision labels therefore do not justify two
different implementations or fixtures. Differences between conformance
profiles are explicit configuration snapshots and are recorded in manifests.

`DreamerProfile.PAPER` is the default. It is the 2025 Nature published profile,
not the superseded 2023 arXiv-v1 experiment. Its DMC contract is 20 tasks for
both Vision and Proprio, 1,000,000 environment steps, action repeat 1, and the
200M model. The implementation-critical profile differences are:

| Setting | `paper` | explicit `upstream-current` |
| --- | --- | --- |
| Image encoder/decoder | stride-2 convolution/transposed convolution | convolution plus pooling/upsampling (`strided=False`) |
| LaProp | AGC 0.3, beta1 0.9, beta2 0.99, epsilon 1e-20 | AGC 0.3, beta1 0.9, beta2 0.999, epsilon 1e-20 |
| DMC configured steps | 1,000,000 | 1,100,000 |
| DMC action repeat | 1 | 1 |
| Default DMC model | 200M in both modes | 200M Vision; 1M Proprio source preset |

`DreamerProfile.UPSTREAM_CURRENT` must be selected explicitly and resolves the
pinned current source defaults without silently inheriting paper overrides.
Runtime identity is exactly the selected profile, complete resolved
configuration, config hash, pinned authority revision, and explicit override
map. Production runs do not hash or authenticate source trees. Numerical
fixture manifests alone record official source hashes for the exact blobs used
to generate their arrays; those hashes are test evidence, not runtime state.

The Nature paper and pinned `dreamerv3/agent.py` agree on actor gradients. The
implementation evaluates policy log probability at a stopped sampled action
and multiplies it by a stopped normalized advantage for both shipped continuous
and discrete policies. The superseded 2023 arXiv-v1 prose described a pathwise
continuous-action alternative; it is not an authority for this profile.

## 2. Canonical trees, shapes, and time alignment

Symbols used below are: collection environments `N`, replay batch `B`, model
sequence `T`, replay context `K`, imagination horizon `H`, stochastic variables
`S`, categorical classes `C`, deterministic width `D`, flattened model-action
width `A`, and observation event shape `E`.

### 2.1 Observation and action trees

One environment observation is a dictionary whose leading shape is `[N]` in
vector collection and absent for one unbatched environment:

- `image`: `[N,64,64,3]`, `uint8`, Vision only;
- declared proprio keys: `[N,*E]`, native dm_control floating dtype converted
  to `float32` at the adapter boundary, Proprio only;
- `reward`: `[N]`, `float32`;
- `is_first`, `is_last`, `is_terminal`: `[N]`, `bool`.

Metadata never enters the encoder. Vision excludes proprioceptive values;
Proprio excludes pixels. A model-action tree contains one leaf per declared
control action, `[N,A]` `float32`, expressed in the environment's normalized
coordinate system. Policy samples are intentionally unbounded: neither the
Agent nor replay clips them to the nominal `[-1,1]` action-space interval.
`TensorSpace.low`/`high` are declaration and serialization metadata only; they
are not Agent or replay validation bounds. Replay stores every finite non-final
raw sample unchanged. Only the DMC call boundary clips each
raw sample to `[-1,1]` and scales it to the native Control Suite bounds. The
driver's environment-action tree additionally owns `[N]` boolean `reset`.
`reset` is control state, not a policy output.

### 2.2 Replay rows and batches

A replay row contains the observation and reward returned by an environment,
boundary flags, the newly sampled action for the *next* environment call, the
writer identity, and optional encoder/RSSM/decoder context entries emitted by
policy inference. The row at logical time `t` was produced by action `t-1` but
stores action `t`. On every `is_last` row, whether terminal or truncated, the
stored next model action must already be zero and the driver's separate next
reset flag is true. `DreamerReplay.add()` never rewrites an action leaf: it
accepts an already-zero final-row tree and rejects a nonzero final-row tree
transactionally. `DreamerRunner` alone masks the just-sampled action to zero
before assembling the final replay row.

Training batches are batch-major `[B,T,...]`. The previous action consumed by
the latent transition is exactly

```text
prev_action = concat(carry.prev_action[:, None], data.action[:, :-1], axis=1)
```

after replay-context trimming. `is_first[:,t]` resets recurrent state before
observing row `t`, so any prepended cross-episode action is ignored. A sampled
batch also carries `stepid: [B,T,20] uint8` and `consec: [B,T] int32`; these are
sampling/writeback annotations, not model observations.

### 2.3 Model, carry, parameter, RNG, and state trees

- `RSSMState`: `deter [B,D]`, `stoch [B,S,C]`, floating compute dtype.
- `RSSMTrajectory`: posterior and prior states/logits `[B,T,...]`, features
  `[B,T,D+S*C]`, and final state.
- `AgentCarry`: encoder carry, `RSSMState`, decoder carry, and previous action
  tree `[B,*action_shape]`. Policy, train, and report use the same structure.
- Parameters: a Flax parameter pytree with stable module/path names for
  one-to-one oracle translation. Optimizer, normalizers, and slow target are
  explicit state and never hidden in parameters.
- RNG ownership is explicit. `DriverState.scheduler` owns exactly two scalar
  `np.int64` call counters: `policy_call_counter` is shared by collection and
  evaluation policy calls, and `batch_seed_counter` is shared by successful
  train and report batch calls. `DreamerV3Config.seed` is the sole canonical
  public-seed owner for model initialization, model-call outer roots, and the
  named native DMC seed policy. Copies in canonical argv, manifests, `DMCSpec`,
  summaries, and checkpoint identity are derived equality-checked projections,
  not separate mutable owners. Replay selection is a separate official RNG
  domain whose fresh construction constant is always zero. Each model call derives
  one legacy `uint32[2]`
  outer key from that seed and its current counter as section 2.4 specifies;
  neither the outer key nor any Ninjax-style child/cursor is serialized.
  `DreamerTrainState` therefore owns no random key. `UniformSelector` owns its
  NumPy replay-selection generator, and each `DMCEnvironment` owns its
  task/environment generator.
  Fresh `DreamerReplay` constructs `UniformSelector(seed=0)` internally; the
  public run seed never enters replay construction. `UniformSelector(seed)`
  remains the narrow component constructor for direct selector tests.
  `UniformSelector` owns a `numpy.random.default_rng()` whose exact bit
  generator is `PCG64`; its JSON-safe state is the complete PCG64 mapping
  `{bit_generator: "PCG64", state: {state: uint128-as-Python-int,
  inc: uint128-as-Python-int}, has_uint32: 0|1, uinteger: uint32-as-Python-int}`.
  Both inner integers are in `[0,2**128)` and `uinteger` is in `[0,2**32)`;
  the outer selector state also stores exactly `bit_generator="PCG64"`, its
  dense index/key arrays, and that complete `rng_state` mapping.
  Fresh replay saves and restores the complete PCG64 state, so an advanced
  selector resumes at the exact next sample rather than reconstructing progress
  from the constant. Each DMC task owns the legacy `RandomState` state described
  in section 9.
  Each named stream has one serialized owner, one advancement site, and no
  duplicate checkpoint leaf; no function reads a global RNG. Candidate restore
  rejects unknown generators, missing or extra fields, out-of-range integers,
  and wrong shapes/dtypes before mutating live state; roundtrips preserve every
  array byte and tree key exactly.
- JAX-resident integer state uses `jnp.int32` scalar leaves because supported
  execution disables x64: the optimizer step, optimizer RMS/momentum steps,
  slow-target count, and train `update_count` are all shape `[]` `int32` and
  advance once per official train call. Host driver, limiter, environment,
  artifact, checkpoint, and ordinary replay
  counters, including both call counters, are scalar `np.int64` checkpoint
  leaves; JSON-facing inspection and
  metadata encode them as Python integers with an explicit signed-`int64`
  schema and range check. The limiter also carries scalar `np.float64` `avail`.
  Replay's Python-integer `next_chunk_id` and `next_item_id` identity cursors
  are the explicit exceptions to this signed-`int64` rule and have the wider
  cursor domains defined in section 6. Each owner
  performs an owner-local wraparound preflight at its actual mutation boundary;
  restore rejects values outside the declared range and counter invariants.
  There is no proposal-wide overflow transaction or coordinator byte-spy. No
  constructor or restore path requests `jnp.int64` and relies on JAX's
  x64-disabled downcast. Every counter is serialized by its sole owner and
  restored without reconstruction.

### 2.4 Exact counter-derived and within-call RNG schedule

The production runtime implements this schedule with NumPy/JAX and does not
import Ninjax. The schedule translates official
`embodied/jax/agent.py::_seeds`, `Agent.policy/train/report`,
`RSSM.observe/imagine`, and Ninjax 3.6.3. Ninjax
3.6.3 is pinned only as a Task-5 test/oracle-generation dependency and recorded
in the RNG fixture manifest; it is not paper/current profile identity. The
official project permits `ninjax>=3.5.1`, so an unpinned transitive version must
not silently redefine runtime behavior.

Define `draw_no_amount(cursor, reserve, count)` as the pure translation of
Ninjax 3.6.3 `seed(amount=None, reserve=16)`: for each draw, if `reserve` is
empty, compute `block = jax.random.split(cursor, 16)`, set `cursor = block[0]`,
and set `reserve = list(block[1:])`; return and remove `reserve.pop(0)`. Reserve
consumption is FIFO. Define `draw_amount(cursor, count)` as
`block = jax.random.split(cursor, count + 1)`, returning
`(block[0], block[1:])`; it neither reads nor changes the caller's reserve. This
is the exact `seed(amount)` behavior used by `nj.scan`.

For every outer call, the sole helper is the exact official operation
`default_rng([public_seed, int(counter)]).integers(0, np.iinfo(np.uint32).max, (2,), np.uint32)`.
Here `public_seed` is always the bound `DreamerV3Config.seed`; callers cannot
supply or checkpoint another outer-root seed.
The upper bound is exclusive, exactly as NumPy implements it. The raw parameter
initialization key `uint32([public_seed, 0])` is a separate one-time value and
does not read or advance either counter. In stable scheduler order, collection
and evaluation consume consecutive values of `policy_call_counter`; successful
train and report calls consume consecutive values of `batch_seed_counter`.
Fresh bootstrap initializes both counters to scalar `np.int64(0)`. Native cold
resume restores their actual checkpointed values. This deliberately improves
on upstream `_load`, which rewinds its prefetched batch counter to the update
counter and can repeat outer seeds.

A call first validates counter capacity, derives its key from the current
counter, and executes without mutating the counter. A policy counter increments
atomically when the validated policy carry/action is stored. A train counter
increments with the train-state/replay/limiter commit; a report counter
increments with the report carry/copied-output commit. Validation or model
failure before those boundaries consumes nothing. Artifact failure after a
successful report leaves only uncheckpointed in-memory advancement; cold
resume replays from the last durable counter. Cold restore itself derives no
key and advances no counter.

Policy train and evaluation modes have the same within-call draws. Starting
with the counter-derived policy key and an empty reserve, call
`draw_no_amount(..., M + 1)`: the first child is the single-step RSSM posterior
key and the next `M` children are action-leaf keys in sorted JAX pytree order.
The resulting context cursor and reserve are discarded after the call.

For one train call, split its counter-derived key by 16. Child `[1]` is the
discarded `nj.grad(...)._prerun` access key, child `[2]` is `loss_seed`, and the
cursor `[0]`, children `[3:]`, and every post-call Ninjax context remainder is
discarded. The native runtime does not execute the state-discovery pass, but it
discards the same child so the actual loss uses the official key. Within the
loss context:

The post-call Ninjax context remainder is discarded.

1. Compute `loss_block = split(loss_seed, 16)`. Its cursor is
   `loss_block[0]`; `observe_access_seed = loss_block[1]` is the discarded
   `nj.scan` prerun child and `loss_block[2:]` remains the FIFO reserve.
2. Apply `draw_amount(loss_block[0], T)`. The returned cursor is
   `post_scan_root`; the `T` children are `observe_iteration_seeds: [T,2]`.
   For each time index in scan order, split that iteration seed by 16 and use
   child `[1]` as `posterior_keys[t]`. Thus `posterior_keys: [T,2]`; each key
   samples the complete posterior array `[B,S,C]`, so `B` changes array shape,
   not key count. The iteration roots and children `[2:]` are discarded.
3. Pop `imagination_access_seed = loss_block[2]` from the still-independent
   FIFO reserve. It corresponds to the imagination `nj.scan` prerun and is
   discarded. Apply `draw_amount(post_scan_root, H)`, producing
   `imagination_scan_root` and `H` iteration seeds.
4. Let `M` be the number of policy action leaves in JAX pytree order (sorted
   dictionary paths). In each horizon iteration, start with that iteration seed
   and an empty reserve, call `draw_no_amount(..., M + 1)`, assign the first `M`
   children to the action leaves and the next child to the RSSM prior sample.
   This yields `imagination_action_keys: [H,M,2]` and
   `imagination_prior_keys: [H,2]`. The arrays sampled by those keys have leading
   `B*Kstart`; that leading size does not change key count. The helper's refill
   rule is binding when `M + 1 > 15`.
5. After the horizon scan, sample the final policy action from the loss
   context. Start at `cursor=imagination_scan_root` with the unconsumed FIFO
   reserve `loss_block[3:]`, call `draw_no_amount(..., M)`, and assign the result
   in the same action-leaf order as `final_action_keys: [M,2]`. This deliberate
   reuse of the pre-scan reserve, while `seed(H)` advanced the separate cursor,
   is part of the source behavior.

The correspondence is exhaustive for a shipped train call: posterior keys map
to `rssm.py::_observe` line 87, each horizon's action keys map to
`agent.py::sample` through `policyfn`, each horizon prior key maps to
`rssm.py::imagine(single=True)` line 100, and final action keys map to
`Agent.loss::lastact`. Initializer draws occur only during parameter creation;
the shipped network calls do not enable dropout. Neither belongs to a train
transition.

Report starts its loss directly from the counter-derived report key: the loss
uses the same steps 1--5 above without the outer gradient access/loss split.
After the loss, its still-live loss context performs the extra diagnostic
rollouts in exact source order. The prefix `observe` first consumes one FIFO
access child, then `draw_amount(current_cursor, T//2)`; each iteration uses
child `[1]` of its 16-way split as a posterior key. The suffix `imagine` then
consumes one FIFO access child and
`draw_amount(current_cursor, T - T//2)`; because recorded actions are supplied,
each iteration uses only child `[1]` as its prior key. Decoder calls draw no
keys. Canonical `report_gradnorms=false`; enabling it is outside both frozen
profiles until its repeated gradient-loss draw schedule has its own fixture.
The final report cursor/reserve is discarded.

Task 4d fixtures prove policy train/evaluation and report internal child order,
including posterior/action draws and report loss plus extra observe/imagine.
Task 5 proves the complete train child schedule and an interleaved outer-call
fixture `collection -> evaluation -> train -> report -> checkpoint -> resume`,
including the second-root case that a recursive-key implementation gets wrong.
Changing stochastic sites, tree order, or dropout requires a new versioned RNG
fixture and architecture review. Matching a discarded post-call cursor is not
parity evidence; composed stochastic outputs and every named child are.

## 3. Source correspondence and migration disposition

| Native owner | Production API | Official counterpart | Disposition |
| --- | --- | --- | --- |
| `config.py` | profile/mode/model enums; `SequenceShapeConfig`; network, RSSM, encoder, decoder, head, policy, optimizer, replay, run, loss, imagination, slow-value, and normalizer configs; `RuntimeOverrides`; `DebugSnapshot`; `DreamerV3Config`; `resolve_dreamer_run` | `dreamerv3/configs.yaml` | Retain typed configs, remove legacy constructor mode, correct paper values, bind each profile to one pinned authority revision, and centralize every identity-bearing runtime override. |
| `distributions.py` | `symlog`, `symexp`, `MSEOutput`, `AggregateOutput`, `NormalOutput`, `BinaryOutput`, `CategoricalOutput`, `OneHotOutput`, `TwoHotOutput` | `embodied/jax/nets.py::symlog/symexp`; output classes in `embodied/jax/outs.py` | Retain after composed parity. `_Output` and bin helpers are private implementation details. |
| `networks.py` | `Initializer`, `RMSNorm`, `Linear`, `BlockLinear`, `Conv2D`, `MLP`, `BlockGRU`, `TensorSpace`, `DictEncoder`, `DictDecoder`, `MLPHead` | `embodied/jax/nets.py`, `embodied/jax/heads.py`, `dreamerv3/rssm.py` encoder/decoder | Retain after composed parity; private output heads stay private. |
| `rssm.py` | `RSSMState`, `RSSMTrajectory`, `RSSM`, flatten/initial/reset helpers | `dreamerv3/rssm.py::RSSM` | Retain after non-singleton sequence and gradient parity. Test-only supplied-key helpers are not runtime API. |
| `replay.py` | `ReplayKey`, `ReplayBatch`, `ReplayChunk`, `ReplayWriter`, `OnlineQueue`, `UniformSelector`, `ConsecutiveStream`, `DreamerReplay` | `embodied/core/replay.py`, `chunk.py`, `selectors.py`, `streams.py`, `limiters.py` | Simplify to algorithmic replay plus exact valid-state persistence. |
| `normalization.py` | `PercentileNormalizer`, `SlowValueState` and update functions | `embodied/jax/utils.py::Normalize`, `embodied/jax/utils.py::SlowModel` | New native explicit state. |
| `agent.py` | `AgentCarry`, `AgentLoss`, `DreamerAgent`, `lambda_return`, replay-context and imagination helpers | `dreamerv3/agent.py::Agent`, `imag_loss`, `repl_loss`, `lambda_return` | New composed production agent. |
| `optimizer.py` | `DreamerOptimizerState`, AGC/LaProp transforms, schedules, `DreamerTrainState`, `validate_next_update_capacity`, `train_step` | `embodied/jax/opt.py`, `Agent._make_opt`, `Agent.train` | New explicit functional optimizer/train state. |
| `dmc.py` | `DMCSpec`, `DMCState`, `DMCEnvironment`, `DMCVectorEnvironment` | `embodied/envs/dmc.py`, `from_dm.py`, wrappers in `dreamerv3/main.py` | New production adapter; reuse wm-marl infrastructure only where contracts match. |
| `driver.py` | `SamplesPerInsertLimiter`, `PendingReplayRow`, `CadenceKind`, `CadenceRequest`, `ActiveReport`, `ActiveEvaluation`, `DriverState`, `RunnerOutput`, `DreamerRunner`, `RunnerRestoreCandidate`, `DreamerRunCoordinator`, collection/train capacity preflights | `embodied/core/driver.py::Driver`; limiter request paths in `embodied/run/parallel.py`; `train.py`, `train_eval.py` | Deterministic cooperative scheduler with bounded cadence service and cold-resume composition. |
| `artifacts.py` | `RunManifest`, `RunSummary`, `ArtifactWriter`, inspection helpers | official logger/report outputs | Direct repository-style manifests, JSONL writers, and atomic numbered outputs. |
| `checkpoint.py` | `CheckpointPayload`, `CheckpointManager` | Elements checkpointing | Versioned complete-state checkpoint; direct composition is owned by `driver.py::DreamerRunCoordinator`. |
| `oracle.py` and fixture generators | manifest, parameter translation, named numerical fixture generation | pinned files above | Test/tooling only; never eagerly imported by runtime package paths. |

`losses.py`, `models.py`, `imagination.py`, `training.py`, and `validation.py`
are approximate compatibility code with no conformant composed counterpart.
They and their exports are deleted after imports migrate. The public CLI imports
only the production modules in the table. Oracle modules are imported only by
fixture commands and parity tests.

### 3.1 Complete live-symbol migration inventory

This is the fresh AST/Serena census of every top-level class or function in
every live package module (218 symbols) plus every current CLI helper (10
symbols). Each occurs exactly once. Disposition is `R` (retain but repair), `X`
(replace at the named boundary), or `D` (delete). Counterpart codes are `CFG`
(`dreamerv3/configs.yaml`), `OUT` (`embodied/jax/outs.py`), `NET`
(`embodied/jax/nets.py` or `heads.py`), `RSSM` (`dreamerv3/rssm.py`), `REP`
(official replay/chunk/selector/stream modules), `AGT`
(`dreamerv3/agent.py`), `RUN` (official driver/run/main modules), `FIX`
(fixture-only adapter around a pinned official symbol), and `NONE`.

Caller/import codes name concrete live sites: `S` is the defining module; `E`
is `src/world_marl/dreamer_v3_baseline/__init__.py`; `C` is
`src/world_marl/scripts/train_dreamer_v3_baseline.py`; `O` is any current
oracle-named module; every other production importer is its exact basename,
such as `rssm.py` or `training.py`. `B` is
`tests/test_dreamer_v3_baseline.py` and `P` is the component test matching the
module; tests are informative but excluded from the production-import AST
equality gate. A suffix after `S:` names a module-local caller. Import callers
are mechanically derived from every package `ImportFrom` node; a table row is
not complete merely because its caller field is nonempty.

Incompatibility codes are concrete: `I1` = legacy constructor/profile/runtime
source-provenance fields conflict with the canonical typed profile plus one
pinned revision; `I2` = retained numerical component still needs composed
shape/value/gradient proof; `I3` = approximate split objective/model lacks the
official unified reductions, detach boundaries, or state; `I4` = callback,
interpreter, or source authentication/private-module emulation is
overengineered and must become a direct fixture-only translation; `I5` =
replay has provenance/security or lifetime-scan state beyond bounded
algorithmic replay; `I6` = obsolete synthetic/offline CLI helper conflicts
with the interleaved production runner and artifact/checkpoint interfaces.

| Live symbol | Disp. | Official counterpart | Live callers/imports | Incompatibility |
| --- | --- | --- | --- | --- |
| `config.py::DreamerProfile` | R | CFG | S,E,O,P,fixture_generator.py | I1 |
| `config.py::ObservationMode` | R | CFG | S,E,O,P,fixture_generator.py | I1 |
| `config.py::ModelSize` | R | CFG | S,E,P | I1 |
| `config.py::NetworkSize` | R | CFG | S,E,P | I1 |
| `config.py::RSSMConfig` | R | CFG/RSSM | S,E,networks.py,rssm.py,P,B | I1 |
| `config.py::EncoderConfig` | R | CFG/NET | S,E,networks.py,P | I1 |
| `config.py::_LegacyEncoderConfig` | D | NONE | S | I1 |
| `config.py::DecoderConfig` | R | CFG/NET | S,E,networks.py,P | I1 |
| `config.py::HeadConfig` | R | CFG/NET | S,E,networks.py,P | I1 |
| `config.py::RewardHeadConfig` | R | CFG/NET | S,E | I1 |
| `config.py::ContinueHeadConfig` | R | CFG/NET | S,E | I1 |
| `config.py::PolicyConfig` | R | CFG/NET | S,E,networks.py,P | I1 |
| `config.py::OptimizerConfig` | R | CFG/AGT | S,E,P | I1 |
| `config.py::ReplayConfig` | R | CFG/REP | S,E,replay.py,P | I1 |
| `config.py::RunConfig` | R | CFG/RUN | S,E | I1 |
| `config.py::LossScaleConfig` | R | CFG/AGT | S,E | I1 |
| `config.py::ImaginationConfig` | R | CFG/AGT | S,E | I1 |
| `config.py::SlowValueConfig` | R | CFG/AGT | S,E | I1 |
| `config.py::NormalizerConfig` | R | CFG/AGT | S,E | I1 |
| `config.py::ActorCriticConfig` | D | NONE | S,E | I1 |
| `config.py::DreamerV3Config` | R | CFG | S,E,O,C,B,P,imagination.py,training.py | I1 |
| `config.py::_json_value` | R | NONE | S:DreamerV3Config | I1 |
| `config.py::_default_model_size` | R | CFG | S:DreamerV3Config | I1 |
| `config.py::_paper_components` | X | CFG | S:DreamerV3Config | I1 |
| `config.py::_upstream_current_components` | X | CFG | S:DreamerV3Config | I1 |
| `config.py::resolve_dreamer_config` | R | CFG | S,E,O,P | I1 |
| `distributions.py::symlog` | R | NET | S,E,P,O,networks.py | I2 |
| `distributions.py::symexp` | R | NET | S,E,P,O | I2 |
| `distributions.py::_Output` | R | OUT | S | I2 |
| `distributions.py::AggregateOutput` | R | OUT | S,E,P,O,networks.py,rssm.py | I2 |
| `distributions.py::MSEOutput` | R | OUT | S,E,P,O,networks.py | I2 |
| `distributions.py::NormalOutput` | R | OUT | S,E,P,O,networks.py | I2 |
| `distributions.py::BinaryOutput` | R | OUT | S,E,P,O,networks.py | I2 |
| `distributions.py::CategoricalOutput` | R | OUT | S,E,P,O,networks.py | I2 |
| `distributions.py::OneHotOutput` | R | OUT | S,E,P,O,networks.py,rssm.py | I2 |
| `distributions.py::TwoHotOutput` | R | OUT | S,E,P,O,networks.py | I2 |
| `distributions.py::_symexp_bins` | R | OUT | S:TwoHotOutput | I2 |
| `imagination.py::DreamerImaginedRollout` | D | NONE | S,B,C | I3 |
| `imagination.py::decode_two_hot_logits` | D | AGT/OUT | S | I3 |
| `imagination.py::lambda_returns` | D | AGT | S | I3 |
| `imagination.py::create_dreamer_actor_critic_states` | D | AGT | S | I3 |
| `imagination.py::_actor_action` | D | AGT | S | I3 |
| `imagination.py::dreamer_policy_action` | D | AGT | S,C | I3 |
| `imagination.py::_posterior_start_state` | D | AGT | S | I3 |
| `imagination.py::imagine_dreamer_rollout` | D | AGT/RSSM | S | I3 |
| `imagination.py::_actor_loss` | D | AGT | S | I3 |
| `imagination.py::_critic_loss` | D | AGT | S | I3 |
| `imagination.py::train_dreamer_actor_critic` | D | AGT | S,B,C | I3 |
| `imagination.py::open_loop_diagnostic` | D | AGT | S | I3 |
| `losses.py::symlog` | D | OUT | S,imagination.py,training.py,B | I3 |
| `losses.py::symexp` | D | OUT | S,imagination.py,B | I3 |
| `losses.py::two_hot` | D | OUT | S,E,imagination.py,training.py,B | I3 |
| `losses.py::categorical_kl_loss` | D | RSSM | S,E,B | I3 |
| `losses.py::balanced_categorical_kl_loss` | D | RSSM | S,E,training.py | I3 |
| `models.py::DreamerEncoder` | D | NET | S,E,training.py,B | I3 |
| `models.py::DreamerDecoder` | D | NET | S,E,training.py,B | I3 |
| `models.py::RewardHead` | D | NET | S,E,training.py,B | I3 |
| `models.py::ContinueHead` | D | NET | S,E,training.py,B | I3 |
| `models.py::DreamerActor` | D | NET/AGT | S,E,imagination.py,B | I3 |
| `models.py::DreamerCritic` | D | NET/AGT | S,E,imagination.py,B | I3 |
| `network_oracle.py::_ModuleMeta` | D | NONE | S | I4 |
| `network_oracle.py::_Module` | D | NONE | S | I4 |
| `network_oracle.py::_ParameterContext` | D | NONE | S,O | I4 |
| `network_oracle.py::_seed` | X | FIX | S | I4 |
| `network_oracle.py::_activate` | X | FIX | S,O | I4 |
| `network_oracle.py::_Einops` | D | NONE | S | I4 |
| `network_oracle.py::_Space` | X | FIX | S,O | I4 |
| `network_oracle.py::_join_path` | X | FIX | S | I4 |
| `network_oracle.py::_exec_source` | D | NONE | S,O | I4 |
| `network_oracle.py::_load_official_modules` | X | FIX | S,O | I4 |
| `network_oracle.py::_bind` | X | FIX | S,O | I4 |
| `network_oracle.py::_collect_case` | X | FIX | S | I4 |
| `network_oracle.py::_host` | X | FIX | S | I4 |
| `network_oracle.py::_input_grid` | X | FIX | S | I4 |
| `network_oracle.py::_official_network_arrays` | X | FIX/NET | S,P | I4 |
| `network_oracle.py::_networks_worker` | D | NONE | S | I4 |
| `network_oracle.py::run_networks_case` | X | FIX/NET | S,P,O | I4 |
| `network_oracle.py::_main` | D | NONE | S | I4 |
| `networks.py::_require_compute_dtype` | R | NET | S | I2 |
| `networks.py::_uniform_classes` | R | OUT/NET | S | I2 |
| `networks.py::_activation` | R | NET | S,rssm.py | I2 |
| `networks.py::Initializer` | R | NET | S,E,P | I2 |
| `networks.py::RMSNorm` | R | NET | S,E,P,rssm.py | I2 |
| `networks.py::Linear` | R | NET | S,E,P,rssm.py | I2 |
| `networks.py::BlockLinear` | R | NET | S,E,P | I2 |
| `networks.py::Conv2D` | R | NET | S,E,P | I2 |
| `networks.py::MLP` | R | NET | S,E,P | I2 |
| `networks.py::BlockGRU` | R | RSSM/NET | S,E,P,rssm.py | I2 |
| `networks.py::TensorSpace` | R | NET | S,E,rssm.py,replay.py,P | I2 |
| `networks.py::DictEncoder` | R | RSSM/NET | S,E,P | I2 |
| `networks.py::_OutputHead` | R | NET | S | I2 |
| `networks.py::_DictOutputHead` | R | NET | S | I2 |
| `networks.py::DictDecoder` | R | RSSM/NET | S,E,P | I2 |
| `networks.py::MLPHead` | R | NET | S,E,P | I2 |
| `oracle.py::_callback_implementation_fingerprint` | D | NONE | S | I4 |
| `oracle.py::_referenced_global_names` | D | NONE | S | I4 |
| `oracle.py::_referenced_module_attribute_chains` | D | NONE | S | I4 |
| `oracle.py::_fingerprint_module_dependency` | D | NONE | S | I4 |
| `oracle.py::_fingerprint_module_attribute_binding` | D | NONE | S | I4 |
| `oracle.py::_fingerprint_component` | D | NONE | S | I4 |
| `oracle.py::OracleInvocation` | D | NONE | S,E,O,P | I4 |
| `oracle.py::OracleSourceSpec` | D | NONE | S,E,P,rssm.py | I4 |
| `oracle.py::_rehydrate_oracle_source_spec` | D | NONE | S | I4 |
| `oracle.py::_oracle_source_spec_signature` | D | NONE | S | I4 |
| `oracle.py::_callback_logical_identity` | D | NONE | S | I4 |
| `oracle.py::_validate_bound_callback` | D | NONE | S | I4 |
| `oracle.py::_validate_source_spec_callbacks` | D | NONE | S | I4 |
| `oracle.py::register_oracle_source_spec` | D | NONE | S,rssm.py | I4 |
| `oracle.py::_resolve_source_spec` | D | NONE | S | I4 |
| `oracle.py::oracle_source_spec` | D | NONE | S,O,P | I4 |
| `oracle.py::official_revision` | X | CFG | S,O,P | I1 |
| `oracle.py::profile_overrides` | X | CFG | S,O,P | I1 |
| `oracle.py::TensorSpec` | R | FIX | S,E,P,fixture_generator.py | I4 |
| `oracle.py::OracleManifest` | R | FIX | S,E,P,fixture_generator.py | I4 |
| `oracle.py::ParameterMapping` | R | FIX | S,E,P | I4 |
| `oracle.py::ParameterTranslator` | R | FIX | S,E,P | I4 |
| `oracle.py::OracleHarness` | D | NONE | S,E,P,O | I4 |
| `oracle.py::_validate_config_case_arrays` | X | FIX/CFG | S | I4 |
| `oracle.py::_config_worker` | D | NONE | S | I4 |
| `oracle.py::_distributions_worker` | D | NONE | S | I4 |
| `oracle.py::_load_official_outs` | X | FIX/OUT | S | I4 |
| `oracle.py::_OfficialRandomFacade` | X | FIX | S | I4 |
| `oracle.py::_SuppliedCategoricalNoise` | X | FIX | S | I4 |
| `oracle.py::_supplied_categorical_noise_scope` | X | FIX | S | I4 |
| `oracle.py::_load_official_function` | X | FIX | S | I4 |
| `oracle.py::_load_official_method` | X | FIX | S | I4 |
| `oracle.py::_OfficialHeadStub` | X | FIX/NET | S | I4 |
| `oracle.py::_official_distribution_arrays` | X | FIX/OUT | S,P | I4 |
| `oracle.py::_parameter_path` | R | FIX | S | I4 |
| `oracle.py::_transform_parameter` | R | FIX | S | I4 |
| `oracle.py::_canonical_generator_request` | X | FIX | S | I4 |
| `oracle.py::_canonical_json` | R | FIX | S,O | I4 |
| `oracle.py::_sha256_bytes` | R | FIX | S,O,fixture_generator.py | I4 |
| `oracle.py::_sha256_path` | R | FIX | S,fixture_generator.py | I4 |
| `oracle.py::_git_show` | R | FIX | S,O,fixture_generator.py | I4 |
| `oracle.py::_git_object_exists` | R | FIX | S | I4 |
| `oracle.py::_source_hashes_for` | R | FIX | S,P,fixture_generator.py | I4 |
| `oracle.py::_source_allows_dtype` | R | FIX | S,P | I4 |
| `oracle.py::_FixtureSourceName` | R | FIX | S | I4 |
| `oracle.py::_write_deterministic_npz` | R | FIX | S,fixture_generator.py | I4 |
| `oracle.py::_main` | D | NONE | S | I4 |
| `replay.py::_ReplayRestored` | X | REP | S | I5 |
| `replay.py::_require_exact_keys` | X | REP | S | I5 |
| `replay.py::_coerce_value` | X | REP | S | I5 |
| `replay.py::_validate_latent_value` | R | REP | S | I5 |
| `replay.py::_space_signature` | R | REP | S | I5 |
| `replay.py::ReplayKey` | R | REP | S,E,P | I5 |
| `replay.py::ReplayBatch` | R | REP | S,E,P | I5 |
| `replay.py::ReplayChunk` | R | REP | S,E,P | I5 |
| `replay.py::OnlineQueue` | R | REP | S,E,P | I5 |
| `replay.py::UniformSelector` | R | REP | S,E,P | I5 |
| `replay.py::ConsecutiveStream` | R | REP | S,E,P | I5 |
| `replay.py::ReplayWriter` | R | REP | S,E,P | I5 |
| `replay.py::DreamerReplay` | R | REP | S,E,P | I5 |
| `replay_oracle.py::_native_module_violations` | D | NONE | S | I4 |
| `replay_oracle.py::_require_isolated_worker_modules` | D | NONE | S | I4 |
| `replay_oracle.py::_installed_elements_provenance` | D | NONE | S | I4 |
| `replay_oracle.py::_Section` | D | NONE | S | I4 |
| `replay_oracle.py::_Timer` | X | FIX | S | I4 |
| `replay_oracle.py::_UUID` | X | FIX | S | I4 |
| `replay_oracle.py::_RWLock` | D | NONE | S | I4 |
| `replay_oracle.py::_Limiters` | X | FIX/REP | S | I4 |
| `replay_oracle.py::_timestamp` | X | FIX | S | I4 |
| `replay_oracle.py::_live_shim_hashes` | D | NONE | S | I4 |
| `replay_oracle.py::_runtime_contract` | D | NONE | S | I4 |
| `replay_oracle.py::_live_generator_file_hashes` | D | NONE | S | I4 |
| `replay_oracle.py::_normalized_contract_source` | D | NONE | S | I4 |
| `replay_oracle.py::_validate_live_generator_contract` | D | NONE | S | I4 |
| `replay_oracle.py::_case_contract` | X | FIX/REP | S | I4 |
| `replay_oracle.py::_row_schema_contract` | X | FIX/REP | S | I4 |
| `replay_oracle.py::_same_contract` | D | NONE | S | I4 |
| `replay_oracle.py::_request_path` | D | NONE | S | I4 |
| `replay_oracle.py::_execution_path` | D | NONE | S | I4 |
| `replay_oracle.py::_validate_replay_generator_provenance` | D | NONE | S | I4 |
| `replay_oracle.py::_stable_replay_request` | X | FIX/REP | S | I4 |
| `replay_oracle.py::_build_replay_invocation` | D | NONE | S | I4 |
| `replay_oracle.py::_resolve_replay_generator_invocation` | D | NONE | S | I4 |
| `replay_oracle.py::_extract_classes` | X | FIX | S | I4 |
| `replay_oracle.py::_load_source_classes` | X | FIX/REP | S | I4 |
| `replay_oracle.py::_source_attestation` | D | NONE | S | I4 |
| `replay_oracle.py::_row` | X | FIX/REP | S | I4 |
| `replay_oracle.py::_key_arrays` | X | FIX/REP | S | I4 |
| `replay_oracle.py::_rng_bytes` | X | FIX/REP | S | I4 |
| `replay_oracle.py::_official_arrays` | X | FIX/REP | S,P | I4 |
| `replay_oracle.py::_worker` | D | NONE | S | I4 |
| `replay_oracle.py::run_replay_case` | X | FIX/REP | S,P | I4 |
| `replay_oracle.py::_main` | D | NONE | S | I4 |
| `rssm.py::RSSMState` | R | RSSM | S,E,P,B,imagination.py,training.py | I2 |
| `rssm.py::RSSMTrajectory` | R | RSSM | S,E,P | I2 |
| `rssm.py::ninjax_scan_sample_keys` | X | FIX/RSSM | S,P | I2 |
| `rssm.py::RSSM` | R | RSSM | S,E,P | I2 |
| `rssm.py::flatten_rssm_state` | R | RSSM | S,C,B,imagination.py,training.py | I2 |
| `rssm.py::initial_rssm_state` | X | RSSM | S,C,B,training.py | I2 |
| `rssm.py::reset_rssm_state` | X | RSSM | S,C,B,training.py | I2 |
| `rssm_oracle.py::_host` | X | FIX | S | I4 |
| `rssm_oracle.py::_grid` | X | FIX | S | I4 |
| `rssm_oracle.py::_scan_keys` | X | FIX/RSSM | S | I4 |
| `rssm_oracle.py::_scan_key_scope` | X | FIX/RSSM | S | I4 |
| `rssm_oracle.py::_source_scan` | X | FIX/RSSM | S | I4 |
| `rssm_oracle.py::_OrderedCategoricalNoise` | X | FIX/RSSM | S | I4 |
| `rssm_oracle.py::_RSSMSpace` | X | FIX/RSSM | S | I4 |
| `rssm_oracle.py::_ordered_noise_scope` | X | FIX/RSSM | S | I4 |
| `rssm_oracle.py::_categorical_forbidden` | D | NONE | S | I4 |
| `rssm_oracle.py::_gumbel_noise` | X | FIX/RSSM | S | I4 |
| `rssm_oracle.py::_load_feat2tensor` | X | FIX/RSSM | S | I4 |
| `rssm_oracle.py::_source_model_dimensions` | X | FIX/RSSM | S | I4 |
| `rssm_oracle.py::_InitializationDistribution` | X | FIX/RSSM | S | I4 |
| `rssm_oracle.py::_initialize_parameters` | X | FIX/RSSM | S | I4 |
| `rssm_oracle.py::_loss_from_exact_source` | X | FIX/RSSM | S | I4 |
| `rssm_oracle.py::_official_arrays_exact` | X | FIX/RSSM | S,P | I4 |
| `rssm_oracle.py::_worker` | D | NONE | S | I4 |
| `rssm_oracle.py::run_rssm_case` | X | FIX/RSSM | S,P | I4 |
| `rssm_oracle.py::_main` | D | NONE | S | I4 |
| `training.py::DreamerWorldModel` | D | AGT/RSSM | S,imagination.py,C | I3 |
| `training.py::dreamer_action_features` | D | AGT/RSSM | S,C | I3 |
| `training.py::create_dreamer_train_state` | D | AGT | S,B | I3 |
| `training.py::dreamer_world_model_loss` | D | AGT | S | I3 |
| `training.py::dreamer_train_step` | D | AGT | S,B | I3 |
| `training.py::train_dreamer_world_model` | D | AGT | S,C | I3 |
| `validation.py::finite_metric_check` | D | NONE | S,C | I6 |
| `validation.py::loss_decreased` | D | NONE | S,C | I6 |
| `scripts/train_dreamer_v3_baseline.py::parse_args` | X | RUN | S:main,pyproject | I6 |
| `scripts/train_dreamer_v3_baseline.py::_config_payload` | D | NONE | S:main | I6 |
| `scripts/train_dreamer_v3_baseline.py::_collect_steps` | D | NONE | S:_config_payload,_make_batch | I6 |
| `scripts/train_dreamer_v3_baseline.py::_write_png` | X | RUN | S:main | I6 |
| `scripts/train_dreamer_v3_baseline.py::_to_rgb_panel` | X | AGT | S:main | I6 |
| `scripts/train_dreamer_v3_baseline.py::_squeeze_single_agent_axis` | D | NONE | S:_evaluate_real_env | I6 |
| `scripts/train_dreamer_v3_baseline.py::_evaluate_real_env` | X | RUN | S:main | I6 |
| `scripts/train_dreamer_v3_baseline.py::_make_batch` | D | NONE | S:main | I6 |
| `scripts/train_dreamer_v3_baseline.py::_run_accounting` | X | RUN | S:main | I6 |
| `scripts/train_dreamer_v3_baseline.py::main` | X | RUN | pyproject entrypoint | I6 |

The I4 oracle migration is intentionally sequential. Task 1b removes dead
generic invocation/harness/profile/config-source APIs and migrates only stale
manifest/request/generator assertions and imports in the distribution,
network, and RSSM component tests to the compact fixture schema; their
equations, tolerances, and numerical assertions do not change. Task 1b makes
`OracleManifest` and `fixture_generator.py` depend only on immutable private
source-hash/dtype tables and direct lookup helpers; the thin fixture source-name
constants likewise do not instantiate or query the runtime registry. Because
`rssm.py` and the package root still import source-spec registration symbols
until the runtime/tooling seam is cut, Task 1b may retain only that exact
temporary `OracleSourceSpec`/registry surface. Its complete production
reference set is `oracle.py`, `rssm.py`, and `__init__.py`; neither the manifest
class nor fixture-generator tooling may reference it. Task 1c first removes
those runtime import edges and then deletes the temporary class, registration
functions, and global registry from `oracle.py` in the same unit. The immutable
fixture source tables and direct helpers remain. The three Task-1b manifest
tests that freeze the temporary deletion seam are sequentially Task-1c-owned
only for their post-deletion rewrite: they require exact registry absence while
still loading canonical fixtures through those retained direct tables. At the
Task 1c boundary the component suites must collect without importing tooling
through production modules; their complete
numerical execution is deliberately deferred to Tasks 3a-3c. The live network
and RSSM tests still construct the removed legacy config shape, so treating
their current execution failures as a Task 1c import-boundary failure would be
a false gate. Tasks 3b and 3c migrate those constructors while validating the
corresponding equations. Replay's removed `run_replay_case` caller is not hidden
by a compatibility worker; its migration remains Task 2 ownership.

The package-root edge is split by symbol as well: Task 1b removes only the dead
`OracleHarness` and `OracleInvocation` imports and exports together with their
definitions so package import stays valid; Task 1c removes the remaining eager
oracle/tooling surface when it cuts the runtime seam.

#### Legacy import-site inventory (complete)

These are all current import statements that keep `losses.py`, `models.py`,
`imagination.py`, `training.py`, or `validation.py` reachable. Line numbers are
current and one-based. Task 9 migrates every row before deleting those modules;
the acceptance test also reparses imports so a later site cannot hide here.

| Import site | Imported legacy names | Replacement owner |
| --- | --- | --- |
| `tests/test_dreamer_v3_baseline.py:14` | actor-critic trainer from `imagination` | Task 10 production Agent/runner gate |
| `tests/test_dreamer_v3_baseline.py:17` | KL, symlog/symexp, and two-hot from `losses` | Task 10 retained distribution/RSSM gates |
| `tests/test_dreamer_v3_baseline.py:23` | six classes from `models` | Task 10 `networks`/Agent gates |
| `tests/test_dreamer_v3_baseline.py:37` | train-state factory and train step from `training` | Task 10 `optimizer.py` gate |
| `src/world_marl/dreamer_v3_baseline/imagination.py:12` | `symexp`, `symlog`, `two_hot` from `losses` | importing module deleted after Tasks 3-4c |
| `src/world_marl/dreamer_v3_baseline/imagination.py:13` | actor and critic from `models` | importing module deleted after Task 4c |
| `src/world_marl/dreamer_v3_baseline/imagination.py:15` | world model from `training` | importing module deleted after Task 4d |
| `src/world_marl/dreamer_v3_baseline/training.py:12` | balanced KL, symlog, two-hot from `losses` | importing module deleted after Task 5 |
| `src/world_marl/dreamer_v3_baseline/training.py:17` | encoder/decoder/reward/continue heads from `models` | importing module deleted after Task 5 |
| `src/world_marl/dreamer_v3_baseline/__init__.py:35` | KL and two-hot exports from `losses` | Task 9 retained distributions/RSSM exports |
| `src/world_marl/dreamer_v3_baseline/__init__.py:40` | six approximate model exports from `models` | Task 9 networks/Agent exports |
| `src/world_marl/scripts/train_dreamer_v3_baseline.py:13` | policy action and actor-critic trainer from `imagination` | Task 9 Agent/runner |
| `src/world_marl/scripts/train_dreamer_v3_baseline.py:22` | world model, action features, and trainer from `training` | Task 9 Agent/optimizer/runner |
| `src/world_marl/scripts/train_dreamer_v3_baseline.py:27` | finite/loss-decrease helpers from `validation` | Task 9 measured status handling |

### 3.2 Shared lifecycle rules

All array signatures preserve arbitrary leading batch axes unless literal
`[B,T]` or `[N]` axes are shown. Distribution objects and stateless network
modules own no mutable runtime state: Flax parameters are supplied to `apply`,
reset is a no-op, and their parameter subtree is serialized through
`DreamerTrainState`. Runtime dataclasses and `FrozenDict` may be useful typed
objects, but they are never passed to the checkpoint codec. Only explicitly
listed state is mutable. Every checkpoint owner exposes a complete
`state_dict()` (or the exact class-specific projection below) and a closed
inverse validator; episode reset changes only caller-owned carry or environment
state. Resource handles are never checkpoint state.

An owner record is already canonical before Task 8b sees it. Its complete value
language is plain `dict` with the schema-declared string keys (plus replay's one
declared bytes-key `refs` map), plain `list`, `str`, `bytes`, Python finite
numbers and booleans, `None`, and numeric NumPy/JAX arrays. Tuples, Python or
Flax struct dataclass instances, `FrozenDict`, enums, paths, file handles, and
arbitrary objects are rejected at the owner boundary. `state_dict()` allocates
fresh lists/dicts and arrays; inverse validators reject missing/extra keys,
shape/dtype/range errors, and aliases before reconstructing a fresh runtime
object. Owners do not share a generic object converter, tag registry,
constructor hook, or recursive fallback. Task 8b validates this already-closed
primitive tree and adapts only the four declared wide integers.

The exact owner projections are:

- Task 1: every config/debug/runtime record has `state_dict()` and closed
  `from_state()`, while `ResolvedDreamerRun.identity_state()` returns the
  canonical config/hash/authority/debug/runtime-override mapping used by
  manifests and checkpoints;
- Tasks 2/2b: `AgentCarry`, `ReplayBatch`, `ReplayChunk`, `ReplayWriter`,
  `OnlineQueue`, `UniformSelector`, `ConsecutiveStream`, and `DreamerReplay`
  produce primitive records and closed inverses; a live stream stores its
  current batch through `ReplayBatch.state_dict()`, and
  `DreamerReplay.from_state_dict(...)` constructs a fresh replay rather than
  mutating an existing receiver. The bound agent/spaces and stream/active-report
  owners supply every inverse's exact external shape invariant;
- Tasks 3-5: `RSSMState`, normalizer states, slow-value state,
  `DreamerOptimizerState`, and `DreamerTrainState` have exact primitive
  projections. Task 5 owns the pure `DreamerTrainStateSchema`, one-time
  initializer, `DreamerTrainState.state_dict()`, and its closed inverse;
  parameter and optimizer mappings are
  recursively unfrozen into fresh plain mappings and reconstructed with exact
  key/shape/dtype validation;
- Task 6: each child is a `DMCState`; vector capture is a fresh
  `list[DMCState]`, while expected runtime specs may remain tuples because they
  are constructor arguments, not checkpoint leaves;
- Task 7: limiter and complete `DriverState` records project pending rows,
  cadence tagged unions, aggregation windows, carries, the two call counters,
  summaries, and every FIFO into primitive mappings/lists/arrays. Its named
  closed inverse receives agent, run config, sequence shape, and spaces;
- Task 8a: `ArtifactWriter.state_dict()` is a closed primitive mapping of
  immutable run identity, durable offsets, and numbered-file cursors.

Cold resume validates its checkpoint and artifact prefix before opening append
handles, then directly truncates the two active JSONL files and removes numbered
outputs at or above the checkpoint cursors. There is no recovery archive,
checkpoint-generation writer protocol or retry state machine; section 10 defines
the small idempotent reconciliation operation.

### 3.3 Configuration classes

| Native class/function | Constructor/result | State, reset, serialization | Exact official correspondence |
| --- | --- | --- | --- |
| `DreamerProfile` | `paper` or explicit `upstream-current` string enum | immutable; no reset; serialize by value | profile snapshots from `dreamerv3/configs.yaml` plus Nature overrides |
| `ObservationMode` | `vision` or `proprio` string enum | immutable; no reset; serialize by value | DMC modality selections in published protocol |
| `ModelSize` | named typed size preset | immutable; no reset; serialize by value | size names in `dreamerv3/configs.yaml` |
| `NetworkSize` | resolved width/count scalars | immutable; complete dataclass in config | `dreamerv3/configs.yaml` size presets |
| `RSSMConfig` | deter/stoch/classes/unimix/norm/activation/dtype | immutable; complete dataclass | `dreamerv3/configs.yaml::rssm`; `dreamerv3/rssm.py::RSSM` |
| `EncoderConfig` | key regexes, CNN/MLP depths, striding, norm/activation | immutable; complete dataclass | `dreamerv3/configs.yaml::encoder`; `dreamerv3/rssm.py::Encoder` |
| `DecoderConfig` | key regexes, CNN/MLP depths, striding, norm/activation | immutable; complete dataclass | `dreamerv3/configs.yaml::decoder`; `dreamerv3/rssm.py::Decoder` |
| `HeadConfig` | distribution family, layers/units/bins/unimix/outscale | immutable; complete dataclass | head dictionaries in `dreamerv3/configs.yaml`; `embodied/jax/heads.py::MLPHead` |
| `RewardHeadConfig` | reward `HeadConfig` specialization | immutable; complete dataclass | `dreamerv3/configs.yaml::reward` |
| `ContinueHeadConfig` | continuation `HeadConfig` specialization | immutable; complete dataclass | `dreamerv3/configs.yaml::cont` |
| `PolicyConfig` | action distributions/std/unimix/entropy/layers | immutable; complete dataclass | `dreamerv3/configs.yaml::policy`; `embodied/jax/heads.py::DictHead` |
| `OptimizerConfig` | lr, AGC, beta1/2, epsilon, decay, schedule/warmup/anneal | immutable; complete dataclass | `dreamerv3/configs.yaml::opt`; `dreamerv3/agent.py::Agent._make_opt` |
| `SequenceShapeConfig` | sole owner of train `batch_size=B`, `sequence_length=T`, `context=K`, `consecutive=C`, report `report_length=T_report`, and `report_consecutive=C_report`; derived `raw_length=K+T*C` and `report_raw_length=K+T_report*C_report` | frozen; positive `B,T,C,T_report,C_report`, nonnegative `K`; complete config/checkpoint identity | `dreamerv3/configs.yaml::{batch_size,batch_length,report_length,consec_train,consec_report,replay_context}`; `dreamerv3/main.py::make_replay/make_stream`; `embodied/core/streams.py::Consec` |
| `ReplayConfig` | capacity, chunk size, and online queue size | immutable; complete dataclass; contains no replay construction seed and no train/report sequence-shape field | `dreamerv3/configs.yaml::replay`; `embodied/core/replay.py::Replay` |
| `RunConfig` | train ratio, positive physical-frame `log_every`, other physical-frame cadences/budgets, train/evaluation vector counts including `eval_envs`, `report_batches`, evaluation, and checkpoint only | immutable; complete dataclass; contains no seed and no train/report sequence-shape field; limiter `minsize` is derived | source `run` controls including `eval_envs=4` and `report_batches=1`, plus the explicit native physical-log decision below; `embodied/run/parallel.py`, `train.py`, `train_eval.py` |
| `LossScaleConfig` | named rec/rew/con/dyn/rep/policy/value/repval scales | immutable; complete dataclass | `dreamerv3/configs.yaml::loss_scales`; `dreamerv3/agent.py::Agent.loss` |
| `ImaginationConfig` | horizon/lambda/contdisc/slowtar/ac_grads/repval switches | immutable; complete dataclass | `dreamerv3/configs.yaml`; `dreamerv3/agent.py::Agent.imag_loss/repl_loss` |
| `SlowValueConfig` | EMA rate and update period | immutable; complete dataclass | `dreamerv3/configs.yaml::slowvalue`; `embodied/jax/utils.py::SlowModel` |
| `NormalizerConfig` | kind/rate/percentiles/limit/debias | immutable; complete dataclass; `valnorm` and `advnorm` inherit `Normalize.debias=True`, while `retnorm` explicitly sets `debias=False` | normalizer entries in `dreamerv3/configs.yaml`; `embodied/jax/utils.py::Normalize` |
| `DreamerV3Config` | complete aggregate plus profile/mode/task/model and exactly one checked scalar `seed` field | immutable; canonical JSON, SHA-256, and full config serialize; `seed` is the sole canonical public-seed owner | resolved top-level `dreamerv3/configs.yaml::seed` snapshot |
| `RuntimeOverrides` | exact optional identity-bearing fields `env_steps`, `num_envs`, `batch_size`, `batch_length`, `train_ratio`, `eval_every`, `eval_episodes`, `report_every`, `checkpoint_every`, and environment-only `camera` | frozen; rejects unknown fields and preserves separate algorithm/environment explicit maps | typed native boundary for the Task-9 CLI |
| `DebugSnapshot` | exact noncanonical resource snapshot named `debug-local-v1` | frozen full replacement values; serialized in resolved identity | native local-execution profile; equations and DMC semantics unchanged |
| `resolve_dreamer_run(*, mode, task, profile=DreamerProfile.PAPER, seed=0, model=None, debug_local=False, overrides=RuntimeOverrides())` | validated `ResolvedDreamerRun(config, explicit_overrides, debug_snapshot)` | pure; `mode` and `task` are required keyword-only inputs, while omitted `profile` selects the paper snapshot; validates the primary seed, applies the merge order in section 4, revalidates, then emits canonical JSON/hash | official config merge plus one typed native runtime boundary |

Every Task-1 config row above projects through an exact plain-mapping
`state_dict()` and a closed `from_state()` that reconstructs the named enum or
frozen runtime record. DreamerV3Config.seed is the sole canonical public-seed owner.
Every accepted public constructor enforces the same exact Python primitive,
tuple-element, finiteness, and nested-record rules as its inverse, so
`type(record).from_state(record.state_dict()) == record` for every constructible
record. Bool-as-int, int-as-float, NumPy scalar, list-for-tuple, and wrong
nested-record values are rejected at construction rather than emitted into a
noncanonical owner projection.
Seed is a primary resolver input, not a `RuntimeOverrides` field.
`ResolvedDreamerRun.identity_state()` is the only checkpoint/manifest
projection and contains canonical config, `config_sha256`,
authority revision, nullable debug snapshot name, and the exact algorithm and
environment runtime-override mappings. These projections contain no dataclass,
enum, tuple, `FrozenDict`, path, or arbitrary object leaf.

Camera selection is environment identity, not model/algorithm configuration:
camera is not a `DreamerV3Config` field and is excluded from its canonical JSON
and config hash. It is an immutable `DMCSpec` value and appears in the runtime
override map, canonical argv, manifest environment identity, and checkpoint
compatibility tuple.

`resolve_dreamer_config(*, mode, task, profile=DreamerProfile.PAPER, seed=0, model=None, debug_local=False)`
remains only a no-override convenience wrapper and calls
`resolve_dreamer_run(mode=mode, task=task, profile=profile, seed=seed, model=model, debug_local=debug_local, overrides=RuntimeOverrides())`
before returning `.config`; the public CLI calls the latter with keywords.
`ActorCriticConfig` and `_LegacyEncoderConfig` are migration-only and are
removed; they are not production classes. The `E` caller annotation on
`config.py::ActorCriticConfig` in the live-symbol inventory is an exact
package-root ownership edge: Task 1a removes the `ActorCriticConfig` import and
`__all__` entry and adds imports and `__all__` entries for exactly
`DebugSnapshot`, `ResolvedDreamerRun`, `RuntimeOverrides`,
`SequenceShapeConfig`, and `resolve_dreamer_run` in
`src/world_marl/dreamer_v3_baseline/__init__.py` in the same atomic change that
deletes the legacy class. No compatibility alias, forwarding import, or
deprecated export survives. Task 1c later removes only eager
oracle/tooling imports and exports from that file; Task 9 later replaces the
remaining production exports.

### 3.4 Distribution classes

Shared calls are `pred() -> [...,*E]`, `sample(key) -> [...,*E]`,
`logp(event) -> [...]`, `prob(event) -> [...]`, `entropy() -> [...]`,
`kl(other) -> [...]`, and `loss(target) -> [...]`. Base outputs implement
`prob(event) = exp(logp(event))`. `AggregateOutput.prob()` is the pinned
`Agg.prob` exception: it sums the base output's per-event probabilities over
exactly the configured trailing event axes; it does not exponentiate the
aggregated log probability. Unavailable operations raise instead of changing
semantics. Distribution objects reconstruct from head tensors and have no reset
or independent checkpoint entry.

| Native class | Constructor and output contract | Gradient/update behavior | Exact official symbol |
| --- | --- | --- | --- |
| `AggregateOutput` | `(base, event_ndims)`; `loss`, `logp`, `prob`, entropy, and KL reduce exactly the trailing event axes using the pinned operation (`prob` sums base probabilities) | differentiable reduction | `embodied/jax/outs.py::Agg` |
| `MSEOutput` | `(mean [...,*E], squash=None)`; `pred()` is the encoded mean and loss is squared error against the stopped optionally squashed target | gradient to mean; `squash(float32(target))` and target are stopped | `embodied/jax/outs.py::MSE` |
| `NormalOutput` | `(mean [...,*E], stddev [...,*E] | scalar)` is a plain normal; native bounded head uses `mean=tanh(raw_mean)` and `stddev=(maxstd-minstd)*sigmoid(raw_std+2)+minstd` | reparameterized sample where consumed; ordinary logp gradients | `embodied/jax/outs.py::Normal` plus `embodied/jax/heads.py::Head.normal` |
| `BinaryOutput` | `(logits [...,*E])`; Bernoulli prediction/loss | stable cross entropy | `embodied/jax/outs.py::Binary` |
| `CategoricalOutput` | `(logits [...,C], unimix=0.0)`; mixing is opt-in and defaults off | gradients through probabilities, mixed only when explicitly configured | `embodied/jax/outs.py::Categorical`; ordinary `Head.categorical` passes no unimix |
| `OneHotOutput` | `(logits [...,C], unimix)`; hard one-hot sample | explicit key; straight-through probability gradients | `embodied/jax/outs.py::OneHot` |
| `TwoHotOutput` | `(logits [...,Q], bins [Q])`, canonical `Q=255`; symlog interpolation/symexp expectation | stopped target interpolation | `embodied/jax/outs.py::TwoHot` |

`symlog` and `symexp` are pure functions mapped to
`embodied/jax/nets.py::symlog/symexp`. `_Output` and bin helpers remain private.

### 3.5 Network and RSSM classes

Every module row stores only its Flax parameter subtree, updates only by
backpropagation through its call, has no episode reset unless shown, and
serializes only inside train parameters.

| Native class | Constructor; principal input -> output | Reset/state exception | Exact official symbol |
| --- | --- | --- | --- |
| `Initializer` | `(name, scale)` resolves the source distribution/fan pair; `(key, shape, dtype) -> parameter` | deterministic from the supplied key; no parameters | `embodied/jax/nets.py::Initializer(dist, fan, scale)` |
| `RMSNorm` | `(epsilon, scale, param_dtype, compute_dtype)`; `[...,F] -> [...,F]` | optional learned final-axis scale | `embodied/jax/nets.py::Norm(impl='rms')` |
| `Linear` | `(units, activation, normalization, output_scale, bias, dtypes)`; `[...,I] -> [...,*units]` | kernel, optional bias/norm subtree | source `embodied/jax/nets.py::Linear(units)` followed at its call sites by `Norm`/activation; native fusion must retain that order |
| `BlockLinear` | `(units, blocks, bias, initializer, output_scale, normalization, activation, dtypes)`; accepts `[...,I]` with `I % blocks == 0`, internally reshapes to `[...,blocks,I/blocks]`, and returns `[...,units]` | block kernel `[blocks,I/blocks,units/blocks]` and optional bias/norm; no externally visible block axis | source `embodied/jax/nets.py::BlockLinear(units, blocks)` followed at its call sites by `Norm`/activation |
| `Conv2D` | `(depth, kernel, stride, transp, groups, pad, bias, normalization, activation, dtypes)`; flattened-batch `[M,H,W,C] -> [M,H',W',depth]` | convolution kernel and optional bias/norm; transpose mode repeats/masks before convolution as in source | source `embodied/jax/nets.py::Conv2D(depth, kernel, stride)` followed at its call sites by `Norm`/activation |
| `MLP` | `(layers, units, activation, normalization, bias, dtypes)`; `[...,I] -> [...,units]` | ordered `Linear`/`Norm` subtrees | `embodied/jax/nets.py::MLP(layers, units)` plus source fields |
| `BlockGRU` | `(RSSMConfig, action_dim, dtypes)`; `(deter [...,D], stoch [...,S,C], action [...,A], is_first [...] | None) -> deter [...,D]` | native `is_first` adapter zeroes the three inputs before the core; source `_observe` performs that reset before calling `_core(deter, stoch, flattened_action)` | `dreamerv3/rssm.py::RSSM._core` (there is no `_gru`) |
| `TensorSpace` | `(shape, dtype, low, high, discrete)`; validates shape/dtype/finiteness, but never enforces action `low`/`high` | immutable metadata; bounds are declaration/serialization metadata only | official `Agent` `obs_space`/`act_space` leaf contract |
| `DictEncoder` | `(spaces, EncoderConfig, dtypes)`; native `apply(params, observations[key] [...,*E]) -> token [...,F]` | stateless native adapter; `DreamerAgent` supplies the official empty carry/reset/training wrapper and empty replay entries | `dreamerv3/rssm.py::Encoder.__call__(carry, obs, reset, training, single)` |
| `DictDecoder` | `(spaces, DecoderConfig, dtypes)`; native `apply(params, features)` consumes a mapping with at least `deter [...,D]` and `stoch [...,S,C]` and returns `outputs[key]` | stateless native adapter; `DreamerAgent` supplies the official empty carry/reset/training wrapper and empty entries | `dreamerv3/rssm.py::Decoder.__call__(carry, feat, reset, training, single)`, whose `feat` is the mapping, not a preflattened tensor |
| `MLPHead` | `(TensorSpace, HeadConfig | PolicyConfig, dtypes)`; `features [...,F] -> Output` | no recurrent state; one native head per output/action key where a tree is required | `embodied/jax/heads.py::MLPHead.__call__(x, bdims)` and `DictHead` |
| `RSSMState` | `deter [...,D]`, `stoch [...,S,C]` | caller pytree zeroed on first; `state_dict()` copies the exact two arrays and `from_state(state, config, expected_leading_shape)` validates/reconstructs them | state dictionary of `dreamerv3/rssm.py::RSSM` |
| `RSSMTrajectory` | posterior/prior/logits/features `[B,T,...]`, final state | immutable result; not independently checkpointed | `dreamerv3/rssm.py::RSSM.observe/imagine` outputs |
| `RSSM` | `(RSSMConfig, action spaces)`; `initial(B)`, `img_step(state, action, key, is_first)`, `obs_step(state, action, token, key, is_first)`, `observe(state,tokens,actions,is_first,keys)`, and `imagine(state,policy/actions,H,keys)` | caller owns recurrent state; native explicit keys replace source `nj.seed()` while preserving split order | `dreamerv3/rssm.py::RSSM.initial/observe/imagine/_observe/_core` |

### 3.6 Replay classes

| Native class | Constructor; principal input -> output | Persistent state/update | Reset/serialization; exact official symbol |
| --- | --- | --- | --- |
| `ReplayKey` | `(chunk_uuid, offset)`; 20-byte `uint8` step id | immutable identity | no reset; bytes serialize; official `stepid` made stable |
| `ReplayBatch` | `(data, step_ids)` with `[B,K+T,...]` data leaves; `consec` is an `int32` data leaf and step ids are `[B,K+T,20] uint8` | immutable copied sample | `state_dict()` and `ReplayBatch.from_state(state, transition_spaces, latent_spaces, expected_batch_size, expected_time_length)` copy and validate every leaf; `ConsecutiveStream` and `ActiveReport` restoration supply those trusted expected values; official `Replay._getseq/_assemble_batch` result |
| `ReplayChunk` | `(chunk_id, capacity, transition_spaces, latent_spaces, owner_id)`; `append/link/read/update_context` | rows, mutable context, successor, length/refcount/owner metadata | all valid fields serialize; `embodied/core/chunk.py::Chunk` |
| `ReplayWriter` | `(worker_id, replay)`; `add(row) -> ReplayKey` | current chunk/cursor, last-boundary flag, bounded suffix owned through replay | flags remain row data; cursor/suffix serialize; official `Replay.current` writer state |
| `OnlineQueue` | `(maxlen)`; enqueue/dequeue fresh starts | bounded ordered keys | contents serialize; `embodied/core/replay.py::Replay.online` |
| `UniformSelector` | `(seed)`; `insert(item_id)`, `delete(item_id)`, `sample() -> item_id` | dense item ids/index map plus `default_rng` `PCG64` state with exact section-2.3 schema | runtime index map serializes as the exact ordered string-keyed records below; generator state/order roundtrip exactly; `embodied/core/selectors.py::Uniform` |
| `ConsecutiveStream` | `(source, sequence_length=T, consecutive, context=K)`; `next() -> ReplayBatch` | current raw batch/slice index | independent train and training report stream instances serialize; `embodied/core/streams.py::Consec` |
| `DreamerReplay` | `(ReplayConfig, SequenceShapeConfig, transition_spaces, latent_spaces)`; immutable train `raw_length = K + T * C` and report `report_raw_length = K + T_report * C_report`; internally constructs `UniformSelector(seed=0)`; `can_sample_batch(mode)`, `prepare_add/commit_add`, `prepare_sample/commit_sample`, `update_context`, `stats`, `validate` | `can_sample_batch("train" | "report")` independently proves all `B` items of the mode-specific raw length are available without changing selector, stream, queue, replay RNG, limiter, or counters; bounded add/sample plans touch only current chunks, one eviction, selector/queue, two streams/RNG/counters and identity cursors | `state_dict()` returns the complete primitive record including complete PCG64 state and `from_state_dict(state, config, sequence_shape, transition_spaces, latent_spaces)` constructs a fresh validated replay transactionally at the exact next sample; `dreamerv3/main.py::make_replay`; `embodied/core/replay.py::Replay` |

### 3.7 Agent, optimizer, environment, and online-system classes

| Native class/function | Constructor; principal input -> output | Persistent state/update | Reset/serialization; exact official symbol |
| --- | --- | --- | --- |
| `PercentileNormalizerState` | `lo`/`hi` scalar `float32` leaves and a scalar `corr` only when `debias=true`; canonical return normalization has `debias=false` and therefore serializes exactly `{lo,hi}` | stopped EMA values; `state_dict()` copies this exact primitive mapping and `PercentileNormalizerState.from_state(state, config)` rejects wrong fields/shapes/dtypes against its trusted resolved normalizer config | no episode reset; checkpoint leaf; `embodied/jax/utils.py::Normalize` variables |
| `PercentileNormalizer` | `(NormalizerConfig)`; `stats`, `update(state,x)` | pure EMA percentile transition, no gradients | state serializes; `embodied/jax/utils.py::Normalize` |
| `SlowValueState` | slow critic parameter tree with keys/shapes/dtypes identical to the online critic and scalar `int32` count | initialization copies every initialized online leaf into the empty slow tree without casting; scheduled EMA/copy follows every completed official optimizer call | `state_dict()` recursively unfreezes a fresh plain tree and `SlowValueState.from_state(state, online_critic_params, config)` validates exact tree/count against trusted restored online critic leaves and resolved slow-value config; official `SlowModel._initonce/update` |
| `AgentCarry` | encoder carry, `RSSMState`, decoder carry, previous action | caller-owned; call-site gradient rules | `state_dict()` returns a fresh primitive mapping and `AgentCarry.from_state(state, agent, expected_leading_shape)` asks the trusted bound agent for encoder, decoder, RSSM, and action schemas and validates every leaf; first flags reset components; `Agent.initial` carry |
| `AgentLoss` | frozen `(total_loss, named_losses, metrics, carry, context_entries, tokens, replay_features, normalizer_states)`; scalar `float32[]`, named array pytrees with their objective axes, `AgentCarry`, context/token/feature batch-time pytrees, and exactly `retnorm`/`valnorm`/`advnorm` proposals | immutable result consumed by one call whose outer key is an explicit argument; Task 4a declares every field once, Task 4b populates world fields, Task 4c populates imagination/replay-value losses and normalizer proposals, and Task 4d only composes/consumes them | no independent reset/checkpoint or recurring key; `dreamerv3/agent.py::Agent.loss` result |
| `DreamerAgent` | `(obs_space, act_space, config)`; `initial`, `policy`, `apply_replay_context`, `loss`, `report` | supplied Flax params; no hidden mutation | caller carry serializes; `dreamerv3/agent.py::Agent` |
| `DreamerOptimizerState` | optimizer `jnp.int32[]` step plus RMS `jnp.int32[]` step/tree and momentum `jnp.int32[]` step/tree | all three steps and both moment trees advance once per bfloat16/float32 optimizer call | `state_dict()` recursively unfreezes fresh plain mappings and `DreamerOptimizerState.from_state(state, params, config)` validates exact trees/steps/dtypes against trusted restored parameters and resolved optimizer config; official `Optimizer`, `scale_by_rms`, and `scale_by_momentum` |
| `DreamerTrainStateSchema` | immutable closed key/shape/dtype/byte schema derived by `DreamerTrainState.schema(agent, observation_spaces, action_spaces, resolved_config)` | pure abstract derivation; creates no live parameters, moments, normalizers, counters, or device buffers and mutates nothing | trusted static input to checkpoint decode and `DreamerTrainState.from_state`; exact abstract counterpart of the official complete train-path initialization |
| `DreamerTrainState` | params, optimizer, slow/normalizers, scalar `jnp.int32[] update_count`, config hash; no RNG leaf | `DreamerTrainState.initialize(agent, observation_spaces, action_spaces, resolved_config)` runs the exact one-time official sequence; one functional state transition follows per train call | `state_dict()` is a closed primitive tree; `from_state(state, agent, observation_spaces, action_spaces, resolved_config)` orchestrates the component inverses against the pure schema and never initializes or retains candidate containers |
| `validate_next_update_capacity` | `(DreamerTrainState) -> None`; validates the next optimizer/slow/train increments | pure; raises before RNG/replay/limiter consumption if any fixed-width counter cannot advance | native overflow guard around the pinned one-update transition |
| `train_step` | `train_step(agent, state, carry, batch, outer_seed) -> (state, carry, writeback, metrics)` | the bound agent and counter-derived call-local `uint32[2]` key feed one unconditional bfloat16/float32 optimizer update, one slow-value update, then replay writeback | no hidden/global model closure, recurring RNG, retry, or skipped-batch return; section 8; `Agent.train` plus `Optimizer` |
| `DMCSpec` | canonical task plus exact domain/task mapping, profile, mode, public/vector/child seed identity, image size, nullable camera override, effective camera, action repeat, and locked backend identity | immutable closed record; validates the native `wm_marl_seedsequence_v1` public/evaluation offset and `SeedSequence` derivation before construction | serialize by value with the exact section-9 schema; bounded native divergence from the unseeded `embodied/envs/dmc.py::DMC` constructor |
| `DMCState` | public `TypedDict` with exactly `compatibility`, `dmc_spec`, `format`, `format_version`, and `mutable`; private nested TypedDicts close every mapping and give every scalar/array leaf its section-9 type and shape | value record only: `state_dict()` returns a new mapping and owned array copies; restore validates and copies the input and never retains or mutates caller-owned containers/arrays | direct primitive/NumPy checkpoint tree; no object hook, fixture field, or runtime reference |
| `DMCEnvironment` | `(DMCSpec)`; `reset()`, `step(environment_action) -> (observation_row, native_steps: int)`, `state_dict() -> DMCState`, and classmethod `from_state(state, expected_spec) -> DMCEnvironment` | physics/task legacy-MT19937 RNG/counter/pending-reset/current `TimeStep`; restore constructs a replacement and never mutates any existing environment | reset emits a later first row with zero native steps; full exact section-9 state serializes; official DMC/action wrappers |
| `DMCVectorEnvironment` | `(specs)`; batched `reset()`, `step(environment_action) -> (rows, native_steps: int32[N])`, `state_dict() -> list[DMCState]`, and classmethod `from_states(states, expected_specs) -> DMCVectorEnvironment` | ordered child states/seeds; capture returns a fresh list and all restore candidates are staged under one cleanup stack | per-child reset; replacement construction is all-child atomic and never mutates an existing vector; official parallel-env composition |
| `SamplesPerInsertLimiter` | `(samples_per_insert,tolerance,minsize)`; `want_insert()`, `want_sample()`, `insert()`, `sample()` | immutable rates/bounds; scalar signed-`int64` `size` and scalar `float64` `avail`, initialized as section 10 | `state_dict()` returns configuration plus exact primitive state and `from_state_dict()` validates a fresh limiter; `embodied/core/limiters.py::SamplesPerInsert` |
| `PendingReplayRow` | immutable replay row plus worker index, `applied_control` flag, and episode event | produced by one vector call and queued in stable child order | serializes inside `DriverState`; native cooperative bridge between official driver and limiter request paths |
| `CadenceKind` | ordered string enum `evaluation`, `report`, `log`, `checkpoint` | immutable fixed simultaneous-event order | native serialization of official periodic run activities |
| `CadenceRequest` | `(kind, threshold_env_frames, observed_env_frames, event_sequence)` | immutable due event; threshold/observed frames and sequence are signed int64 | serializes in the pending FIFO; created by physical-frame crossings |
| `ActiveReport` | cadence request, phase, immutable staged report batch, nullable immutable computed arrays, and otherwise-unowned emit progress | one bounded phase per service quantum; contains no report carry or RNG input | its staged batch restores through the exact `ReplayBatch.from_state` validation; named `DriverState.report` remains the sole carry owner |
| `ActiveEvaluation` | cadence request, phase, immutable requested episode count, and nullable immutable copied episode output awaiting emission | one full evaluation-vector call per service quantum; contains no action/reset, carry, RNG, accumulator, or return window | serializes only as the scheduler tagged-union payload; named `DriverState.evaluation` remains the sole live evaluation owner |
| `DriverState` | `scheduler`, `collection`, `train`, `report`, `evaluation`, and `summary` subtrees; scheduler owns the two call counters and there is no environment state | sole owner of partial scheduler work and all four carries; report/evaluation/summary update only their own subtree | `state_dict()` projects all tagged records/FIFOs/carries/counters/windows into a fresh tree; `DriverState.from_state(state, agent, run_config, sequence_shape, observation_spaces, action_spaces)` derives vector sizes from trusted run config and validates every nested inverse |
| `RunnerOutput` | frozen `(metrics, scores, open_loop, evaluations, cadence_requests)` where each field is a tuple and `RunnerOutput()` is five empty tuples; mappings and numeric arrays are copied before construction | immutable direct runner output from one scheduler quantum; no writer/checkpoint callback or methods beyond dataclass construction | consumed once after its quantum commits; never serialized independently |
| `DreamerRunner` | fresh constructor `(agent, train_state, replay, train_environment, evaluation_environment, limiter, run_config, sequence_shape)`; cold classmethod `from_state(agent, train_state, replay, train_environment, evaluation_environment, limiter, run_config, sequence_shape, driver_state)`; `advance() -> RunnerOutput` | `__init__` creates the exact initial `DriverState` with both call counters zero; `from_state` calls the exact DriverState inverse and assigns owned references without invoking `__init__` | every completed quantum is serializable even with partial rows/credits; `close()` owns both vectors; official `Driver` plus `parallel.py` limiter transitions |
| `RunnerRestoreCandidate` | private constructor fields `(train_state, replay, limiter, driver_state, train_environment, evaluation_environment, runner, cleanup_stack, transferred=False)`; closed `stage(payload, agent, resolved_config, observation_spaces, action_spaces)`, `transfer() -> DreamerRunner`, and `close()` | calls only named production `from_state`/`from_states` methods in section-10 order; the stack owns created environments until one transfer and reverse-closes them on rejection | not checkpointed; no callback/factory mapping or input-controlled constructor |
| `DreamerRunCoordinator` | fresh `(runner, artifact_writer, checkpoint_manager)`; `consume_output(output)`, `save_safe_point()`, `advance()`, `close()`, and cold `resume(checkpoint_path, expected_identity, resolved_config, observation_spaces, action_spaces, run_dir)` | no independent mutable or serialized state; resume uses the closed production construction order and transfers a staged runner and writer only after validation | runner quantum -> artifact consumption -> optional checkpoint; `close` closes writer and runner with aggregate cleanup; cold resume is section 10 |
| `RunManifest` | authority/config/task/CLI/environment metadata | immutable generation record | atomic JSON; official run config logging |
| `RunSummary` | exact Task-9 compatibility inputs: schema/model/profile/task/mode/seed/status/backend, `config_sha256`, nullable `debug_snapshot`, exact `runtime_overrides`, target and measured counters, evaluation/last-loss/gate/run identity | derived from current state; never reconstructs a measured value from requested budgets | atomically replaced; never checkpoint authority; official summaries |
| `ArtifactWriter` | fresh `(run_dir, manifest)` or cold `resume(run_dir, writer_state)`; append metrics/scores, numbered outputs, `state_dict`, `flush`, `close` | lifecycle `OPEN` or `CLOSED`; durable byte offsets and next-file cursors only | resume directly reconciles active files to checkpointed prefixes before opening handles; official logger conventions |
| `CheckpointPayload` | literal versioned tree in section 10 plus checkpoint generation | immutable candidate snapshot with one owner per leaf | sole complete resume boundary; official runner checkpoint use |
| `CheckpointManager` | `(directory, schema)`; `save(snapshot_without_generation) -> (CheckpointPayload, Path)`, pure `restore_candidate(path, expected_identity, owner_schemas)` | reads the authoritative regular-file `latest`, assigns its successor, and atomically replaces that numbered file and `latest` | short Flax MessagePack wrapper and exact publication in section 10; no algorithm reset |

## 4. Typed configuration

`DreamerProfile`, `ObservationMode`, and `ModelSize` are serialized string enums.
`NetworkSize` is the resolved immutable width table. `RSSMConfig`,
`EncoderConfig`, `DecoderConfig`, `HeadConfig`, `PolicyConfig`,
`OptimizerConfig`, `SequenceShapeConfig`, `ReplayConfig`, `RunConfig`, `LossScaleConfig`,
`ImaginationConfig`, `SlowValueConfig`, and `NormalizerConfig` are frozen
dataclasses. `RuntimeOverrides` and `DebugSnapshot` are also frozen typed
records. Head specializations may validate family-specific values but do
not introduce a second behavior path. The redundant legacy `ActorCriticConfig`
and legacy shape/action constructor fields are removed or migrated to the
canonical aggregate. Removing the class and its package import/export while
adding exactly the five replacement public config interfaces named above is
one Task-1a migration: retaining a compatibility symbol or omitting a new
public interface would preserve the wrong package boundary and is forbidden.
Task 1c's separate `__init__.py` edit is
oracle/tooling-only, and Task 9 owns the later production export surface.

`SequenceShapeConfig` is the sole owner of train `B,T,K,C` and report
`T_report,C_report`. Its defaults are the source-derived
`batch_size=16`, `sequence_length=64`, `context=1`, `consecutive=1`,
`report_length=32`, and `report_consecutive=1`; its two immutable derived
lengths are respectively `65` and `33`.
Neither `ReplayConfig` nor `RunConfig` repeats those fields. `RunConfig` owns no
seed or batch/sequence shape. `DreamerV3Config` owns the complete immutable
aggregate, including exactly one `SequenceShapeConfig` and exactly one scalar
`seed`. Base resolution selects profile, observation mode and task, applies the
full profile snapshot, resolves the profile/mode model default, selects the
pinned authority revision, and validates the aggregate.
The paper profile selects
`bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01`; explicit upstream-current selects
`e3f02248693a79dc8b0ebd62c93683888ddaccfe`. Runtime resolution stores that
single revision string and does not require a reference checkout, source path
tuple, or source digest map. Fixture-generation tests independently hash only
the official blobs actually used by each numerical fixture.

Paper Vision and Proprio resolve 200M. Explicit current Vision resolves 200M;
explicit current Proprio resolves the pinned `size1m` preset. Callers cannot
silently patch paper-critical fields.

`resolve_dreamer_run` is the one typed runtime resolution API. Its exact
signature is
`resolve_dreamer_run(*, mode, task, profile=DreamerProfile.PAPER, seed=0, model=None, debug_local=False, overrides=RuntimeOverrides())`.
All parameters are keyword-only. `mode` and `task` have no defaults and remain
required; `profile` is genuinely omittable and defaults to
`DreamerProfile.PAPER`. Only explicit `DreamerProfile.UPSTREAM_CURRENT` selects
the current snapshot, and positional calls are invalid.
Seed is a primary resolver input, not a `RuntimeOverrides` field. Before any
NumPy conversion, `seed` must be a non-bool Python integer satisfying
`0 <= seed <= 2**32 - 1 - 10_000`; `bool`, NumPy integer scalars, negative
values, and `2**32 - 10_000` are rejected. Validation finishes before NumPy
conversion. Its merge order is exact: (1) base profile/mode/task/model snapshot
plus the validated scalar seed; (2), when requested, the complete
`debug-local-v1` snapshot below; (3) only fields explicitly present in
`RuntimeOverrides`; (4) all field-local and cross-field validation; (5)
canonical JSON and SHA-256 over the final resolved config, including `seed`;
(6) the exact sorted explicit override map. The canonical resolved config
JSON/hash and `ResolvedDreamerRun.identity_state()` therefore include the same
checked `seed`. The allowed identity-bearing override fields are
exactly `env_steps`, `num_envs`, `batch_size`, `batch_length`, `train_ratio`,
`eval_every`, `eval_episodes`, `report_every`, `checkpoint_every`, and `camera`.
The two
batch fields replace the corresponding leaves of the sole
`SequenceShapeConfig`; no second owner is created. Unknown fields, `None`
encoded as explicit values, nonfinite ratios, nonpositive dimensions/cadences,
or a cross-field violation fail before construction. The selected debug
snapshot name, resolved config, canonical JSON/hash, and exact explicit
override maps are manifest and checkpoint identity. `camera` is validated and
returned in the environment override map but is never merged into
`DreamerV3Config` or its hash; the DMC factory later places it in `DMCSpec`.
The no-override wrapper has exact signature
`resolve_dreamer_config(*, mode, task, profile=DreamerProfile.PAPER, seed=0, model=None, debug_local=False)`
and forwards every argument by keyword without a second defaulting or validation
path.

Locked configuration validation compares every non-overridable component leaf
against the selected paper/current profile, including the report sequence,
evaluation-environment count, and report-batch count. `ResolvedDreamerRun`
then reconstructs the expected final config from
`profile/mode/task/seed/model`, the nullable exact `debug-local-v1` snapshot,
and the complete explicit override record. It rejects any component-family
patch, debug presence/value mismatch, or missing, extra, or disagreeing
override before accepting canonical JSON/hash identity; this reconstruction is
pure and does not recurse through `ResolvedDreamerRun`.

`RunConfig.log_every` is a positive physical-`env_frames` integer. Both the
`paper` and explicit `upstream-current` snapshots resolve `log_every=1_000`.
This is a declared native wm-marl operational choice: one log window per 0.1%
of the paper's one-million-frame budget, at most 1,000 metric rows. It is not a
conversion of the pinned source's `run.log_every=120` wall-clock seconds and
does not distinguish algorithm profiles. `log_every` is intentionally not a
`RuntimeOverrides` field and Task 9 exposes no `--log-every`; changing it needs
a typed profile/debug snapshot revision rather than an invocation-only patch.
Thus it is not a public CLI override.

The complete deterministic noncanonical debug snapshot is
`debug-local-v1`: model label `debug-local-v1`; network MLP layers `1` and
units `32`; RSSM `deter=32, stoch=4, classes=4`; Vision encoder/decoder depths
`(8,16,32,64)` with the selected profile's unchanged stride/pooling rule;
`SequenceShapeConfig(batch_size=1, sequence_length=4, context=0,
consecutive=1, report_length=4, report_consecutive=1)`; replay capacity `256`,
chunk size `32`, online queue `16`; run `env_steps=48, num_envs=1,
eval_envs=1, train_ratio=4, eval_every=16,
eval_episodes=1, report_every=16, log_every=16, checkpoint_every=16,
report_batches=1`;
imagination horizon
`5`. It does not change loss scales, distributions, detach switches,
normalizer/slow-value equations, action repeat, image size, task time limit,
modality, optimizer ordering, or authority revision. This snapshot plus no
explicit override is the resolved identity for the literal 48-frame CPU
command; explicit Task-9 flags then replace only their named allowed leaves.
Task 1 tests every Task-9 override individually and together.

`--out-dir`, `--resume`, `--dry-run`, `--dry-run-matrix`, and
`--stop-after-env-steps` are invocation-only and excluded from config JSON,
hash, override maps, manifest identity, and checkpoint compatibility. Camera is
outside Dreamer config identity but remains in the environment override map,
canonical argv, `DMCSpec`, manifest environment identity, and checkpoint
compatibility as defined below.

The resolved target budget is immutable run identity. An initial invocation
and every resume therefore resolve the same `RunConfig.env_steps` and canonical
config hash. The debug-only CLI option `--stop-after-env-steps INT` is a
per-invocation interruption control, not a `DreamerV3Config`, checkpoint
identity, or `RunManifest` field. It is valid only with `--debug-local`, must be
positive, strictly below the resolved target, and greater than the restored
frame counter. Final targets and debug-stop values are lower bounds for the
synchronous vector interface. The runner stops collection after the first
full-vector quantum whose actual native-frame result reaches or crosses the
bound, then reaches a quiescent safe point after all rows, earned updates, and
crossed cadences belonging to that turn. At that
safe point it flushes direct artifacts, durably publishes a complete checkpoint,
and only then exits zero. The interruption control never changes run identity
or adds a process-history artifact.

Single-run `--dry-run` resolves and validates the same canonical configuration,
then atomically writes exactly one `manifest.json` and exits without
constructing an environment, model, replay, writer, checkpoint manager, or
runner. Its exact keys are `schema_version`, `kind` (literal
`dreamer_v3_dry_run`), `profile`, `observation_mode`, `task`, `seed`, `camera`,
`resolved_config`, `config_sha256`, `authority_revision`, `debug_snapshot`,
`runtime_overrides`,
and `canonical_argv`; the digest is over canonical `resolved_config`, and no
ephemeral run id is minted. The revision exactly equals the resolved profile's
pinned authority. `canonical_argv` is the normalized identity-bearing argument list,
excluding `--out-dir`, `--dry-run`, and per-invocation controls; for the Task 9
paper Vision command it is exactly `['--profile','paper','--observation-mode',
'vision','--task','walker_walk','--seed','0']`. Temporary-sibling write/fsync, rename, and parent
directory fsync publish the sole file. Production construction is exercised by
a separate `--debug-local` subprocess, never by calling the dry-run path a
one-step runner.

The profile snapshot test is the authority for every scalar. In particular the
paper profile uses the table in section 1, while explicit upstream-current
uses the pinned `configs.yaml` values. Unsupported combinations fail before
environment or accelerator initialization.

## 5. Distributions, networks, and RSSM

### 5.1 Outputs

Every output object exposes official-style `pred`, `sample(seed)`, `logp`,
`prob`, `entropy`, `kl`, and `loss` where defined. Event dimensions reduce
exactly once and batch/time dimensions remain. Direct parity tests cover
`prob(event)` for every implemented distribution and both plain and aggregate
events; the continuation head explicitly compares `prob(1)` with the pinned
`Binary` output.

- `MSEOutput(mean, squash=None)` returns elementwise squared error against
  `stop_gradient(squash(float32(target)))`, with no factor one-half, while
  `pred()` remains the encoded mean. Proprio continuous reconstruction heads
  use `squash=symlog`; this does not symexp their prediction.
- `AggregateOutput` reduces exactly its configured trailing event axes.
- `BinaryOutput` uses stable Bernoulli cross entropy; continuation target is
  based on `is_terminal`, not `is_last`.
- `NormalOutput` implements the exact bounded-normal head:
  `mean=tanh(raw_mean)` and
  `stddev=(maxstd-minstd)*sigmoid(raw_std + 2)+minstd`, without a tanh sample
  transform or Jacobian term.
- Categorical uniform mixing is opt-in and defaults off. Ordinary categorical
  heads construct `CategoricalOutput(logits)` with no mixing; policy OneHot and
  RSSM categorical latents (the OneHot/RSSM cases) pass their configured unimix to `OneHotOutput`.
  `OneHotOutput` samples a hard one-hot with straight-through probability
  gradients.
- `TwoHotOutput` uses 255 uniform symlog coordinates transformed by `symexp`,
  clamps targets, interpolates adjacent bins, and computes the ordered
  negative/positive expectation used by the source.

### 5.2 Network primitives

`Initializer` maps source initializer names and preserves zero output kernels.
`RMSNorm` divides by final-axis RMS without mean subtraction. `Linear`,
`BlockLinear`, `Conv2D`, and `MLP` preserve source parameter naming, shapes,
normalization, activation, and initialization. The recurrent matrix is an
eight-block `BlockLinear`, never a dense GRU substitute.

`BlockGRU` normalizes each action component by a stopped
`max(1, abs(action))`, separately projects deterministic, stochastic, and action
inputs, repeats the concatenation across blocks, and emits reset/candidate/update
gates in source order. It resets state, stochastic input, and action at
`is_first` before transition.

For flattened leading size `M`, deterministic width `D`, and `g=8` blocks, the
three normalized input projections are concatenated, repeated to
`[M,g,*]`, joined with `reshape(old_deter,[M,g,D/g])`, and flattened before the
configured block hidden layers. The final
`BlockLinear(units=3*D, blocks=g)` output is reshaped to `[M,g,3*D/g]`, split on
the last axis, and flattened into `reset_raw`, `cand_raw`, and `update_raw`, each
`[M,D]`. The core equations are exactly

```text
reset  = sigmoid(reset_raw)
cand   = tanh(reset * cand_raw)
update = sigmoid(update_raw - 1)
deter  = update * cand + (1 - update) * old_deter
```

No conventional GRU rearrangement or candidate bias is substituted.

`DictEncoder` selects declared keys, sorts them, maps `uint8` images to
`float32 / 255 - 0.5`, symlogs vector values, and emits tokens with outer
`[B,T]` dimensions preserved. For image keys, it concatenates sorted HWC images,
casts and scales once, then flattens only the leading axes to `[M,H,W,C]`. At
each resolved depth stage, the paper profile applies
`Conv2D(kernel=5,stride=2)` followed by RMS normalization and activation. The
explicit upstream-current profile applies `Conv2D(kernel=5,stride=1)`, reshapes
the result exactly to `[M,H//2,2,W//2,2,Cstage]`, takes `max(axis=(2,4))`, and
only then applies normalization and activation. Thus pooling never precedes the
convolution or normalization.

`DictDecoder` returns one output object per observation key. Let
`H0,W0 = image_resolution / 2**len(depths)`, `C0=depths[-1]`,
`u=H0*W0*C0`, and `g=bspace=8`. Its image branch separately consumes
`deter [M,D]` and flattened `stoch [M,S*C]`. It projects deter with
`BlockLinear(units=u, blocks=g)` and rearranges the flat result exactly as
`'... (g h w c) -> ... h w (g c)'` with `h=H0,w=W0,g=g`. It projects stoch
through `Linear(2*units) -> norm -> activation -> Linear([H0,W0,C0])`, sums the
two spatial tensors, then applies norm and activation. In reversed depth order,
the paper profile applies stride-2 transposed convolution followed by norm and
activation; upstream-current repeats rows and columns by two, applies ordinary
stride-1 convolution, then norm and activation. Because `outer=false`, the
final paper image layer is another stride-2 transposed convolution and the
current layer is another repeat-by-two plus convolution. The final image tensor
is `sigmoid(logits)`, reshaped back to `[B,T,H,W,Ctotal]`, and split by declared
image-channel widths.

Each split image is exactly
`AggregateOutput(MSEOutput(out), event_ndims=3, reduction=jnp.sum)`. Its target
is stopped `float32(image)/255`; squared error is summed over the complete HWC
event and leaves a `[B,T]` loss before the per-key batch/time mean. Composed
Vision fixtures for both profiles verify encoder tokens, posterior/prior
states, decoder sigmoid predictions, per-image HWC reconstruction losses, the
world scalar, and gradients through encoder, RSSM, and decoder. Vector heads
use the declared space/family. `MLPHead` constructs reward, continuation,
policy, and value distributions from resolved configuration.

### 5.3 RSSM

`RSSM.initial(B)` returns zero state. `img_step(state, action, seed, is_first)`
runs reset, block GRU, prior logits, uniform mixing, and stochastic one-hot
sampling. `obs_step` then forms posterior logits from deterministic state and
encoder token and samples the posterior. `observe` scans these operations over
`T`; `imagine` scans prior transitions only. Neither uses argmax or injects
posterior information after imagination starts.

`ninjax_scan_sample_keys` is fixture tooling, not a production RSSM algorithm.
Task 1 moves its sole definition and import path to
`rssm_oracle.py::ninjax_scan_sample_keys`; parity fixtures import that tooling
module directly. Production `RSSM.observe` consumes explicit caller-owned keys,
and neither `rssm.py` nor the package root imports or exports the helper.

Dynamics KL is
`max(1 nat, KL(stop_gradient(posterior) || prior))`. Representation KL is
`max(1 nat, KL(posterior || stop_gradient(prior)))`. The categorical event axis
is reduced exactly as the official output wrapper specifies. Tests cover
non-singleton `B`, `T`, `S`, and `C`, mid-sequence resets, supplied sampling
noise, and gradients.

## 6. Replay and replay context

`ReplayKey` is the stable chunk-id/offset identity encoded into 20 `uint8`
bytes. `ReplayChunk` stores fixed-capacity immutable transition leaves and
mutable context leaves, links to at most one successor, and exposes explicit
`state_dict`/restore. `ReplayWriter` owns one open cursor per environment,
retains the bounded suffix needed to emit starts, and seals/opens chunks without
crossing writer identity. Episode boundaries are flags inside a writer stream,
not discontinuities in storage.

`DreamerReplay` is the sole identity allocator. Allocatable chunk IDs are
`[1, 2**128 - 1]`; the Python-integer `next_chunk_id` cursor has state domain
`[1, 2**128]`, where `2**128` is the exhausted sentinel. Allocation encodes an
allocatable value big-endian into exactly 16 bytes, checks that neither the live
chunk map nor any writer/link already owns it, and preflights exhaustion before
changing a cursor or chunk. A successful allocation of `2**128 - 1` sets the
cursor to the valid nonallocatable sentinel; the next allocation request fails
without mutation. Allocatable item IDs are `[0, 2**63 - 1]`; the Python-integer
`next_item_id` cursor has state domain `[0, 2**63]`, where `2**63` is the
exhausted sentinel. Selector insertion consumes the current allocatable value
and increments only after the start and collision checks succeed. These two
identity cursors are explicit exceptions to the signed-`int64` rule; identity
payloads use checked Python integers so both sentinels roundtrip exactly.
Restore accepts either sentinel, rejects anything outside the cursor domain,
and validates both cursors against all chunk, item, writer, queue, and link
identities. Tests allocate each final valid ID, serialize/restore the exhausted
state, and prove only the next request is rejected with no mutation.

Runtime replay maps remain direct and efficient, but their checkpoint form must
be accepted by public Flax 0.10.4 restore, whose MessagePack decoder rejects
integer map keys. `UniformSelector.indices: dict[int,int]` therefore serializes
as a list of exact string-keyed records `{"index": int, "item_id": int}` in
increasing item-id order. `DreamerReplay.writers:
dict[int,ReplayWriter]` serializes as a list of records
`{"state": ReplayWriterState, "worker_id": int}` in increasing worker-id order.
Restore requires exact record keys/types, strict order, no duplicates, exact
selector key/index agreement, and nested/outer worker-id agreement before it
reconstructs fresh runtime maps. `refs` retains its bytes-key map because public
Flax restore accepts bytes keys and canonicalizes them. Task 2b recursively
checks a representative complete replay state and rejects any remaining
integer-key mapping, so later replay subtrees cannot silently reintroduce the
same incompatibility. It owns only these two state-boundary conversions and
proves public-Flax roundtrip plus exact next behavior; the checkpoint codec adds
no mapping adapter.

`DreamerReplay.raw_length = K + T * C` and
`DreamerReplay.report_raw_length = K + T_report * C_report` are immutable public
properties derived from the sole `SequenceShapeConfig`; neither is a trimmed
learner length. Direct tests use nonzero context and consecutive counts greater
than one to distinguish them. For a mode-specific raw length `R`, a start enters the uniform selector
only after all `R` linked rows exist. Online eligibility has an independent
scalar `int64` phase counter per writer. On each writer-local add, the source
ordering is binding: test the old writer length for online eligibility, enqueue
the eligible start, and only then increment that writer's length once. In
zero-based logical-start coordinates this yields `1 + n*R` for `R>1`; for
`R=1`, every start is eligible. Multi-writer phases never share a counter.
Eligible starts enter the one global online queue in the exact global `add()`
call order, so equal-time writers are resolved by the caller's stable worker
order. All writer phase counters and the FIFO queue serialize, and restore must
reproduce the initial phase as well as a partially advanced phase. Training
drains eligible online starts before uniform fill; the training report stream
samples uniformly and never consumes the train queue. `ConsecutiveStream` holds a raw
batch and current slice index for one mode and returns `[B,K+T,...]` slices at
offsets `i*T`; train and report own independent stream state. There is no
evaluation replay stream. The deliberately minimal policy-return evaluation
emits episode returns/lengths only and evaluation never owns or mutates replay.
`DreamerAgent.report` receives batches solely from the training report stream.

Sampling copies leaves, forces `is_first[:,0] = True`, and sets
`is_last[:,:-1] |= is_first[:,1:]`. Natural resets can occur inside a sampled
sequence. Step ids must follow chunk links exactly. Eviction removes only items
whose reference counts reach zero and never invalidates a selected live start.

`DreamerAgent.apply_replay_context` first constructs normal previous actions by
prepending `AgentCarry.prev_action`. Replay-context reconstruction and the
`[:,K-1:-1]` slice are entered only when `K>0`. For the first consecutive slice
(`consec[:,0] == 0`) with `K>0`, it reconstructs encoder/RSSM/decoder carries
from the first `K` stored entries, drops the first `K` observations and step
ids, and uses replay actions `[:,K-1:-1]` as previous actions. Later slices use
incoming carry and the normal shifted actions. At `K=0`, no context branch or
negative-index slice is evaluated: every slice uses the incoming carry and the
normal prepended-previous-action alignment from section 2.2. Training writes
inferred context entries back to their exact step ids and sets outgoing
`prev_action` to the final replay action. Task 2 includes explicit zero-context
and nonzero-context assertions for carry, previous actions, observations, and
step ids.
Equivalently, at `K=0` the normal path is mandatory.

Identity-resume coverage checkpoints immediately before a chunk rollover and
item allocation, restores, and compares all later chunk/item IDs, samples,
online-queue entries, successor links, and latent writeback targets with the
uninterrupted branch.

Replay insertion performs work bounded by the current row, current/open chunk,
bounded context suffix, and any single planned eviction. It must not scan
capacity, lifetime history, source text, callbacks, or interpreter state on the
hot path. `validate()` is an explicit offline audit for chunk links, selectors,
queues, references, ownership, and counters. Persistence validates schema,
config/space signatures, chunk geometry, refs, item/selector bijection, writer
cursors, both streams, queue, selector RNG, identity/sample counters, and mutable
context. Restore is transactional. Security hardening unrelated to algorithmic
state and lifetime-growing allocation histories are outside scope.

## 7. Unified agent objective

### 7.1 Agent state and policy

`PercentileNormalizerState` stores stopped EMA 5th and 95th percentiles.
It contains `corr: float32[]` only when its config has `debias=true`; the
canonical return normalizer has `debias=false` and serializes only `lo` and
`hi`. `PercentileNormalizer.update(returns)` produces new state and returns
`offset=low`, `scale=max(high-low,1)` at rate `0.01`, without gradients. The
canonical value and advantage normalizers are identity. `SlowValueState` owns a
separate critic-parameter tree updated at rate `0.02` every update; it is never
in the gradient parameter tree. Before the first value call/update, the slow
critic tree is empty. Initialization applies pinned `SlowModel._initonce`
semantics: require a nonempty initialized online critic, copy each source key
and array into the slow path with identical key, shape, value, and dtype, and
set scalar `int32` count to zero. The first train update computes
`mix = rate` because `count % every == 0`, writes
`mix * online + (1 - mix) * slow` leafwise, and increments count to one. A
failed train call publishes neither the optimizer result nor this update; there
is no rejected-batch state or subsequent continue path. The complete copied
tree and count are serialized.

`DreamerAgent.initial(B)` creates component carries and zero previous actions.
`DreamerAgent.policy(params, carry, observation, mode, outer_seed)` preprocesses
once, encodes, performs one posterior step, evaluates the policy, and samples
in both train and evaluation modes. It returns the new carry, model-action tree,
latent replay entries, and finite diagnostics. Mode does not substitute mean or
argmax actions.

### 7.2 World loss

Decoder outputs, reward, continuation, and both KL terms are per-step `[B,T]`.
Each observation key has its own reconstruction loss and its own mean; keys are
not pre-averaged. With default scales, the scalar world objective is

```text
sum_key mean(reconstruction[key])
+ mean(reward_loss)
+ mean(continuation_loss)
+ mean(dynamics_KL)
+ 0.1 * mean(representation_KL)
```

Reward target is the replay reward. Continuation target is
`float(~is_terminal)` and, when `contdisc` is enabled, is multiplied by
`1 - 1/horizon`. `reward_grad=false` stops representation features only on the
reward-head path; canonical `reward_grad=true` leaves that path connected.
Reconstruction targets and all supervised scalar targets are stopped.

### 7.3 Every-state imagination and lambda returns

With canonical `imag_last=0`, all valid posterior states after context removal
are starts: `Kstart=T`. RSSM start state: `[B*T,...]`, obtained by flattening
the replay batch/time axes without adding a time dimension. RSSM imagined output: `[B*T,H,...]`
for `RSSM.imagine(starts, policy, H, ...)` features and prior actions.
Separately, the replay posterior feature and final policy action are each
expanded to one row `[B*T,1,...]`. The concatenated feature/action sequences: `[B*T,H+1,...]`.
No `[B*T,1,...]` value is passed as
the RSSM recurrent start. If `ac_grads=false`, the initial replay feature is
stopped before concatenation; if `ac_grads=true`, gradients may reach only that
initial replay feature. Every feature generated after imagination begins is
stopped in both branches, and every imagined policy call receives stopped
carries. The switch never changes the world-model objective. Tests must prove
both branches: zero initial-feature gradient when false, source-matching
initial-feature gradient when true, and zero later-feature/carry gradient in
both.

For predicted continuation `con`, source ordering is:

```text
disc = 1                         if contdisc else 1 - 1/horizon
weight = cumprod(disc * con, axis=time) / disc
target_value = slow_value if slowtar else online_value
return = lambda_return(last=0, terminal=1-con, reward,
                       target_value, bootstrap=target_value,
                       disc=disc, lambda=0.95)
```

`lambda_return` computes, for time indices after the first,
`live=(1-terminal[t])*disc`, `cont=(1-last[t])*lambda`, and reverses
`reward[t] + (1-cont)*live*bootstrap[t] + live*cont*next_return`. Returns,
target values, and weights used as targets are stopped at the loss boundary.

The return normalizer updates from stopped returns. Actor advantage is
`(return - target_value[:-1]) / return_scale`, then the configured advantage
normalizer (identity canonically) is applied. The policy loss is

```text
stop(weight[:-1]) * -(
  logp(policy, stop(action[:-1])) * stop(normalized_advantage)
  + 3e-4 * sum(action_entropies))
```

There is no pathwise action shortcut. The value loss fits normalized stopped
returns and adds the configured online-value likelihood loss toward stopped
slow-value predictions. `slowtar=false` means online values, not slow values,
form canonical lambda-return targets; the slow network remains a regularizer.

Replay-value loss uses posterior replay features (connected only when
`repval_grad=true`), replay rewards, terminal/last flags, and the first imagined
return as bootstrap. Replay discount is unconditionally `disc = 1 - 1 / horizon`
before replay `lambda_return`, independent of
`contdisc`; only imagination switches between continuation-only `disc=1` and
the finite-horizon value. Thus imagination switches while replay uses the fixed finite-horizon value
in both branches. Its weight is `float(~is_last)`, lambda-return
distinguishes termination from truncation, and slow-value regularization
matches the imagination critic. `repval_loss=false` removes only this term. The
complete default scalar loss adds `policy + value + 0.3*repval` to the world
objective.
Normalizer transitions are computed inside the loss call, and the slow-target
update follows the optimizer call exactly once per completed training update;
neither is an accidental parameter gradient.

`AgentLoss` returns the scalar, every named per-step loss, metrics, new component
carries, context writeback, and proposed normalizer state; it returns no RNG
state. `DreamerAgent.policy(params, carry, observation, mode, outer_seed)` and
`DreamerAgent.report(params, carry, batch, outer_seed) -> (arrays,
new_report_carry)` receive call-local counter-derived keys.

Report reconstructs the observed prefix and rolls the suffix forward using
only recorded future actions. It materializes target/prediction/error video
panels only for keys in the decoder's image-key set, exactly matching official
`for key in self.dec.imgkeys`. Vision open-loop cursor/files are nonempty after
a due successful report. Proprio has no decoder image key: Proprio open-loop
cursor is zero, Proprio open-loop directory is absent, and fabricating a
numeric/video output is an error. The evaluation cursor/files are nonempty in
both modalities.

- Vision open-loop cursor/files are nonempty.
- Proprio open-loop cursor is zero.
- Proprio open-loop directory is absent.
- evaluation cursor/files are nonempty in both modalities.

The caller commits one `batch_seed_counter` advancement per successful report
computation together with the returned carry and stages copied arrays for the
later writer phase. Sampling or computation failure commits neither counter
nor carry. Once computation succeeds, the owner is advanced even if the later
artifact write fails: that failure terminates the coordinator without
publishing another checkpoint, so cold resume deterministically replays the
report from the last published checkpoint. No callback or generic two-phase
transaction framework is introduced. Report purity forbids
parameter, optimizer, normalizer, slow-value, and replay mutation; it does not
mean key reuse. Here replay mutation means the Agent cannot change stored rows,
context, selector, or stream state; the scheduler's preceding transactional
`stage-report-batch` phase is the sole caller-owned advancement of the separate
training report stream needed to supply the immutable batch. The shared batch
counter makes successive train/report calls use the official interleaved
outer-key sequence. `ActiveReport` serializes only
the request, phase, staged immutable batch, copied result, and emit progress; it
does not duplicate report carry or the shared counter. `DriverState.report`
remains the single carry owner and `DriverState.scheduler` the single counter
owner. A mid-report resume therefore produces the same arrays, returned carry,
counter, and emitted file as uninterrupted
execution. Task 7 owns report sampling, computation, copied-output staging, and
`test_report_compute_failure_no_advance`. Task 8c exclusively owns writer
failure, cold-resume replay, post-crossing artifact/checkpoint failure, and
`test_report_writer_failure_cold_resume_replays`.
This is one counter advancement per successful report batch.
In compact failure-boundary terms: sampling or computation failure advances
neither report carry nor batch counter; successful compute atomically advances the
named owner and stages copied output; writer failure may leave only
uncheckpointed in-memory advancement, terminates without another checkpoint,
and cold resume replays from the last durable generation.

## 8. Optimizer and train step

`DreamerOptimizerState` stores the official optimizer step, independent int32
RMS and momentum steps, and float32 RMS and momentum pytrees. For each parameter
tensor, update order is:

1. AGC: scale the tensor gradient to at most
   `0.3 * max(1e-3, ||parameter||_2)`;
2. RMS second moment with profile `beta2`, epsilon `1e-20`, step increment, and
   bias correction;
3. divide the clipped gradient by `sqrt(corrected_rms) + epsilon`;
4. momentum with `beta1=0.9`, its own step increment and bias correction;
5. optional declared weight decay exactly as pinned
   `optax.add_decayed_weights`: for a matched parameter add `wd * parameter` to
   the normalized momentum and call that complete value `laprop_output`;
6. compute the descent update
   `parameter_delta = -schedule(s) * laprop_output`; `schedule(s)` already
   contains `L` exactly once, and the negative sign is the default
   `flip_sign=True` behavior of pinned `optax.scale_by_learning_rate`;
7. use additive `optax.apply_updates(parameter, parameter_delta)`, write the new
   optimizer state, and increment
   the optimizer step by one.

RMS before momentum is LaProp. Reordering, Adam, a global clip, or missing bias
correction is nonconformant. Let `s` be the global train-update index at
which the rate is evaluated, `W` the warmup length, `A` the global anneal
endpoint, the post-warmup local step `u = s - W`, and
`q = clip(u / (A - W), 0, 1)`. The offset is applied exactly once. The exact
schedules are:

```text
warmup(s) = L * clip(s / W, 0, 1)                         (W > 0)
const(u)  = L
linear(u) = L + (0.1 * L - L) * q
cosine(u) = L * ((1 - alpha) * 0.5 * (1 + cos(pi * q)) + alpha)
alpha     = 0.1 * L  # pinned cosine_decay_schedule third argument, unusually scaled
schedule(s) = warmup(s) if s < W else selected(u)
```

For `W=0`, the selected schedule begins immediately. Linear/cosine require
`A>W`; constant ignores `A`. The schedule index is the official optimizer/train
call count and advances once per bfloat16/float32 update.

Because weight decay precedes `scale_by_learning_rate` in `Agent._make_opt`, it
is inside the single negative schedule multiplication; it is neither an
unscaled post-update subtraction nor a second learning-rate multiplication.
For a positive scalar gradient, a positive no-decay parameter must decrease.
Equivalently, schedule(s) already contains `L` exactly once, the implementation
uses additive `optax.apply_updates`, and the parameter decreases; there is no
second rate factor.
Task 5 checks parameter decrease and no extra `L` for constant, warmup, linear,
and cosine schedules at every step of the five-step fixture, in addition to the
full LaProp state trajectory.

The pinned `Optimizer.__init__` wraps its Optax transform in
`optax.apply_if_finite` only when `COMPUTE_DTYPE == float16`, because that is the
only loss-scaling branch. Neither conformance profile uses float16. For the
paper and upstream-current bfloat16/float32 paths, `Optimizer.__call__` always
computes `opt.update`, applies the returned updates, writes optimizer state, and
increments its step by one. The native implementation must not add a
proposal-wide finite scan, rejected-batch counter, no-op writeback, retry, or
skip-and-continue branch. Optional diagnostics may terminate the run before a
new state or artifact generation is published, but they never convert a sampled
batch into a resumable rejected update.

`DreamerTrainState` owns parameters, optimizer state, slow-value state,
normalizer states, scalar `jnp.int32[] update_count`, and config hash. It owns
no outer-call key or counter. Before the next counter increment would exceed
`2**31 - 1`, `validate_next_update_capacity` fails before limiter availability consumption,
replay sampling, or outer-key derivation.

Fresh construction is one exact public boundary:
`DreamerTrainState.initialize(agent, observation_spaces, action_spaces, resolved_config)`.
It runs exactly once during fresh Task-9 bootstrap and never during cold
resume. The argument is exactly `resolved.config`, so
`resolved_config.seed == resolved.config.seed` is checked before initialization.
Its paper-authority sequence is the direct native translation of official
`_init_params`:

1. create the raw parameter seed `uint32([resolved_config.seed, 0])` without
   reading either call counter;
2. create zero complete observation/action/extra data `[B,T+K,...]` from the
   trusted spaces, with `B=resolved_config.batch_size`, `T=batch_length`, and
   `K=replay_context`;
3. trace/materialize the complete train carry through the native equivalent of
   `model.init_train`, then trace/materialize the complete train path with that
   carry, data, and raw seed so every encoder, decoder, RSSM, head, policy,
   critic, and slow-source parameter exists exactly once;
4. initialize the optimizer step/moment trees from the complete parameter tree,
   copy the online critic leaves byte/dtype-exactly into the slow-value tree,
   initialize each configured normalizer, set all fixed train counters to zero,
   and store the resolved config identity.

`DreamerTrainState.schema(agent, observation_spaces, action_spaces, resolved_config)`
uses the same trusted spaces and abstract train path via shape evaluation. It is
pure: it allocates no live parameter/device buffer or runtime state and mutates
nothing. It returns `DreamerTrainStateSchema`, the exact closed parameter,
optimizer, slow-value, normalizer, counter, config-key, shape, dtype, and byte
schema. Initialization must match that schema leaf-for-leaf. Static checkpoint
decode derives it after constructing the non-resource agent; it never calls
the initializer. `DreamerTrainState.from_state(state, agent,
observation_spaces, action_spaces, resolved_config)` uses that schema and
orchestrates `DreamerOptimizerState.from_state`,
`SlowValueState.from_state`, and every
`PercentileNormalizerState.from_state`, rejecting all missing/extra or
shape/dtype/config/count mismatches without retaining candidate containers.

`train_step(agent, state, carry, batch, outer_seed)` applies replay context;
constructs the exact section-2.4 within-call scan/action schedule from the
call-local legacy `uint32[2]` key; computes the
unified loss and gradients; and performs the unconditional AGC/LaProp update.
It then follows pinned `Agent.train` ordering: update slow value, construct and
publish replay-context writeback, set outgoing previous action from the final
replay action, and return `(new_state, new_carry, writeback, metrics)`. One
sampled batch causes exactly one update, one caller-owned batch-counter
advancement at commit, one
`update_count` increment, one optimizer/RMS/momentum step increment, one
slow-value update, and one replay writeback. Any thrown diagnostic failure ends
the functional call before it returns a new train state or writeback; collection
does not silently sample the next batch. The later run coordinator owns the
integration rule that a failed turn cannot publish a checkpoint or batch
counter increment. Task 5 compares the initialized state to the pure schema,
proves its first transition against the official fixture, and proves fresh
bootstrap invokes initialization once while Task-8 cold restore invokes it zero
times.

## 9. DMC environment contract

`DMCSpec` is immutable and includes task, mode, seed, image size, optional camera
override, action repeat, and profile. Both modes accept exactly:

The deterministic schedule is named exactly `wm_marl_seedsequence_v1`. It is
one bounded native environment-integration/reproducibility exception shared by both profiles,
not paper/current algorithm behavior and not a translation of official DMC.
Direct source truth: both pins omit `use_seed` for DMC; `make_env` therefore forwards no seed. The
official `DMC` accepts no seed and calls `suite.load(domain, task)`. With no
`task_kwargs`, locked dm-control creates `RandomState(None)` automatically.
These source facts are tested independently from the native policy.

```text
acrobot_swingup, ball_in_cup_catch, cartpole_balance,
cartpole_balance_sparse, cartpole_swingup, cartpole_swingup_sparse,
cheetah_run, finger_spin, finger_turn_easy, finger_turn_hard, hopper_hop,
hopper_stand, pendulum_swingup, quadruped_run, quadruped_walk, reacher_easy,
reacher_hard, walker_run, walker_stand, walker_walk
```

The canonical identifier is resolved through one explicit total mapping, never
by splitting on the last underscore:

```text
acrobot_swingup -> (acrobot, swingup)
ball_in_cup_catch -> (ball_in_cup, catch)
cartpole_balance -> (cartpole, balance)
cartpole_balance_sparse -> (cartpole, balance_sparse)
cartpole_swingup -> (cartpole, swingup)
cartpole_swingup_sparse -> (cartpole, swingup_sparse)
cheetah_run -> (cheetah, run)
finger_spin -> (finger, spin)
finger_turn_easy -> (finger, turn_easy)
finger_turn_hard -> (finger, turn_hard)
hopper_hop -> (hopper, hop)
hopper_stand -> (hopper, stand)
pendulum_swingup -> (pendulum, swingup)
quadruped_run -> (quadruped, run)
quadruped_walk -> (quadruped, walk)
reacher_easy -> (reacher, easy)
reacher_hard -> (reacher, hard)
walker_run -> (walker, run)
walker_stand -> (walker, stand)
walker_walk -> (walker, walk)
```

Thus `ball_in_cup_catch` calls `suite.load("ball_in_cup", "catch")`, rather
than producing a nonexistent domain or task by naive underscore splitting.

Paper validation requires repeat 1. Default camera is 2 for quadruped tasks and
0 for every other task. A CLI camera override changes `DMCSpec`, the runtime
override map, canonical argv, manifest environment identity, and checkpoint
compatibility; it is not part of `DreamerV3Config` or its config hash.
The adapter uses the native Control Suite task time limit and adds no independent
truncation. Debug runs reduce collection budget, model/batch sizes, or cadence,
not the task time limit, modality, action conversion, or boundary semantics.

`DMCEnvironment` wraps one dm_control task. Its `step(environment_action)`
receives the full environment-action mapping defined in section 2: all
model-action leaves plus the boolean `reset` leaf. `DMCVectorEnvironment.step`
receives the same mapping tree with a leading environment axis on every leaf,
and the driver serializes that same mapping tree as its pending action. The
adapter renders `[64,64,3] uint8` pixels with the task's declared/default camera
for Vision or exposes declared proprioceptive observation leaves for Proprio.
The model-action leaves deliberately accept
finite raw normalized-coordinate samples outside the nominal action interval.
Only at this call boundary does it clip to `[-1,1]`, then linearly scale to each
dm_control action bound, reproducing the official
`ClipAction(NormalizeAction(DMC))` order in `dreamerv3/main.py::wrap_env`.
Agent/replay never preclip or reject that sample. The adapter attempts the
scaled action up to `repeat` native controls and accumulates rewards according
to the official adapter, breaking early on `is_last`/`is_terminal`. It returns
the actual native-step count with the row: zero for reset and `1..repeat` for a
non-reset action. Both shipped profiles require `action_repeat=1`; the general
counter still uses the returned count.

`is_terminal` is true only for an environment terminal, while `is_last` is true
for terminal or time-limit truncation. The terminal/truncated observation and
reward are emitted as their own final row. The next call with the boolean
`reset=true` leaf
ignores continuous action, resets, and emits `is_first=true`; it never overwrites
the preceding final row. Continuation therefore uses `~is_terminal`, while RSSM
reset and replay boundaries use `is_first`/`is_last`.

Resume compatibility is intentionally bound to the locked backend
`dm-control==1.0.17`; a different installed version fails before environment
construction. A public CLI seed must nevertheless produce reproducible fresh
local runs, distinct train/evaluation and vector-child streams, the
deterministic all-20 fixture, and exact uninterrupted-versus-resumed tests.
That repository requirement is the rationale for `wm_marl_seedsequence_v1`.
`DMCVectorEnvironment` owns `N` independent instances. For base seed `s` and
zero-based child index `i`, the native deterministic child seed formula is exactly
`int(np.random.SeedSequence([np.uint32(s), np.uint32(i)]).generate_state(1,
dtype=np.uint32)[0])`; both inputs are range-checked before conversion and the
result is serialized in the child `DMCSpec`.

The public seed domain is exactly
`0 <= public_seed <= 2**32 - 1 - 10_000`. Checked Python-integer arithmetic
defines `train_base_seed = public_seed` and
`evaluation_base_seed = public_seed + 10_000` before either value is converted
to `np.uint32`; the chosen `+10_000` role offset is native policy and cannot
wrap around.
This check occurs before `np.uint32` conversion.
For each vector, `0 <= i <= 2**32 - 1` is checked as a Python integer, then the
equivalent literal formula
`SeedSequence([base_seed, child_index])` above derives its child seed. The
manifest identity, both top-level `DMCSpec` values, every serialized child
`DMCSpec`, and checkpoint identity record this mapping. Task 0b's pure closed-
spec matrix validates the mapping and derived values without constructing extra
environments. Production cold-resume rejection is implemented and tested later
by Tasks 8b-8c before environment construction; Task 0b is not evidence for
that future checkpoint path.

Parity tests that compare DMC traces inject and record the same explicit native
child seed on both sides. Source-conformance tests separately prove the
automatic unseeded official constructor and this declared divergence. DMCSpec
remains the only serialized representation of the policy through its existing
public/base/child fields; there is no codec or provenance framework, second
mutable seed owner, policy-name field, or fixture field.

The locked state identity is a closed immutable `DMCSpec`. Production owns
exactly one literal ordered `DMC_TASKS` mapping containing each canonical ID,
load target, and camera default/bounds; `DMC20_ORDER = tuple(DMC_TASKS)` is a
derived view, not another table. `DMC_STATE_SCHEMA` is the sole state-schema
authority in `src/world_marl/dreamer_v3_baseline/dmc.py`;
runtime code never imports the test fixture. Task 6a creates and solely owns
the exact optional dm-control/MuJoCo dependency pins and lock plus these literal
tables, `DMCSpec`, and basic `DMCEnvironment`; its parity test recursively
compares these tables with the canonical fixture. Its exact fields are
`canonical_task`, `domain`, `suite_task`, `profile`, `mode`, `public_seed`,
`vector_role`, `base_seed`, `child_index`, `child_seed`, `image_size=[64,64]`,
nullable `camera_override`, `effective_camera`, `action_repeat=1`, and backend
identity `{dm_control: 1.0.17, mujoco: 3.1.3, legacy_step: true,
schema_format: world_marl.dreamer_v3.dmc_state}`. The fixture contains the total
20-task map and declares both modes/profiles, public/evaluation seed ranges and
offset, child derivation, default camera for every task, and override rule as
structured fields. The pure Task-0b matrix covers both profiles, both modes,
train/evaluation roles, public-seed endpoints, child indices `0`, `1`, and
`2**32-1`, and default/legal-override non-quadruped and quadruped cameras. Its
real state proof is explicitly narrower: all 20 tasks use only
`paper`/`proprio`/`train`, public seed 7, child index 0, and no camera override.
Camera IDs are exact Python integers in `[0,1]` for non-quadruped tasks
and `[0,3]` for quadruped; both image dimensions are exact Python integers equal
to 64. A bool never satisfies an integer field.

Task 6a declares one public production tree type rather than leaving the state
name implicit:

```python
class DMCState(TypedDict):
    compatibility: DMCCompatibilityState
    dmc_spec: DMCSpecState
    format: Literal["world_marl.dreamer_v3.dmc_state"]
    format_version: Literal[1]
    mutable: DMCMutableState
```

`DMCCompatibilityState`, `DMCSpecState`, `DMCMutableState`, and their nested
private TypedDicts close the mappings spelled out below; no field is `Any` and
no wildcard mapping is admitted. The type is the serialization tree itself,
not a class instance requiring an object decoder. Its immutability is an
ownership rule: `state_dict()` allocates every container and copies every
NumPy array; validation and restore never mutate an input or retain an input
container/array reference. The returned tree is caller-owned and later caller
mutation cannot affect the live environment. `from_state` validates first and
copies values into its candidate; no fixture path or digest is production
state.

The state record has exactly `format`, `format_version`, `dmc_spec`,
`compatibility`, and `mutable`. Compatibility is exact backend,
compiled-model hash, integration profile, action/observation specs and bounds,
`physics.legacy_step is true`, static environment/task values, sensor gate, and
the enumerated derived caches. Mutable state is the full integration vector,
complete listed model arrays, environment counters, captured task fields,
MT19937 mapping, and adapter-owned current `TimeStep`. The upstream runtime
`_step_count` is a Python integer, while its serialized form is exact
zero-dimensional `int64`; `_reset_next_step` is runtime bool and
serialized `bool[]`. Runtime `dm_env.StepType` is serialized as exact `int8[]`
in `{0,1,2}`. Reward and discount are exactly `None` or finite `float64[]`, and
observations are a closed key-exact array mapping with the declared native
dtype and numeric shape. Before construction the validator enforces the locked
reachable state machine: FIRST has count 0, reset false, and null
reward/discount; MID has `1 <= count < step_limit`, reset false, finite reward,
and discount 1; LAST has `count == step_limit`, reset true, finite reward, and
discount 1. All DMC20 tasks inherit dm-control 1.0.17's no-early-termination
implementation, so this LAST equality is part of the locked profile. MT19937
position and flags are exact `int64[]`; keys
are `uint32[624]` and cached Gaussian is finite `float64[]`. No validation path
coerces a float or bool to an integer.

`_timeout_progress` in the locked cheetah/hopper classes is reset to zero but
never read. It is deliberately omitted from behavioral checkpoint state rather
than accepting arbitrary dead state. The all-20 gate reaches a genuine
time-limit LAST through ordinary steps, restores and recaptures that current
LAST exactly, then executes and compares the following genuine FIRST reset; it
never manufactures `_reset_next_step` on a MID record.

No last-action copy is environment state. The integration `ctrl` plus the driver pending environment action are the two behavioral owners: `ctrl` is the
already-applied native control and the driver mapping is the next raw policy
sample/reset input. Clipped/scaled actions are pure call-boundary temporaries.
This shallow single-owner split avoids three redundant action leaves in every
environment checkpoint.

Physics data uses MuJoCo 3.1.3 `mjtState.mjSTATE_INTEGRATION`, validated at
numeric value `8191`, with dynamic `mj_stateSize`, `mj_getState`, and
`mj_setState`; `Physics.get_state()` is forbidden. The vector contains time,
physics state, warmstart, control, applied forces, equality activation, mocap,
userdata, and plugin state but not mutable `MjModel` arrays. Task 0b freezes the
closed canonical test schema at
`tests/fixtures/dreamer_v3/dm_control_1_0_17_state_schema.json`: 102198 bytes,
SHA-256 `55c1b76180e0a811c96efd0742ff972e61d5424f5de41d3ae54ea641b141dbd7`.
It is compact sorted-key UTF-8 JSON with one trailing newline.

These integration profiles give exact component `float64` lengths in fixed
order `time, qpos, qvel, act, qacc_warmstart, ctrl, qfrc_applied,
xfrc_applied, eq_active, mocap_pos, mocap_quat, userdata, plugin_state`.

| Profile | Exact component lengths | Total shape |
| --- | --- | --- |
| `I18` | `1,1,1,0,1,1,1,12,0,0,0,0,0` | `float64[18]` |
| `I28` | `1,2,2,0,2,1,2,18,0,0,0,0,0` | `float64[28]` |
| `I35` | `1,2,2,0,2,2,2,24,0,0,0,0,0` | `float64[35]` |
| `I37` | `1,4,4,0,4,2,4,18,0,0,0,0,0` | `float64[37]` |
| `I39` | `1,3,3,0,3,2,3,24,0,0,0,0,0` | `float64[39]` |
| `I69` | `1,7,7,0,7,4,7,36,0,0,0,0,0` | `float64[69]` |
| `I91` | `1,9,9,0,9,6,9,48,0,0,0,0,0` | `float64[91]` |
| `I226` | `1,23,22,12,22,12,22,108,4,0,0,0,0` | `float64[226]` |

The exact per-task state table is:

| Canonical task | Load target | Integration | Static task fields | Captured task fields | Complete mutable MjModel arrays | Derived caches | Sensor gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `acrobot_swingup` | `acrobot/swingup` | `I28` | `_sparse: bool=false; _visualize_reward: bool=false` | `none` | `none` | `none` | `float64[0], disabled/zero` |
| `ball_in_cup_catch` | `ball_in_cup/catch` | `I37` | `_visualize_reward: bool=false` | `none` | `none` | `none` | `float64[0], disabled/zero` |
| `cartpole_balance` | `cartpole/balance` | `I28` | `_sparse: bool=false; _swing_up: bool=false; _visualize_reward: bool=false` | `none` | `none` | `none` | `float64[0], disabled/zero` |
| `cartpole_balance_sparse` | `cartpole/balance_sparse` | `I28` | `_sparse: bool=true; _swing_up: bool=false; _visualize_reward: bool=false` | `none` | `none` | `none` | `float64[0], disabled/zero` |
| `cartpole_swingup` | `cartpole/swingup` | `I28` | `_sparse: bool=false; _swing_up: bool=true; _visualize_reward: bool=false` | `none` | `none` | `none` | `float64[0], disabled/zero` |
| `cartpole_swingup_sparse` | `cartpole/swingup_sparse` | `I28` | `_sparse: bool=true; _swing_up: bool=true; _visualize_reward: bool=false` | `none` | `none` | `none` | `float64[0], disabled/zero` |
| `cheetah_run` | `cheetah/run` | `I91` | `_visualize_reward: bool=false` | `none` | `none` | `none` | `float64[1], disabled/zero` |
| `finger_spin` | `finger/spin` | `I39` | `_visualize_reward: bool=false` | `none` | `dof_damping: float64[3] (hinge); site_rgba: float32[4,4] (target/tip alpha)` | `none` | `float64[12], disabled/zero` |
| `finger_turn_easy` | `finger/turn_easy` | `I39` | `_target_radius: float=0.07; _visualize_reward: bool=false` | `none` | `site_pos: float64[4,3] (target x/z); site_size: float64[4,3] (target radius)` | `none` | `float64[12], disabled/zero` |
| `finger_turn_hard` | `finger/turn_hard` | `I39` | `_target_radius: float=0.03; _visualize_reward: bool=false` | `none` | `site_pos: float64[4,3] (target x/z); site_size: float64[4,3] (target radius)` | `none` | `float64[12], disabled/zero` |
| `hopper_hop` | `hopper/hop` | `I69` | `_hopping: bool=true; _visualize_reward: bool=false` | `none` | `none` | `none` | `float64[3], disabled/zero` |
| `hopper_stand` | `hopper/stand` | `I69` | `_hopping: bool=false; _visualize_reward: bool=false` | `none` | `none` | `none` | `float64[3], disabled/zero` |
| `pendulum_swingup` | `pendulum/swingup` | `I18` | `_visualize_reward: bool=false` | `none` | `none` | `none` | `float64[0], disabled/zero` |
| `quadruped_run` | `quadruped/run` | `I226` | `_desired_speed: int=5; _visualize_reward: bool=false` | `none` | `none` | `_hinge_names: clear_then_lazy_rebuild; _sensor_types_to_names: clear_then_lazy_rebuild` | `float64[12], disabled/zero` |
| `quadruped_walk` | `quadruped/walk` | `I226` | `_desired_speed: float=0.5; _visualize_reward: bool=false` | `none` | `none` | `_hinge_names: clear_then_lazy_rebuild; _sensor_types_to_names: clear_then_lazy_rebuild` | `float64[12], disabled/zero` |
| `reacher_easy` | `reacher/easy` | `I35` | `_target_size: float=0.05; _visualize_reward: bool=false` | `none` | `geom_pos: float64[10,3] (target x/y); geom_size: float64[10,3] (target radius)` | `none` | `float64[0], disabled/zero` |
| `reacher_hard` | `reacher/hard` | `I35` | `_target_size: float=0.015; _visualize_reward: bool=false` | `none` | `geom_pos: float64[10,3] (target x/y); geom_size: float64[10,3] (target radius)` | `none` | `float64[0], disabled/zero` |
| `walker_run` | `walker/run` | `I91` | `_move_speed: int=8; _visualize_reward: bool=false` | `none` | `none` | `none` | `float64[1], disabled/zero` |
| `walker_stand` | `walker/stand` | `I91` | `_move_speed: int=0; _visualize_reward: bool=false` | `none` | `none` | `none` | `float64[1], disabled/zero` |
| `walker_walk` | `walker/walk` | `I91` | `_move_speed: int=1; _visualize_reward: bool=false` | `none` | `none` | `none` | `float64[1], disabled/zero` |

The complete validator checks every closed key, scalar domain, dtype, shape,
finite constraint, static value, task mapping, seed/camera identity, backend,
compiled model, sensor gate, spec, integration profile, model array, task field,
RNG leaf, counter, and TimeStep leaf before it calls the constructor or mutates
model/data/RNG/attributes. Restore order is literally:

`validate_closed_candidate -> construct_locked_task -> copy_complete_model_arrays -> mj_setState(INTEGRATION) -> mj_step1(legacy_step=True) -> restore_task_rng_and_mutable_task_fields -> restore_environment_counters_and_adapter_current_time_step -> clear_only_enumerated_derived_caches`.

The immediate single `mj_step1` is required by validated
`physics.legacy_step is true`. Corruption tests include fractional RNG position
combined with a changed integration time and prove constructor-not-called plus
byte/state-exact source preservation. The child-only executable inventory runs
under `MUJOCO_GL=off`; the pytest parent neither sets that variable nor imports
dm_control, MuJoCo, GLFW, or a renderer. Task 6's real Vision gate runs in a
separate process with a validated render-capable backend. Task 0b proves exact
mid-episode and reset continuation for all 20 state rows; Task 6 repeats those
cases for both adapter modes.

The sequential implementation ownership is closed: Task 6b adds only
`state_dict`, `from_state`, and their private validation/application helpers to
the Task-6a class; Task 6c creates the vector class and owns `state_dict` and
`from_states`. Neither later task duplicates or edits Task-6a dependency, task,
camera, or schema authorities.

The literal restore APIs are:

```text
DMCEnvironment.state_dict() -> DMCState
DMCEnvironment.from_state(state: DMCState, expected_spec: DMCSpec) -> DMCEnvironment
DMCVectorEnvironment.state_dict() -> list[DMCState]
DMCVectorEnvironment.from_states(states: list[DMCState],
                                 expected_specs: tuple[DMCSpec, ...])
    -> DMCVectorEnvironment
```

Both `from_state` classmethods are nonmutating replacement constructors. Closed candidate validation completes
before task construction. Each constructed candidate owns its resources until
the complete restore succeeds and ownership transfers to the returned result;
rejection closes the candidate. Vector construction stages all children under
one cleanup stack, so all-child atomicity means no replacement is returned and
all candidates close if any child fails. The caller swaps the validated result
into its owner first; the old environment closes only after ownership transfer.
If old close raises, the new result remains the active valid owner, every old
child is still attempted, and the aggregate close error is reported; retrying
old close is safe. No candidate failure or old-close failure partially mutates
any live environment's scientific state.

Vector `state_dict()` allocates a fresh list and fresh child mappings/arrays.
`from_states` accepts only a plain list as checkpoint state and never retains or
mutates it. The `expected_specs` tuple is a typed runtime constructor argument,
not a checkpoint leaf passed through public Flax serialization.

Both environment classes have an idempotent `close()` lifecycle. Closing one
`DMCEnvironment` releases its viewer/physics/task resources once; later closes
are no-ops, and reset/step after close fail clearly. Construction unwinds and
closes every resource acquired before a later wrapper/spec failure.
`DMCVectorEnvironment.close()` attempts every constructed child in stable order
even when one child raises, aggregates the failures, marks the vector closed,
and then propagates the aggregate error. Partial vector construction closes all
successfully created children before re-raising. `DreamerRunner` and the public
CLI own both training and evaluation environments and invoke close in `finally`
for normal completion, setup failure, collection/step failure, evaluation
failure, checkpoint failure, and interruption.

The existing `world_marl.envs.dmc_pixel_adapter` remains a separate foundation
adapter. Its public observation contract is normalized `float32` pixels in
`[0,1]`, including the singleton agent axis, for all existing world-model and
benchmark consumers. Dreamer-specific `DMCEnvironment` alone exposes direct
`uint8 [64,64,3]` Vision observations. Task 6 does not change the shared
adapter. Task 9 performs explicit conversion at comparison boundaries between
the shared arm's slash task names/`pixels` modality and Dreamer's canonical
underscore task names/`vision` modality.

Replay rows and physical actions are distinct. `replay_rows` counts every
returned row committed to replay, including a reset-generated `is_first` row,
and every such row performs one limiter insert. `control_steps` counts only
child calls with `reset=false` that actually apply a scaled control; reset rows
consume zero physical frames. `env_frames` advances by the actual native-step
count returned by the adapter, so early termination under a future repeat>1
profile is not overcounted. Replay ratio remains replayed sequence elements per
replay row.
The paper target and public `requested_target_env_steps`/`actual_env_frames`
fields use this physical control-frame budget. Terminal/reset/multi-episode tests prove that
the manifest, summary, stop checks, comparison translation, and checkpoint
counters retain these units. A real local dm_control reset/step/render run is
mandatory acceptance evidence; fakes cover boundary injection only.

## 10. Online driver, artifacts, and exact resume

`SamplesPerInsertLimiter` directly translates pinned
`embodied/core/limiters.py::SamplesPerInsert`. Immutable configuration is
`samples_per_insert`, `tolerance`, and `minsize`; derived bounds are
`min_avail = -tolerance` and
`max_avail = tolerance * samples_per_insert`. Persistent state is exactly
signed-`int64` `size` and `float64` `avail = -minsize` at construction.
`want_insert()` returns true while `size < minsize`, for a nonpositive rate, or
while `avail < max_avail`. `want_sample()` returns false while
`size < minsize`; afterward it returns true for a nonpositive rate or only when
`min_avail < avail`. `insert()` increments `size` once and adds
`samples_per_insert` only when the new `size >= minsize`; `sample()` subtracts
exactly one from `avail`.

Construction is `samples_per_insert = train_ratio / batch_length`,
`tolerance = 4 * batch_size`, and
`minsize = batch_size * replay.raw_length`, where the immutable train raw length
is `K + T * C` from the sole `SequenceShapeConfig`; report sampling separately
uses `report_raw_length = K + T_report * C_report`. Configuration,
bounds, `size`, and
`avail` serialize exactly. Restore rejects mismatched configuration or invalid
state transactionally. The native single-thread runner never blocks inside a
limiter request. It uses a deterministic cooperative scheduler while preserving
the official exactly one `insert()` transition per committed replay row and
exactly one `sample()` transition per reserved sequence.

Task 7a creates `DreamerRunner` and solely owns both `__init__` and the cold
classmethod `from_state`; Task 7b adds
`RunnerOutput` and collection/training `advance` behavior, and Task 7c adds
evaluation/cadence/run/close behavior. The exact constructor inputs are the
agent, train state, replay, training and evaluation vector environments,
limiter, run config, and sequence-shape config. It validates these
already-created owners without stepping an environment, sampling replay,
deriving a key, changing the
limiter, or opening an artifact/checkpoint owner. Initial training/evaluation
environment actions are the full declared mapping: zero model-action leaves
with leading vector axes and a true boolean `reset` leaf for every child.
Collection/evaluation carry is `agent.initial(N)` for its vector size; train and
report carry is `agent.initial(B)`, including zero previous actions. The three
call sites receive no persistent key. The scheduler initializes
`policy_call_counter=np.int64(0)` and `batch_seed_counter=np.int64(0)`.

Initial pending rows and cadence requests are empty, sample credits and every
signed-int64 event/row/control/frame/episode/window counter are zero, update
demand is false, and active cadence is null. Episode accumulators, return
windows, and metric sums/counts are empty/zero. Summary imagined transitions is
zero and its loss/evaluation values are null. Each next evaluation, report, log,
and checkpoint threshold equals its configured positive physical-frame period.
Constructor validation failure returns no runner and leaves every supplied
owner byte/state-exact.

The cold signature is
`DreamerRunner.from_state(agent, train_state, replay, train_environment,
evaluation_environment, limiter, run_config, sequence_shape, driver_state) ->
DreamerRunner`. It accepts only already-restored production owners and one
closed `DriverState`. Before allocation it checks config/space/vector-size,
replay raw-length/readiness, limiter configuration/state, train-state config,
all driver counter/carry/action shapes, cadence thresholds, and pending-row
references against those owners. It then allocates without invoking `__init__`
and performs reference assignments plus an owned copy of `driver_state`; it
does not initialize a carry/action, derive a key, query/mutate replay or limiter,
reset/step an environment, mutate an input, or retain a caller-mutable driver
container. Failure closes nothing because the staging caller still owns every
argument; Task 8c's candidate stack handles created environment resources.

The closed inverse is
`DriverState.from_state(state, agent, run_config, sequence_shape, observation_spaces, action_spaces)`.
The already-bound `agent` supplies encoder/decoder/RSSM/action carry schemas.
`run_config` supplies train/evaluation vector sizes and cadences;
`sequence_shape` supplies `B/K/T/consecutive`; the trusted spaces supply
row/action leaves; and the bound agent/train resolved identity supplies the
canonical seed when later runner calls derive roots. `RunConfig` owns no seed
or batch/sequence shape. `DriverState` has no duplicate DriverState seed leaf;
it serializes only the two call counters, while seed remains in
`DreamerV3Config.seed`. It delegates train/report carries to
`AgentCarry.from_state`, every staged report batch to
`ReplayBatch.from_state`, and closes every mapping/tag/count before returning a
fresh state. No generic dependency bag or self-described candidate dimension
is accepted.

`DriverState` is the sole owner of partial scheduler state. Its training
substate contains the train carry. Its collection substate contains the current
environment action/reset tree, collection carry, and episode accumulators. Its
scheduler subtree contains these exact fields:

- `pending_rows: tuple[PendingReplayRow, ...]`, a FIFO in stable child order;
- `pending_sample_credits: np.int64`, constrained to `[0, batch_size - 1]`;
- `pending_update_demand: bool`, true from the first reserved credit until its
  batch commits or remains blocked;
- `pending_cadence_requests: tuple[CadenceRequest, ...]`, in event-sequence
  order;
- `active_cadence: {tag: "report", value: ActiveReport} |
  {tag: "evaluation", value: ActiveEvaluation} | None`, the sole active-service
  leaf;
- scalar `np.int64 policy_call_counter` shared by collection/evaluation and
  scalar `np.int64 batch_seed_counter` shared by train/report, both constrained
  to `[0, 2**63 - 1]` and preflighted before use;
- signed-int64 `replay_rows`, `control_steps`, `env_frames`, episode, cadence,
  and output-sequence counters, plus train/report/replay aggregation windows
  with name-sorted float64 sums and signed-int64 counts;
- `summary = {imagined_transitions, last_loss,
  latest_evaluation_mean, evaluation_window_identity}`, the persistent summary
  owner described below.

Each `PendingReplayRow` stores the complete immutable row, worker index,
whether the vector input used `reset=false`, and any episode event. The report
substate owns exactly the report carry. The evaluation substate owns its
action/reset, evaluation carry, episode accumulators/counters,
and policy-return window. Neither subtree embeds
an environment state: train and evaluation vector environments are separate
checkpoint leaves. Only the two scheduler counters derive model-call keys;
replay RNG remains in `UniformSelector`, and task RNGs remain in those
environment leaves. All fields above serialize at every completed scheduler
quantum; checkpointing does not flush or discard a partial batch, FIFO, cadence
request, active service, or window.

An active cadence contains no carry, counter, action, or aggregation window.
The named report/evaluation subtrees are the sole owners of those mutable
values; the active union holds only request, phase, immutable staged input or
copied output, and bounded emit progress. `DreamerTrainState.update_count` is
the sole update counter. Task 7 perturbs each candidate duplicate and performs
a mid-active restore to prove no second owner can affect continuation.

`driver.summary` owns signed-int64 `imagined_transitions` starting at zero,
nullable finite `last_loss`, and the latest completed evaluation mean as a
nullable finite float paired with its exact evaluation window identity. An
atomic train commit adds `B*Kstart*H` and replaces
`last_loss`; an atomic evaluation commit replaces the mean and window identity.
A log-window reset does not change any persistent summary field. The literal
checkpoint includes `driver.summary`. `train_updates` derives directly from
`DreamerTrainState.update_count`, never from a driver mirror.

All cadence periods are positive physical `env_frames` intervals. The first
threshold is one period, crossings use `old_env_frames < threshold <=
new_env_frames`, and reset-only quanta add zero and queue nothing. Every crossed
threshold is queued, including catch-up across multiple thresholds. Requests
are ordered by ascending `threshold_env_frames`, then fixed `CadenceKind` order
`evaluation`, `report`, `log`, `checkpoint`; `event_sequence` is the next
signed-int64 sequence in that total order. A `CadenceRequest` contains exactly
`kind`, `threshold_env_frames`, `observed_env_frames` (the actual counter after
the crossing quantum), and `event_sequence`. Each kind maintains one serialized
next threshold counter, including the log next threshold. A completed log
request appends one exact metrics row from `metric_means` and `metric_counts`,
flushes and resets that aggregation window, then advances the next threshold.
There is no wall-clock cadence.

`DreamerReplay.can_sample_batch(mode)` is a nonmutating readiness query. It
proves that all `batch_size` items required by `mode` are presently sampleable
and is called before limiter availability, replay RNG, selector/stream state,
or pending credits can change. Readiness is distinct for training and report
streams; a limiter credit never stands in for an absent replay item.

`DreamerRunner.advance() -> RunnerOutput` performs at most one of the following
atomic quanta using this stable action priority:

1. If the FIFO head exists and `want_insert()` is true,
   `DreamerReplay.prepare_add` builds a bounded immutable mutation plan covering
   only the current/open chunk, identity allocation, bounded suffix, selector/
   queue changes, and at most one eviction. It must stage and prevalidate every
   allocation/collision/counter before limiter mutation. The commit section
   performs exactly one `insert()` and then the plan's nonallocating/no-fail
   `commit_add`, followed by FIFO/`replay_rows`/episode aggregates. It never
   copies or swaps the full replay and remains bounded independently of replay
   lifetime/capacity. A preparation failure changes nothing. Physical counters
   were already committed by collection and are not incremented again.
2. Otherwise, if an update is demanded,
   `can_sample_batch("train")` is true, and `want_sample()` is true, reserve one
   sequence by exactly one `sample()`. Before the first credit, prevalidate the
   eventual train/counter capacity and replay-batch availability. Intermediate
   credits atomically update only limiter availability and
   `pending_sample_credits`. For the final credit, bounded
   `prepare_sample`/context-writeback plans and functional
   `train_step(agent, state, carry, batch, outer_seed)` outputs are computed
   from the current batch counter without live replay mutation; the commit invokes one
   `sample()`, the plans' no-fail local commits, swaps the functional train
   state, increments `batch_seed_counter`, aggregates metrics/cadences, and
   resets credits/demand. A diagnostic failure leaves live state and the batch
   counter at the preceding credit boundary.
3. Otherwise, if no update is active, `can_sample_batch("train")` is true, and
   `want_sample()` is true, set
   `pending_update_demand=true` and reserve its first credit by rule 2. This is
   repeated after each completed batch, so the scheduler must drain all
   currently owed full updates earned by already-produced rows before cadence
   service or optional collection.
4. Otherwise, if `active_cadence` exists, advance it by one bounded service
   quantum. An evaluation quantum is at most one full evaluation-vector
   reset/step call; a report quantum is exactly one of stage-report-batch,
   compute-functional-report, or emit-copied-report. Collection and training
   pause until the active report/evaluation completes, so parameters remain
   fixed throughout it. The active phase/request/staged immutable input or
   copied output serializes after every phase; mutable carry, counters, action, and
   windows remain only in their named driver subtrees. Successful report
   computation atomically advances the report carry and shared batch counter
   and stages copied output; sampling or computation failure advances neither.
   Each successful evaluation policy phase similarly increments the shared
   policy counter with its carry/action commit. A writer failure may leave only uncheckpointed in-memory
   advancement, terminates without another checkpoint, and cold resume replays
   the request. Evaluation never owns or mutates replay, model,
   optimizer, limiter, or the training report stream.
5. Otherwise, if `pending_cadence_requests` is nonempty, inspect its head. A
   ready report/evaluation is popped, activated, and advanced; log/checkpoint
   emits its copied request. A report whose `can_sample_batch("report")` is
   false is deferred unchanged while physical budget remains. After a bound is
   reached, it deterministically emits `skipped-unavailable` without advancing
   its counter, carry, replay stream, limiter, or selector, so a due report cannot
   pause forever. This priority makes every serviceable active cadence and due
   cadence run before optional new collection.
6. Otherwise, if training demand exists but readiness is false, or a partial
   sample batch is blocked, and the physical lower-bound
   budget has not yet been reached, collect one full vector quantum to obtain
   inserts needed for progress.
7. Otherwise, if `pending_rows` is empty and the physical lower-bound budget
   has not yet been reached, perform one optional full-vector collection
   quantum. Capacity is checked against the known reset mask before the call.
   It derives the collection policy key from the current shared policy counter.
   The returned rows, post-policy carry/action state, incremented policy
   counter, physical counters, every newly crossed cadence request, and FIFO
   commit together. `control_steps`
   increases by the count of `reset=false` children and `env_frames` by the sum
   of actual returned `native_steps`; a reset child contributes one replay row
   and zero physical frames.
8. Otherwise return a copied no-op/safe-stop output.

Thus already-produced rows and all full updates they earned commit first,
every active cadence and due cadence is serviced next, and only then is
optional collection eligible. If both an insert and a sample are eligible,
FIFO insertion wins; if insertion
is backpressured, a sample credit is taken; if a partial batch is sample-
backpressured with no pending row, collection is allowed only to create rows
needed for progress. At `samples_per_insert=4`, `tolerance=64`, and
`size >= minsize, avail = 252`, one insert reaches the max boundary and the next
quantum samples rather than stalling the remaining FIFO. Starting a 16-credit
batch at `avail = -63` reserves one credit, serializes the partial state at
`-64`, then inserts or collects before resuming the same batch without a double
decrement. The four-update trace starts exactly at
`size >= minsize, avail = -64`: 16 inserts advance availability by four each to
zero, and four 16-credit batches decrement it back to `-64`. Tests assert all
16 intermediate insert values, all 64 credit decrements, exactly four committed
updates—four complete updates—and no fifth batch. Tests
execute both pressure states, multiple-update draining, stable FIFO/online-
queue/sample order, and uninterrupted-versus-resumed equality from every
partial scheduler state.

The small readiness RED has two writers with `B=1,T=4,R=4`: the first child
cannot make either stream sampleable, the second completes the first item, and
the training request falls through to collection rather than consuming a
limiter credit early. It covers the zero-credit/no-item state and a report
period below `R`; the report remains deferred while collection can make it
ready, then runs, or becomes `skipped-unavailable` at the bound without
advancing its batch counter. These cases prove that two writers, distinct train/report
streams, and cadence priority cannot deadlock.

The canonical liveness RED uses target frame 48 with all cadence periods 16:
after the collection quantum that reaches frame 16 and its rows/earned updates
drain, evaluation, report, log, and checkpoint are serviced at observed frame
16 before any collection toward frame 48. It performs a mid-evaluation restore
and a mid-report restore, respectively during `ActiveEvaluation` and
`ActiveReport`; subsequent actions, keys,
carries, artifacts, event sequences, and thresholds equal the uninterrupted
trace. Checkpoint, report, and evaluation are not deferred to the final target.

Requested debug stop or final target is checked at every quantum boundary. Both
are lower bounds because the synchronous vector API has no subset/no-op child
step. Collection stops after the first full-vector quantum whose actual
native-frame result reaches or crosses the bound. For shipped repeat-1 runs the
bounded overshoot is in `[0,N-1]`; reset-only quanta may add zero and continue.
The run records `requested_target_env_steps`, optional requested stop,
`actual_env_frames`, and `overshoot_env_frames = actual - reached_bound`.
Every cadence threshold crossed by the final quantum is queued and serviced.
Tests cover `N=2,target=3`, mixed reset/control masks, debug stop, cadence
crossing, and resumed-run equality. Comparison receives the actual measured
budget and must not claim exact requested equality.
The runner first commits all rows already produced, drains every full batch that
can complete without another physical action, and completes all active/due
cadence results.
If the remaining batch is partial and blocked with no physical budget, that
partial credit state is a valid final checkpoint state; it is never discarded
or completed with an atomic credit shortcut. Thus `advance()` is bounded and
checkpointable rather than an uninterruptible collection turn.

Collection preflight is limited to the actual next vector call and its owners:
environment/control counters, policy state, FIFO capacity, episode aggregates,
and cadences. Training preflight is limited to the one pending update. Each
owner performs its owner-local wraparound preflight at the actual mutation
boundary; no global multi-owner capacity protocol is introduced. Collection and learning remain interleaved;
a fixed precollection followed by offline reuse is forbidden. `RunnerOutput`
owns copied values and cannot alias runner state.

After the nonmutating collision check accepts a fresh empty output directory,
`RunManifest` is written before model/environment construction and records the
resolved profile, complete canonical config and hash, pinned
`authority_revision`, task/environment contract, seed, and CLI overrides. It
contains no authority-source map, implementation revision, implementation
source map, or live-code digest.

The manifest `public_seed`, canonical argv `--seed`, summary `seed`, native
train/evaluation `DMCSpec` public/base/child identities, and checkpoint identity
seed are derived projections and must each equal `resolved.config.seed` where
the schema names the public seed. Parameter initialization and every official
counter root read that same field. Replay manifest/checkpoint state contains no
construction-seed identity or public-seed equality check; its complete selector
PCG64 state is operational state. A mismatch in a true public-seed projection
is rejected before the first model, replay, or environment construction.

Canonical JSON uses compact separators, UTF-8, one terminal newline,
`allow_nan=false`, and the literal key order declared here. The manifest schema
is exactly:

```text
schema_version: int = 1
kind: string = "dreamer_v3_run"
run_id: string (32 lowercase hexadecimal UUID4, or explicitly injected test ID)
initial_checkpoint_generation: null
model: string = "dreamer_v3"
profile: string
observation_mode: string
task: string
public_seed: int
authority_revision: string
canonical_config: object
config_sha256: string (64 lowercase hexadecimal digits)
runtime_overrides: object
debug_snapshot: object or null
canonical_argv: list[string]
environment_backend: string = "dm_control"
backend_version: string = "1.0.17"
train_dmc_spec: object
evaluation_dmc_spec: object
```

There are no timestamps or omitted/extra keys. The
`initial_checkpoint_generation` field is deliberately null because the
manifest is immutable and precedes the first checkpoint; current generation is
checkpoint/pointer state, not manifest state.
`ArtifactWriter` follows direct repository-style output conventions. It owns
append-only `metrics.jsonl` and `scores.jsonl`, atomic `manifest.json` and
`summary.json`, committed writer byte offsets, and next-file cursors for
`open_loop/{cursor:020d}.npz` and `evaluation/{cursor:020d}.json`. Vision
open-loop cursor/files are nonempty after every due successful report; each
file contains nonempty arrays `posterior_prefix`, `prior_suffix`, `target`, and
`error` with the report axes. Proprio open-loop cursor is zero for the entire
run, Proprio open-loop directory is absent, and restore/inspection rejects any
numbered Proprio open-loop file. The evaluation cursor/files are nonempty in
both modalities after a completed evaluation. Evaluation files contain exactly `task`,
`observation_mode`, `episode`, `env_frames`, `return`, and `length`; numeric
values are finite. `length` is the episode's sum of actual native control steps
from the first non-reset action through the final transition; reset emissions
are zero and excluded. `env_frames` is the training physical-frame counter at
the evaluation cadence request, not evaluation-environment frames. There are no companion identity files or file-level
provenance records.

Each metrics append trigger is exactly a completed `log` cadence request. Its
exact JSONL row, in canonical key order, is:

```text
schema_version: int = 1
row_type: string = "train_metrics"
run_id: string
event_sequence: signed int64
cadence_kind: string = "log"
threshold_env_frames: signed int64
observed_env_frames: signed int64
env_frames: signed int64
train_updates: signed int64
window_id: signed int64
window_start_env_frames: signed int64
window_end_env_frames: signed int64
metric_means: object[string, finite float64] (keys sorted)
metric_counts: object[string, positive signed int64] (same keys, sorted)
```

Empty metric maps are allowed before any update. A successful append is
flushed before the aggregation window resets and its ID advances. Each scores
append trigger is exactly one completed evaluation cadence, after its numbered
episode files are durable. Its exact JSONL row, in canonical key order, is:

```text
schema_version: int = 1
row_type: string = "evaluation_scores"
run_id: string
event_sequence: signed int64
cadence_kind: string = "evaluation"
threshold_env_frames: signed int64
observed_env_frames: signed int64
env_frames: signed int64
evaluation_window_id: signed int64
episodes_requested: signed int64
episodes_completed: signed int64
evaluation_file_start: signed int64
evaluation_file_stop: signed int64 (exclusive)
mean_return: finite float64
mean_length: finite float64
```

`episodes_completed == episodes_requested` and
`evaluation_file_stop - evaluation_file_start == episodes_completed`; the mean
and window identity equal `driver.summary`. Tests compare exact JSONL rows,
canonical key order, append trigger counts, and resumed append grouping after a
checkpoint at every aggregation boundary.

Each numbered file is published by writing a temporary sibling, flushing and
`fsync`ing it, atomically renaming it to the next final cursor, and `fsync`ing
the parent directory before advancing that cursor. Manifest and summary use the
same temp write, file `fsync`, atomic rename, and parent-directory `fsync`
sequence. JSONL append state stores only each writer byte offset; a flush makes
the current prefix eligible for a checkpoint. `ArtifactWriter.state_dict()` is
exactly the two offsets and two next-file cursors plus immutable run identity;
file handles and lifecycle never serialize.
`ArtifactWriter.resume(run_dir, writer_state) -> ArtifactWriter` validates run
identity, exact nonnegative offsets/cursors, file lengths, numbered names, and
temporary siblings before any append handle opens. It then performs one direct,
idempotent order: truncate and `fsync` each
JSONL file to its checkpointed byte prefix; unlink numbered files at or above
the checkpointed next cursor and their temporary siblings; `fsync` each changed
directory; then open metrics and scores at those exact offsets. Files below a
cursor are immutable and must already exist with valid schemas. There are no
tail archives, auxiliary directories, generation labels, or multi-state writer
protocol. Failure closes any newly opened handle and propagates; cold resume
revalidates and repeats the same reconciliation safely.

The writer state always contains both cursor scalars. Its trusted manifest mode
closes their interpretation: Vision accepts a positive open-loop cursor only
with every lower numbered file present and valid, while Proprio requires cursor
zero and an absent directory. Evaluation uses the same numbered-prefix rule in
both modes. The writer creates the open-loop directory lazily on the first
Vision video and never creates it for an empty report output.

`RunStatus` is the exact string enum `running`, `interrupted`, `completed`, and
`failed`. `RunSummary` is derived and atomically replaced, not append authority.
Its exact Task-9 schema is `schema_version`, `model`, `profile`, `task`,
`observation_mode`, `seed`, `status`, `environment_backend`,
`config_sha256`, nullable `debug_snapshot`, the exact closed
`runtime_overrides` mapping,
`requested_target_env_steps`, nullable `requested_stop_env_steps`,
`actual_env_frames`, `overshoot_env_frames`,
`train_updates`, `imagined_transitions`, nullable `evaluation_return`, nullable
`last_loss`, `learning_gate_passed`, and `run_id`. `evaluation_return` is the
arithmetic mean over exactly the most recently completed evaluation cadence's
configured episode window and is null before one completes. `last_loss` is the
last committed loss: the finite total scalar from the last committed train
update, null before the
first update. `imagined_transitions` is an accumulated measured counter: each
committed update adds exactly `B*Kstart*H`, where
`Kstart = min(imag_last if imag_last != 0 else T, T)` after context trimming.
This is the official Agent start axis: a nonzero `imag_last` selects that many
latest starts rather than subtracting them. The persistent summary counter uses
the actual batch/start/horizon shapes returned by the committed loss. The learning gate
formula is exactly `train_updates > 0 and last_loss is not None and
imagined_transitions > 0`; status does not override it. On completion,
`overshoot_env_frames = actual_env_frames - requested_target_env_steps`. For a
stop reached by the current CLI invocation it equals
`actual_env_frames - requested_stop_env_steps` and the stop is nonnull only in
that invocation's summary.
Overshoot is derived independently of `RunStatus`: `failed` after crossing a
target or stop retains the same nonnegative overshoot. It is zero only before
any bound has been reached. Tests cover a post-crossing artifact failure and a
post-crossing checkpoint failure.
Requested budgets and actual measured budgets are distinct;
no counter/return is inferred from a request. Task 9 owns spelling translation.

Comparison and benchmark dispatch distinguish canonical paper arms from local
debug evidence. The public/default primary matrix never selects
`debug-local-v1` and never forwards `--debug-local`; one explicit
comparison-level `--dreamer-debug-local` opt-in may forward that child flag only
for local smoke execution. Strict normalization always copies and validates
`config_sha256`, nullable `debug_snapshot`, and `runtime_overrides`. Its default
primary mode rejects any nonnull debug snapshot. The explicit debug path labels
rows `comparison_role="debug_local"`; primary tables and parity gates exclude
those rows even if every other identity field matches. A debug artifact is
therefore never described as, counted as, or substituted for a canonical paper
arm.

Comparison camera input is nullable with parser default `None`. If omitted, no
`--dmc-camera-id` child argument is emitted and `DMCSpec` derives effective
camera 2 for quadruped tasks and 0 otherwise. An explicitly supplied camera,
including integer 0 for a quadruped task, is forwarded and preserved as the
camera override and effective identity.

`requested_stop_env_steps` is deliberately invocation-local CLI control, not
durable scientific state. It is absent from canonical config, manifest,
checkpoint identity/payload, `DriverState`, and every owner state. At the start
of every fresh or resumed invocation, the CLI immediately derives and replaces
the summary from checkpointed measured counters plus that invocation's target
and optional stop. The field remains null unless this invocation reaches its
own stop; if the resume command omits a stop, it is null even when an earlier
invocation stopped at that frame. Overshoot uses the current invocation's stop
only after that stop is reached, otherwise the immutable target after the
target is reached, and is zero before either. Thus an interrupted stop at frame
32 may report stop 32, but resume toward target 48 without a stop immediately
rewrites it to null/zero and the final summary is logically and bytewise equal
to an uninterrupted target-48 summary. No hidden stop-history leaf is
reconstructed or checkpointed.

`CheckpointPayload` has the following literal tree; a recursive schema test
enumerates it and rejects duplicate or conflicting leaves:

```text
schema_version
checkpoint_generation
identity = {run_id, canonical_config, config_sha256, authority_revision,
            runtime_overrides, debug_snapshot, train_dmc_spec,
            evaluation_dmc_spec}
train_state
replay
limiter
driver = {
  scheduler,                         # pending rows/credits/demand/cadences,
                                     # minimal active service, two call counters,
                                     # measured counters, windows
  summary = {imagined_transitions, last_loss,
             latest_evaluation_mean, evaluation_window_identity},
  collection = {action, collection_carry, episode_accumulators},
  train = {train_carry},
  report = {report_carry},
  evaluation = {action, evaluation_carry,
                episode_accumulators, return_window},
}
train_environment
evaluation_environment
artifact_writer
```

Each leaf appears exactly once. In particular, `DriverState` never contains an
environment snapshot; train/eval vectors occur only in their named payload
leaves. Collection, train, report, and evaluation carries occur only under
their named driver owner. The driver scheduler alone owns the policy-call and
batch-seed counters. `train_state` alone owns optimizer, slow-value,
normalizer, and update counter. `replay` alone owns selectors,
streams, metrics, and both identity cursors. Unknown, missing, duplicated, or
conflicting logical owner paths are rejected before candidate construction.
The active cadence contains no carry, counter, action, or aggregation window; named
report/evaluation subtrees are the sole owners. Recursive tests perturb each
candidate duplicate, reject it, and compare uninterrupted with mid-active
restore.

Checkpoint bytes use a short non-executable wrapper around the public Flax 0.10.4
`flax.serialization.msgpack_serialize` and
`flax.serialization.msgpack_restore` functions. Flax is already a direct
runtime dependency and its transitive MessagePack/array support is sufficient;
Task 8b owns no dependency or lock edit. Pickle, object hooks, dynamic imports,
user callbacks, and input-selected constructors are forbidden.

A file is exactly the following envelope and nothing else:

```text
magic        8 bytes   ASCII WMDRM3CK
version      1 byte    0x01
body_length  8 bytes   unsigned big-endian
body_sha256 32 bytes   SHA-256 of body
body         body_length bytes from flax.serialization.msgpack_serialize
EOF
```

Decode checks a regular file, the 49-byte fixed envelope, the owner-derived
maximum total size, magic/version, exact `49 + body_length` file size, and body
digest before calling Flax. A truncated body, length mismatch, checksum mismatch,
second object, or any trailing byte is rejected. The codec registers no
extension hook of its own. Flax's documented array extension handles NumPy/JAX
numeric arrays and bfloat16; an unknown extension restores as an unsupported
leaf or raises and is rejected by the closed owner schema.

The input is not a general value language. Before encoding,
`CheckpointPayload.validate(owner_schemas)` requires the literal payload tree
above and every owner-local key/type/shape/dtype/range invariant. Every owner
has already produced the primitive record specified in section 3.2; Task 8b
does not inspect or convert runtime dataclasses. It explicitly rejects tuple,
Python dataclass, Flax struct dataclass, `FrozenDict`, path, enum, and arbitrary
object leaves. It then builds a fresh canonical tree whose structural
string-keyed mappings are recursively inserted in UTF-8 byte order and whose
plain lists retain their schema-declared order. Task 2b has already converted replay's two integer-key runtime maps to
ordered string-keyed record lists; its bytes-key `refs` map is the only
non-string mapping and public Flax canonicalizes it by byte key. Decode
runs `msgpack_restore(body)`, rejects every unknown/missing/extra leaf through
the same closed schemas, applies the four path-specific integer adapters below,
and canonicalizes again. It finally requires
`msgpack_serialize(canonical_adapted_tree) == body`; this rejects duplicate or
out-of-order raw map keys, noncanonical encodings, unknown extensions, and
schema aliases without implementing a second MessagePack grammar.

Only integers at these exact paths need adaptation before Flax serialization:

| Payload path | Domain | Encoded leaf |
| --- | --- | --- |
| `replay.next_chunk_id` | `[1, 2**128]` | exactly 17 unsigned big-endian bytes |
| `replay.next_item_id` | `[0, 2**63]` | exactly 8 unsigned big-endian bytes |
| `replay.selector.rng_state.state.state` | `[0, 2**128)` | exactly 16 unsigned big-endian bytes |
| `replay.selector.rng_state.state.inc` | `[0, 2**128)` | exactly 16 unsigned big-endian bytes |

Decode accepts those byte leaves only at those four paths, checks exact width
and domain, and reconstructs Python integers before replay validation. No
logical tag registry or arbitrary-precision adapter exists. All other
checkpoint integers use their already-declared fixed NumPy/JAX scalar arrays or
MessagePack-supported Python range.

The decoder's file limit comes from the resolved closed owner schemas rather
than a global multi-gigabyte constant. `CheckpointPayload.max_body_bytes` is
computed once from: exact train/optimizer/carry/environment array shapes; the
configured replay capacity, chunk size, spaces, and maximum writer/stream/queue
state; vector sizes; closed driver/artifact fields; exact identity/config string
bytes; and the four fixed-width leaves above. The formula is
`expected_leaf_bytes + expected_utf8_bytes + 64 * closed_node_count +
1_048_576`. The one-MiB constant is the only codec overhead allowance; every
other term is mechanically derived from the resolved owner schemas and replay
capacity. Encode asserts its body fits this bound. Decode rejects a larger file
before reading the body and then validates actual node counts, replay
cardinality, shapes, and leaf byte lengths against those same owner schemas
before resource construction.

Task-8b tests first assemble one representative complete payload from every
real owner record and pass it directly through public Flax serialize/restore;
they separately prove tuple, both dataclass kinds, `FrozenDict`, and an
unsupported object are rejected before envelope work. They then exercise deterministic equal bytes in two fresh processes,
envelope length/hash/trailing rejection, malformed and unknown extension/schema
rejection, fresh-process bfloat16 NumPy/JAX roundtrip, each of the four wide
integer boundary roundtrips, canonical-key re-encoding, and a body one byte over
the owner-derived bound. They also prove exact uninterrupted/resumed behavior;
they do not maintain a tag golden table, Unicode-normalization policy, generic
container grammar, or independent resource-attack framework.
Fresh-run `DreamerRunCoordinator` directly composes `(runner, artifact_writer,
checkpoint_manager)` and owns no serializable state. `consume_output(output)`
writes a copied output exactly once; `save_safe_point()` flushes/fsyncs and
publishes the literal payload; `advance()` performs one runner quantum, consumes
it, then handles its cadence. For a checkpoint request it captures the literal
tree without a generation and calls
`CheckpointManager.save(snapshot_without_generation) -> (CheckpointPayload,
Path)`; no checkpoint publishes after runner, diagnostic,
writer, or manager failure.

`checkpoints/latest` is one regular UTF-8 file containing exactly
`{generation:020d}.ckpt\n`; numbered names use the same 20-digit `.ckpt`
format. In the supported single-writer contract, the next generation is zero
when `latest` is absent and otherwise exactly the validated latest generation
plus one. Save injects that value into the payload, encodes deterministic
checkpoint bytes, writes/flushes/`fsync`s a temporary sibling, atomically
replaces the numbered file, `fsync`s the directory, then atomically publishes
and directory-`fsync`s `latest` with the same temporary-sibling sequence. A
crash before `latest` publication leaves an unauthoritative numbered file; the
next save deterministically replaces that same successor rather than scanning,
adopting, preserving, or allocating around it. A referenced missing or malformed
file is fatal. Restore accepts only the path named by `latest` (or that exact
resolved path supplied by the CLI) and validates codec, identity, and every
owner before resource construction.

Representative publication tests inject once before numbered publication and
once after the numbered rename but before `latest`; neither advances the
authoritative pointer. A successful retry publishes one successor. Artifact
resume tests likewise inject once before direct reconciliation and once after
truncation but before handle open; an idempotent cold resume produces the exact
checkpointed prefixes and cursors. This covers the requested durability without
a generalized filesystem transaction framework.

`inspect_run_state(run_dir)` directly parses manifest/summary, both JSONL logs,
the numbered open-loop/evaluation files below active cursors, writer offsets,
counters, and checkpoint state. It validates their schemas and returns logical
contents and cursor positions; it does not build a source/provenance graph or
authenticate the whole run tree. Starting without `--resume` in a directory
containing committed state exits before mutation.

Resume is cold bootstrap, never mutation of an already-live runner. Its only
entry is the closed classmethod
`DreamerRunCoordinator.resume(checkpoint_path, expected_identity,
resolved_config, observation_spaces, action_spaces, run_dir)`. There is no
factory/callback mapping. `expected_identity` is the sole identity input and
contains canonical config/hash, profile/authority, run id, runtime
override/debug snapshot, both complete DMC specs including cameras/backend, and
space signatures. Its canonical config includes the checked `seed`. The other
arguments are the already-resolved production values against which those fields
are checked. Before any owner or resource construction, the classmethod checks
that the canonical config/hash, `expected_identity` seed, both DMC-spec seed
mappings, and `resolved_config.seed` are equal; a seed mismatch fails before
construction. The classmethod then performs this literal order:

1. construct `DreamerAgent(observation_spaces, action_spaces,
   resolved_config)`, which owns no external resource and creates no live train
   state;
2. call `DreamerTrainState.schema(agent, observation_spaces, action_spaces,
   resolved_config)` and derive the remaining closed owner schemas and
   owner-derived byte bound from resolved config, spaces, replay capacity, and
   vector specs; construct the `CheckpointManager`; then call
   `restore_candidate(checkpoint_path, expected_identity, owner_schemas)` to
   decode and validate every payload leaf without resources or run-tree writes;
3. call `RunnerRestoreCandidate.stage(payload, agent, resolved_config,
   observation_spaces, action_spaces)`. It acquires
   `DMCVectorEnvironment.from_states` for training and then evaluation under
   one cleanup stack, then calls the named
   `DreamerTrainState.from_state(state, agent, observation_spaces,
   action_spaces, resolved_config)`, `DreamerReplay.from_state_dict`,
   `SamplesPerInsertLimiter.from_state_dict`, and
   `DriverState.from_state(state, agent, run_config, sequence_shape,
   observation_spaces, action_spaces)` validators in that order, and finally
   calls the exact
   `DreamerRunner.from_state` API above. No candidate input is mutated;
4. call `ArtifactWriter.resume(run_dir, payload.artifact_writer)` to validate,
   directly reconcile files to checkpointed offsets/cursors, and only then open
   append handles;
5. construct the coordinator by no-fail assignments, transfer the candidate
   runner exactly once, and disarm its environment cleanup stack.

`RunnerRestoreCandidate.stage` has no constructor registry or undeclared
dependency record. Its named arguments and production classmethods above are
the complete construction schema. It calls `DreamerTrainState.initialize` zero
times; a cold resume cannot execute fresh parameter/train-path initialization.
The cleanup stack registers training then
evaluation vector closes, so rejection closes evaluation then training and each
vector attempts every child. The runner merely refers to those staged vectors
until transfer; after transfer it is their sole lifecycle owner.
`RunnerRestoreCandidate` is immutable apart from its candidate ownership flag and closes
both vector environments and every staged resource in reverse acquisition order
on rejection. Identity/schema validation or candidate-construction failure
leaves the run tree untouched. Artifact failure closes candidate resources and
any opened writer handles; direct reconciliation is idempotent and is retried
by cold resume from the same checkpoint. There is no fallible algorithm
swap after artifact reconciliation: final coordinator construction only assigns
already-validated references. The contradictory instance
`restore(checkpoint_path)` on a live coordinator does not exist. Boundary tests
inject decode/identity failure, each environment/resource construction failure,
artifact prepare/apply failure, and verify cleanup and no live-owner mutation.
Live runtime-code hashes are outside identity and remain a version-control/test
concern.

Resume identity is tested by constructing uninterrupted and resumed branches
with one explicitly injected deterministic `run_id` and one shared canonical manifest payload
before either executes. The manifest is published independently
in each fresh output tree, and tests compare its bytes and `run_id` directly;
no comparator deletes, rewrites, or normalizes identity fields. Production mints
a fresh run id when none is injected.

The two branches compare every subsequent action, replay sample, loss, parameter,
optimizer/normalizer/slow state, queue/writer/stream, replay identity cursor,
limiter, partial scheduler state, carry, both outer-call counters, replay/task
generator state, artifact, evaluation,
and final serialized state. Both use the same
checkpoint cadence, including a checkpoint at the interruption safe point, so
generation, writer offsets, and next-file cursors match; process restart itself
does not create an artifact record. Raw checkpoint containers are independently
decoded and compared by payload tree rather than by filesystem timestamps.
Parameter-only warm start does not satisfy resume. The policy and batch
counters advance only at their section-2.4 commit boundaries; replay selection
and every train/evaluation environment task generator advance only at their
separate owners. A checkpoint
taken mid-window for train, report, replay, and episode aggregation must yield
byte-identical subsequent metrics/scores grouping and bytes. Crash-recovery tests
also prove active files equal the checkpointed prefix and numbered files begin
exactly below the checkpointed next cursors.

## 11. Runtime and acceptance invariants

- `paper` is the CLI and API default; current behavior requires an explicit
  profile flag.
- Runtime imports never load oracle/generator modules.
- Replay row action alignment and previous-action construction are tested at
  episode start, ordinary transition, terminal, truncation, and auto-reset.
- Every valid replay posterior state starts imagination.
- Every loss has value, reduction, scale, and gradient-boundary tests against
  pinned official behavior; a composed encoder-RSSM-decoder-agent fixture is
  mandatory.
- Row insertion is bounded local work; exhaustive replay validation is opt-in.
- `ac_grads`, `reward_grad`, `repval_grad`, `repval_loss`, `slowtar`, and
  continuation-discount switches each have positive and negative gradient tests.
- Package import, CLI help, dry run, local online DMC collection/update,
  artifacts, checkpoint/resume, and evaluation must exercise production paths.
- Ruff check/format, focused parity tests, full pytest, and two independent
  whole-branch reviews must finish with zero Critical and Important findings.
- Synthetic environments and mocked comparison commands may supplement tests
  but cannot stand in for real local DMC acceptance.
- Linux GPU throughput and full 20-task scientific runs in both modes remain explicitly
  pending unless separately authorized; no local result is described as that
  evidence.

Forbidden substitutions include argmax categorical latents, dense GRU, linear
symlog bins, global reconstruction MSE, pathwise actor training, final-state-only
imagination, Adam, `is_last` continuation targets, offline static batches,
parameter-only checkpoints, mocked-only comparison arms, and shape/finite-only
claims of parity.
