# Jafar/Jasmine World-Model Replacement Plan

## Goal

Replace the previous experimental visual-world-model arm completely with two independent,
source-derived pixel world-model arms:

- `jafar`: VQ-VAE tokenizer, discrete VQ LAM, MaskGIT dynamics.
- `jasmine`: continuous MAE tokenizer, discrete VQ LAM, diffusion-forcing
  dynamics.

Shared replay, reward/continuation, bridge, simulator, PPO, artifact, and
evaluation behavior belongs to `world_marl.latent_action_world_model`. There
will be no compatibility alias and no unsupported third-party implementation claim.

## Pinned sources

- Jafar: `https://github.com/FLAIROx/jafar` at
  `5ff9fc7d5d744c8c2797ba3ad0a095ed7f2e2665`.
- Jasmine: `https://github.com/p-doom/jasmine` at
  `420859bc99eecf6b07a7e9edf65d5d145935f1e1`.
- Both source repositories are Apache-2.0 licensed.

The pinned model, training, schedule, and sampling source files are the source
of truth. Each adapted module records repository, path, commit, and integration
changes in its module docstring.

## Global constraints

- Work only on branch `jafar-jasmine-world-models` in this worktree.
- Keep JAX 0.4.36 and Flax 0.10.4.
- Add only `dm-pix>=0.4.3` and `einops>=0.8.0` with `uv add`; never edit the
  dependency declaration by hand.
- Do not add Torch, Procgen, Grain, or ArrayRecord.
- Do not add `tests/test_packaging.py`.
- Runtime training, sampling, simulation, PPO minibatch/epoch, and JAX-native
  evaluation loops use `jax.lax.scan` with no Python-loop fallback.
- Avoid per-step host synchronization; log only at configured or phase
  boundaries.
- Use `world-marl-runpod`; do not create a pod directly.
- Do not push. Merge locally to main only after every local and GPU gate passes.

## Implementation sequence

### 1. Documentation and attribution

Create and commit this plan, the two architecture documents, the two source
conformance maps, `CHANGES.md`, and `THIRD_PARTY_NOTICES.md` before any code,
test, or dependency edit. Verify the files contain both pinned commits and no
unsupported third-party implementation claims.

### 2. Dependency transaction

Run:

```bash
uv add "dm-pix>=0.4.3" "einops>=0.8.0"
```

Verify `uv.lock`, JAX 0.4.36, and Flax 0.10.4 after the transaction. Commit the
generated dependency changes separately.

### 3. Shared transformer and preprocessing primitives

Create source-derived, arm-local primitives rather than a compatibility
package:

- `src/world_marl/jafar/preprocess.py`
- `src/world_marl/jafar/nn.py`
- `src/world_marl/jasmine/preprocess.py`
- `src/world_marl/jasmine/nn.py`

Add focused failing tests for patch layouts, crop behavior, spatial
non-causality, temporal causality, positional encoding, rematerialization,
initializers, parameter dtypes, full-precision norms, bf16 compute, cuDNN
selection, cosine VQ, and straight-through gradients. Run each test red, port
the minimum source behavior, and run it green before continuing.

### 4. Jafar source arm

Create:

- `src/world_marl/jafar/config.py`
- `src/world_marl/jafar/tokenizer.py`
- `src/world_marl/jafar/lam.py`
- `src/world_marl/jafar/dynamics.py`
- `src/world_marl/jafar/sampling.py`
- `src/world_marl/jafar/training.py`
- `src/world_marl/jafar/__init__.py`

Tests first lock the exact source defaults and then cover tokenizer
reconstruction/loss, LAM action-token alignment and reset-at-50 behavior,
first-frame masking, masked CE, the cosine MaskGIT schedule, the 25-step
refinement scan, the outer autoregressive scan, and frozen staged training.
Deterministic tiny overfit tests prove each of tokenizer, LAM, and dynamics can
reduce its own objective.

### 5. Jasmine source arm

Create:

- `src/world_marl/jasmine/config.py`
- `src/world_marl/jasmine/tokenizer.py`
- `src/world_marl/jasmine/lam.py`
- `src/world_marl/jasmine/dynamics.py`
- `src/world_marl/jasmine/sampling.py`
- `src/world_marl/jasmine/training.py`
- `src/world_marl/jasmine/__init__.py`

Tests first lock exact defaults and the NNX-to-Linen parameter/layout contract.
Then cover per-frame MAE masking, tanh latent bounds, sigmoid reconstruction,
source LAM alignment/VQ/reset, diffusion-level sampling, the exact linear noise
mixture, ramp weighting, clean-latent x-prediction, tokenizer freezing, default
LAM co-training without a decoder, 64-step denoising, context corruption 0.1,
and nested scan lowering. Tiny deterministic overfit tests cover the MAE, LAM,
and diffusion objectives.

### 6. Shared replay and target preparation

Create `src/world_marl/latent_action_world_model/` with focused files:

- `batching.py`: convert time-major `JaxSequenceBatch` arrays to batch-major
  backend sequences; pair `obs[t]`, `action[t]`, and
  `obs/reward/continue[t+1]`; reject non-HWC RGB; exclude
  `is_first[t+1]` transitions.
- `heads.py`: reuse the DreamerV3 255-bin symlog two-hot implementation for
  rewards and BCE-with-logits for continuation; expose sigmoid probabilities
  for rollout.
- `training.py`: train the heads against stop-gradient world-model features and
  provide scanned update functions.

Tests cover episode boundaries, integer/continuous action shapes, HWC
validation, reward support/encoding/decoding, continuation targets, finite
gradients, and deterministic tiny overfit of both heads. Gradient tests prove
that tokenizer, LAM, and dynamics receive zero gradients from both heads.

### 7. Expert calibration and real-action bridge

Create `bridge.py` with an immutable calibration artifact format. A required
NPZ contains observations, actions, `is_first`, environment, and provenance
metadata. The loader validates matching leading dimensions, HWC RGB, episode
boundaries, environment identity, and non-empty provenance.

Infer transition codes with the frozen LAM, retain every observed action in a
per-code list, and sample uniformly from that list. Reject artifacts missing
any of the six codes. There is no replay fallback and no default action.

Tests cover discrete and continuous actions, duplicate preservation, empirical
uniformity with fixed keys, all-six-code coverage rejection, environment and
provenance rejection, and transition-boundary filtering.

### 8. Scanned simulator and unchanged PPO core

Create:

- `simulator.py`: backend-specific state carries Jafar token history or
  Jasmine continuous-latent history plus replay reset context. Every step runs
  the complete source sampler, decodes HWC pixels, predicts reward and
  continuation, samples a calibrated real action, and resets from replay
  context when continuation terminates.
- `ppo.py`: feed decoded pixels to the existing `CNNActorCritic`, six-action
  categorical policy, existing GAE, and existing `ppo_update` without changing
  those implementations. Freeze the full world model during PPO.

Tests inspect lowered JAX programs for scan primitives, exercise termination
reset, verify complete-sampler invocation, assert decoded HWC policy inputs,
and prove that PPO updates actor/critic parameters without changing any world
model parameter. PPO minibatches and epochs remain scanned.

### 9. CLIs, artifacts, and migration

Add:

- `src/world_marl/scripts/train_jafar.py` and console script
  `world-marl-train-jafar`.
- `src/world_marl/scripts/train_jasmine.py` and console script
  `world-marl-train-jasmine`.
- `jafar` and `jasmine` arms in the visual world-model comparison tooling.
- a `world-marl-runpod` job named `jafar-jasmine-quality`.
- a quality evaluator for three fixed seeds and equal budgets.

Both train CLIs emit complete resolved configs, pinned source metadata, replay
metadata, model and optimizer checkpoints, per-stage JSONL metrics, code
usage, bridge/provenance artifacts, PPO metrics, rollout media, real evaluation,
`outcome.json`, and normalized `summary.json`.

Remove the predecessor package, its training script and
entry point, its dedicated tests and docs, and every reference from README,
comparison tools, quality tools, Runpod jobs, and tests. Do not leave import,
CLI, arm-name, or config aliases.

### 10. Acceptance gates

Run focused source-conformance tests after every red-green cycle, then:

```bash
uv run pytest tests/test_jafar_conformance.py tests/test_jafar.py
uv run pytest tests/test_jasmine_conformance.py tests/test_jasmine.py
uv run pytest tests/test_latent_action_world_model.py
uv run pytest tests/test_ippo.py tests/test_gae.py tests/test_train_scan.py
uv run pytest tests/test_compare_visual_wm.py tests/test_runpod.py
uv run pytest -m "not integration"
uv run pre-commit run --all-files
```

Run synthetic-pixel CLI smokes for both arms and inspect all mandatory
artifacts. Use the existing Runpod wrapper for source-sized 64x64
forward/backward/sampler checks and genuine
`playground-vision:CartpoleBalance` MJX/Warp GPU smokes. Finally run three
fixed-seed, equal-budget quality evaluations reporting random,
learned-simulator, and bridged-real returns plus reconstruction, dynamics,
code-usage, throughput, and utilization metrics.

No completion or merge claim is permitted until command output and artifact
paths demonstrate every gate.

## Commit boundaries

1. Documentation and third-party notices.
2. Dependency transaction.
3. Jafar primitives and arm, in red-green units.
4. Jasmine primitives and arm, in red-green units.
5. Shared targets, heads, and bridge.
6. Simulator/PPO integration.
7. CLIs, artifacts, migration, and local acceptance fixes.
8. GPU and quality evidence updates.

Each commit uses a conventional subject and includes only its verified scope.
