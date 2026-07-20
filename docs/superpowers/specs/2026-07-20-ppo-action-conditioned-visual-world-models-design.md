# PPO-Action-Conditioned Visual World Models

## Decision

Replace the expert-calibrated latent-action bridge with direct conditioning on
the actions emitted by PPO. Keep the existing categorical `CNNActorCritic`,
GAE, and `ppo_update`. For `playground-vision:CartpoleBalance`, expose six
categorical actions and decode them deterministically to scalar forces at the
environment-adapter boundary.

The six force values are evenly spaced across the environment range:

```text
action id:  0     1     2     3     4     5
force:     -1.0  -0.6  -0.2   0.2   0.6   1.0
```

The policy action ID is the canonical action everywhere except the final real
environment call. Replay, world-model training, imagined rollouts, PPO
rollouts, checkpoints, and metrics all use the same integer ID. This removes
the need for expert data, inferred-code coverage, or a learned latent-to-real
mapping.

## Motivation and source boundary

Genie v1's expert mapping belongs to its behavioral-cloning experiment, not to
its reproducible world-model experiment. Jafar implements the latter: it
collects random-action CoinRun videos, discards the real actions, and learns a
latent action model from pixels. Jasmine additionally implements an upstream
`use_gt_actions=True` mode that removes the LAM and conditions dynamics on
provided discrete action IDs.

This repository trains PPO from rollouts rather than imitating an expert. PPO
already supplies the action used for each transition, so reconstructing a
second action space and calibrating it back to the first is unnecessary and
lossy.

Source-conformance documentation will make the boundary explicit:

- Jasmine's direct discrete-action path follows its pinned upstream
  `use_gt_actions=True` behavior.
- Jafar's direct discrete-action path is a repository extension. Its tokenizer
  and MaskGIT equations remain source-derived, while an action embedding
  replaces LAM codebook lookup for the PPO integration path.
- The original source-conformant LAM modules remain available and tested, but
  they are not trained or used by the PPO action-conditioned commands.

## Action adapter

Add an optional scalar-action discretizer to the Brax/Playground adapter. The
default adapter behavior remains unchanged. When configured with the six
Cartpole force values, the adapter:

- reports a discrete action space with six categorical actions;
- samples integer action IDs during random scanned collection;
- records those IDs in `WorldModelSequenceBatch.actions`;
- decodes IDs to arrays of shape `(num_envs, 1)` immediately before calling the
  Playground environment;
- applies the same decoding inside random, policy, recurrent, and online
  `jax.lax.scan` rollout paths;
- records the ordered force table and its provenance in environment metadata;
- validates non-integer or out-of-range IDs at host-facing boundaries rather
  than clipping or guessing. Scanned paths obtain IDs only from categorical
  sampling, so their values are valid by construction.

The transform is a pure JAX gather and introduces no per-step host
synchronization. Continuous Brax adapters that do not request discretization
retain their existing interface and behavior.

## Model data flow

For each valid transition, the backend receives:

```text
(pixels[t], action_id[t], pixels[t+1], reward[t+1], continue[t+1])
```

Transitions crossing `is_first[t+1]` remain excluded.

The runner creates the PPO train state before collecting the world-model
training replay. Collection uses that policy's categorical action sampler in a
JAX-native adapter scan, so the recorded actions come from the same policy
network family later trained in imagination. An explicitly random rollout is
collected separately for the random-return baseline; it is not an expert or
imitation dataset. The lower-level collector retains a policy callback so a
warm-started or restored behavior policy can be supplied without changing the
world-model objective.

Both arms embed `action_id[t]` into the source latent-action width and use that
embedding at the same additive conditioning point used by their dynamics
models. The action embedding is trained jointly with dynamics.

The training stages become:

1. Collect action-labeled pixel replay with the initialized, warm-started, or
   restored PPO behavior policy.
2. Train the tokenizer from replay pixels.
3. Train action-conditioned dynamics with the tokenizer frozen.
4. Train reward and continuation heads from frozen world-model features and
   the corresponding real action IDs.
5. Freeze the complete world model and train categorical PPO in imagined
   rollouts.
6. Evaluate the same PPO checkpoint in the real environment, where action IDs
   are decoded by the adapter.

No LAM stage, latent-code inference, expert calibration, replay-derived action
bridge, default action, or policy relabeling occurs in this path.

## Jafar arm

Add a static ground-truth-action mode to the composed Jafar world model. In
that mode:

- instantiate a six-entry, width-32 action embedding;
- do not instantiate or call the LAM from the composed training/sampling path;
- feed embedded action IDs to the unchanged MaskGIT dynamics conditioning
  point;
- retain the source tokenizer, masking objective, causal attention, 25-step
  MaskGIT sampler, and outer autoregressive scan;
- keep standalone LAM source-conformance tests and attribution.

Artifacts must identify this as `jafar` with
`action_conditioning=ground_truth_discrete_repo_extension`, not as exact
upstream Jafar behavior.

## Jasmine arm

Port and enable the pinned Jasmine `use_gt_actions=True` branch for the
diffusion arm. In that mode:

- instantiate a six-entry, width-32 action embedding;
- omit the LAM and LAM decoder;
- condition diffusion dynamics directly on embedded action IDs;
- retain the MAE tokenizer, diffusion equations, dtypes, rematerialization,
  attention behavior, 64-step sampler, and context corruption.

Artifacts must identify this as `jasmine` with
`action_conditioning=ground_truth_discrete_upstream`.

## PPO and simulator

The existing `CNNActorCritic`, categorical distribution, GAE, and
`ppo_update` remain unchanged. PPO logits have width six. A sampled action ID
is passed unchanged to the Jafar or Jasmine simulator and stored unchanged in
the PPO rollout batch.

Each simulator step still runs the complete source sampler, decodes pixels,
predicts reward and continuation, and resets from replay context after a
sampled termination. Simulator state stores action-ID history rather than
latent-code history. All rollout, sampler, and PPO update loops remain
`jax.lax.scan` based.

Real evaluation passes the same categorical policy action to the discretized
Playground adapter. There is no separate real-action policy and no action
selection based on future observations.

## CLI and artifacts

Remove `--expert-calibration` from both training commands and the Runpod
quality job. Delete the expert-bridge module and its tests when no references
remain.

Replace bridge artifacts and metric names:

- remove `bridge.json`, bridge coverage, calibration hashes, and
  `bridged_real_return`;
- add `action_conditioning.json` containing the action IDs, ordered force
  values, environment bounds, source/extension classification, and no-fallback
  declaration;
- report `real_policy_return` alongside `random_return` and
  `learned_simulator_return`;
- retain complete configs, source metadata, replay metadata, checkpoints,
  per-stage JSONL metrics, rollout media, `outcome.json`, and normalized
  `summary.json`;
- omit LAM metrics and code-usage artifacts from the direct-action commands.

The comparison arms remain `jafar` and `jasmine`. No Genie2 compatibility
alias or new Genie2/Genie3 claim is introduced.

## Validation

Use test-driven development for each production change. Required focused
coverage includes:

- exact six-bin values, ID validation, JIT compatibility, and deterministic
  decoding;
- scanned random and policy collection recording IDs while the environment
  receives scalar forces;
- unchanged continuous-adapter behavior when discretization is disabled;
- Jafar and Jasmine action-embedding shapes, conditioning, sampling, and
  gradients;
- absence of the LAM from direct-action parameter trees and optimizer updates;
- transition pairing with action IDs and episode-boundary exclusion;
- scanned simulator and PPO updates using identical action IDs;
- real evaluation using the same policy checkpoint and fixed decoder;
- both synthetic-pixel CLI smokes without an NPZ;
- removal of bridge CLI, files, artifacts, and terminology;
- existing PPO, adapter, world-model, source-conformance, and packaging
  regressions;
- `pre-commit run --all-files` and the full non-integration suite.

GPU validation retains the source-sized forward/backward/sampler checks and
three fixed-seed equal-budget Cartpole quality evaluation. The compared return
metrics become random, learned-simulator, and direct real-policy returns.

## Non-goals

- Continuous Gaussian PPO or changes to `ppo_update`.
- A learned replay-to-action bridge.
- Behavioral cloning or expert-data ingestion.
- Joint LAM/dynamics training in the PPO action-conditioned commands.
- Changes to JAX or Flax versions.
- Torch, Procgen, Grain, or ArrayRecord dependencies.
