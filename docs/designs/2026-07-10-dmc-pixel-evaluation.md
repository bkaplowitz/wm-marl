# Genuine DMC Pixel Evaluation Design

## Objective

Evaluate `dreamer_v3_baseline` and `genie2_continuous_jax` on genuine
DeepMind Control Suite tasks rendered by MuJoCo. The existing
`pixels:pointmass` environment remains a synthetic plumbing fixture and is not
valid benchmark evidence.

## Environment Contract

The new environment namespace is `dmc-pixels:<domain>/<task>`. Construction
uses `dm_control.suite.load()` and `dm_control.suite.wrappers.pixels.Wrapper`
with `pixels_only=True`. The initial benchmark tasks are:

- `dmc-pixels:point_mass/easy`
- `dmc-pixels:point_mass/hard`
- `dmc-pixels:cartpole/swingup`
- `dmc-pixels:finger/spin`

The adapter returns float32 HWC observations in `[0, 1]`, while preserving the
underlying task's continuous action bounds, rewards, terminal timesteps, time
limits, and seeded reset behavior. It implements the same batched single-agent
contract as `DMCVectorAdapter`, including auto-reset and completed-episode
metadata.

Every replay batch and run summary records:

- `environment_backend = "dm_control"`
- `observation_mode = "pixels"`
- DMC domain and task
- image height, width, and camera ID
- seed and number of real environment transitions

Synthetic pixel runs record `environment_backend = "synthetic"` and are
rejected by the genuine benchmark aggregator.

## Model Evaluation Contract

Both model arms receive the same pixel observations, replay interaction budget,
task seed set, evaluation episode count, and action bounds. World-model
training uses actual DMC replay. Evaluation freezes model and policy parameters
and executes the learned controller in a separately seeded real DMC adapter.

Dreamer acts through its imagined actor. Genie2 selects continuous latent
actions in its learned simulator and reaches the real task only through the
learned latent-to-real-action bridge. Real actions do not become the primary
conditioning signal for Genie2 dynamics.

## Benchmark Protocol

The default genuine matrix uses four tasks and seeds `0,1,2,3,4`. Each run
writes its ordinary model artifacts plus environment provenance. Aggregation
reports per task and model:

- successful seed count
- mean and sample standard deviation of real return
- 95% confidence interval using the Student-t critical value for five seeds
- individual seed returns
- environment transitions, model updates, and imagined transitions
- final reconstruction/dynamics, reward, and continue losses
- Genie2 bridge error and bridged return

Random-action returns are collected from the same environment configuration and
seed set. Existing model-free and privileged-state results may be supplied as
additional summary files, but they are labeled as separate observation modes
and are not silently pooled with pixel results.

## Verification Levels

1. Unit tests use fake DMC environments to verify shapes and exact adapter
   semantics.
2. Integration tests instantiate official `point_mass/easy` through MuJoCo and
   verify rendered pixels and provenance.
3. Model smokes run each CLI on official `point_mass/easy` with tiny budgets to
   prove the complete real-environment path.
4. Benchmark runs use the four-task, five-seed matrix and are the only results
   described as genuine comparative evaluation.

References:

- [DeepMind Control Suite](https://github.com/google-deepmind/dm_control)
- [Official point_mass domain](https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/point_mass.py)
- [Official pixel wrapper](https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/wrappers/pixels.py)
