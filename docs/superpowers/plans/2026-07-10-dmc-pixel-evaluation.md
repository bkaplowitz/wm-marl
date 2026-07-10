# Genuine DMC Pixel Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add official dm_control/MuJoCo pixel environments and a reproducible four-task, five-seed evaluation for DreamerV3 and Genie2.

**Architecture:** A new `DMCPixelAdapter` wraps official DMC tasks without changing the existing vector adapter. Shared collection propagates environment provenance; each model CLI consumes the same adapter contract; a benchmark module constructs, executes, validates, and aggregates the fixed experiment matrix.

**Tech Stack:** Python 3.11, NumPy, dm-control, MuJoCo, JAX 0.4.36, Flax Linen, pytest.

## Global Constraints

- Keep `jax==0.4.36` and the existing Flax Linen stack.
- Do not alter `DMCVectorAdapter` behavior.
- Do not report `pixels:pointmass` as a real DMC evaluation.
- Main Genie2 dynamics remains conditioned on continuous latent actions, not real actions.
- Use tasks `point_mass/easy`, `point_mass/hard`, `cartpole/swingup`, and `finger/spin` with seeds `0` through `4` by default.
- Do not create new branches or worktrees and do not use a `codex/` name.

---

### Task 1: Official DMC Pixel Adapter

**Files:**
- Create: `src/world_marl/envs/dmc_pixel_adapter.py`
- Modify: `src/world_marl/envs/__init__.py`
- Test: `tests/test_dmc_pixel_adapter.py`

**Interfaces:**
- Produces: `make_dmc_pixel_env(env_id, seed, height, width, camera_id)`.
- Produces: `DMCPixelAdapter(env_id, num_envs, max_cycles, seed, image_size, camera_id, env_factory, auto_reset, num_workers)`.
- Produces: `is_dmc_pixel_substrate(str) -> bool` and `dmc_pixel_env_name(str) -> str`.

- [ ] **Step 1: Write failing fake-environment tests**

Test HWC float32 observations, continuous action bounds, reward forwarding,
terminal and time-limit handling, auto-reset, completed returns, threaded
stepping, and `close()`.

- [ ] **Step 2: Run the focused tests and confirm import failure**

Run: `uv run pytest tests/test_dmc_pixel_adapter.py -q`

Expected: failure because `world_marl.envs.dmc_pixel_adapter` does not exist.

- [ ] **Step 3: Implement the official wrapper and adapter**

Construct the base environment with:

```python
base = suite.load(
    domain_name=domain_name,
    task_name=task_name,
    task_kwargs={"random": seed},
)
return pixels.Wrapper(
    base,
    pixels_only=True,
    render_kwargs={"height": height, "width": width, "camera_id": camera_id},
)
```

Normalize only the adapter output, not the wrapped environment, and derive all
action metadata from `action_spec()`.

- [ ] **Step 4: Run adapter tests**

Run: `uv run pytest tests/test_dmc_pixel_adapter.py tests/test_dmc_adapter.py -q`

Expected: all pass.

### Task 2: Shared Replay and Provenance

**Files:**
- Modify: `src/world_marl/world_model_foundation/collect.py`
- Modify: `src/world_marl/world_model_foundation/replay.py`
- Test: `tests/test_world_model_foundation.py`

**Interfaces:**
- `make_single_agent_adapter(..., image_size=64, dmc_camera_id=0)` dispatches `dmc-pixels:`.
- `collect_adapter_sequence()` includes backend, observation mode, DMC task, render configuration, and real transition count in metadata.

- [ ] **Step 1: Add failing dispatch and metadata tests**

Assert `dmc-pixels:point_mass/easy` selects `DMCPixelAdapter`; malformed names
raise the DMC-specific error; synthetic and DMC backends have distinct
provenance; HWC shapes survive replay collection.

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `uv run pytest tests/test_world_model_foundation.py -q`

- [ ] **Step 3: Add dispatch parameters and provenance copying**

The collector reads a structured `environment_metadata` mapping from adapters
and records `real_env_transitions = time_steps * num_envs`.

- [ ] **Step 4: Run foundation tests**

Run: `uv run pytest tests/test_world_model_foundation.py tests/test_dmc_pixel_adapter.py -q`

Expected: all pass.

### Task 3: Genuine Benchmark Matrix and Aggregation

**Files:**
- Create: `src/world_marl/scripts/benchmark_dmc_pixels.py`
- Modify: `src/world_marl/scripts/compare_visual_wm.py`
- Modify: `pyproject.toml`
- Create: `tests/test_benchmark_dmc_pixels.py`
- Modify: `tests/test_compare_visual_wm.py`

**Interfaces:**
- CLI: `world-marl-benchmark-dmc-pixels`.
- Default tasks: the approved four-task list.
- Default seeds: `0,1,2,3,4`.
- Produces: `commands.json`, `runs.json`, `aggregate.json`, and `aggregate.csv`.

- [ ] **Step 1: Add failing command-matrix and statistics tests**

Verify 40 commands for two arms, four tasks, and five seeds; each command has a
unique output directory and explicit seed; reject summaries without
`environment_backend == "dm_control"` and `observation_mode == "pixels"`;
verify mean, sample standard deviation, and 95% interval on known values.

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `uv run pytest tests/test_benchmark_dmc_pixels.py tests/test_compare_visual_wm.py -q`

- [ ] **Step 3: Implement dry-run, execution, resume, and aggregation**

Use `subprocess.run(command, check=False)` for each missing run. Preserve failed
runs in `runs.json`, do not treat `--allow-fail` learning gates as process
crashes, and never pool vector and pixel observation modes.

- [ ] **Step 4: Run comparison tests**

Run: `uv run pytest tests/test_benchmark_dmc_pixels.py tests/test_compare_visual_wm.py -q`

Expected: all pass.

### Task 4: Propagate Foundation to Model Branches

**Files:**
- Modify: `src/world_marl/scripts/train_dreamer_v3_baseline.py`
- Modify: `src/world_marl/scripts/train_genie2_continuous_jax.py`
- Modify: both model `ARCHITECTURE.md` files
- Modify: `tests/test_dreamer_v3_baseline.py`
- Modify: `tests/test_genie2_continuous_jax.py`

**Interfaces:**
- Both CLIs accept `--env dmc-pixels:<domain>/<task>`, `--image-size`, and `--dmc-camera-id`.
- Every `summary.json` records environment provenance, training/evaluation seed,
  real transitions, model updates, and imagined transitions.

- [ ] **Step 1: Add failing model-CLI provenance tests**

Use a fake adapter to assert that each CLI forwards render settings and writes
the required fields. Assert Dreamer reports `policy_source=imagined_actor` and
Genie2 reports `policy_source=latent_policy_bridge`.

- [ ] **Step 2: Run focused tests and confirm failure**

Run each model test file with `uv run pytest ... -q`.

- [ ] **Step 3: Implement model plumbing and documentation**

Pass image settings to collection and evaluation adapter construction, preserve
the existing learned-control paths, and copy replay provenance into run
artifacts.

- [ ] **Step 4: Run model suites**

Run: `uv run pytest tests/test_dreamer_v3_baseline.py -q`

Run: `uv run pytest tests/test_genie2_continuous_jax.py -q`

Expected: all pass.

### Task 5: Real MuJoCo Verification

**Files:**
- Modify only if verification exposes a concrete defect.

**Interfaces:**
- Official DMC integration test marker: `integration`.
- Real model smokes write complete artifacts under `/tmp`.

- [ ] **Step 1: Run the official adapter integration test**

Run: `uv run --extra dmc pytest tests/test_dmc_pixel_adapter.py -m integration -q`

Expected: `point_mass/easy` reset returns nonblank HWC pixels and a real step
returns finite reward with dm_control provenance.

- [ ] **Step 2: Run Dreamer on official point_mass/easy**

Run the Dreamer CLI with `--env dmc-pixels:point_mass/easy --image-size 32
--num-envs 1 --collect-steps 8 --train-steps 2 --policy-train-steps 2
--eval-episodes 1 --allow-fail`.

- [ ] **Step 3: Run Genie2 on official point_mass/easy**

Run the Genie2 CLI with the same environment and budget.

- [ ] **Step 4: Validate artifacts and dry-run the full matrix**

Confirm both summaries identify `dm_control` and `pixels`, then run
`world-marl-benchmark-dmc-pixels --dry-run` and verify 40 commands.

- [ ] **Step 5: Run full regression and lint**

Run foundation, Dreamer, Genie2, existing JEPA/genwm tests, and
`uv run ruff check src tests`.

Expected: all pass.
