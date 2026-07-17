# Jafar/Jasmine Change Log

## 2026-07-16

### Starting point

- Branch: `jafar-jasmine-world-models`.
- Starting commit: `44eb8e94d39dae4b129ba5fe89ddd78670fc4d64`.
- Baseline command: `uv run pytest -m "not integration"`.
- Baseline result: 427 passed, 3 skipped, 7 deselected; exit 0.
- JAX: 0.4.36.
- Flax: 0.10.4.

### Documentation-first change

- Added architecture and source-conformance contracts for the Jafar arm.
- Added architecture and source-conformance contracts for the Jasmine arm.
- Added the approved implementation plan and acceptance sequence.
- Added Apache-2.0 third-party notices for both pinned sources.

No model, test, dependency, CLI, or runtime behavior changed in this
documentation-first commit.

### Source ports

- Documentation contract commit: `1cc649b`.
- Dependency commit: `983b2e2`; added `dm-pix>=0.4.3` and `einops>=0.8.0`
  through `uv add`. JAX 0.4.36 and Flax 0.10.4 were unchanged.
- Jafar source-port commit: `f34f746`; its 16 focused conformance tests passed.
- Jasmine source-port commit: `9955ab7`; its 13 focused conformance tests passed.
- Shared replay, reward/continuation, bridge, simulator, and scanned-PPO commit:
  `9c72eae`; its 10 focused tests passed.

### Training, migration, and quality workflow

- Added independent `world-marl-train-jafar` and
  `world-marl-train-jasmine` commands with staged scan training, source-sized
  stage batches, frozen-model heads, complete samplers, scanned PPO, real
  bridged evaluation, checkpoints, JSONL metrics, media, provenance, outcome,
  and normalized summary artifacts.
- Added Jafar and Jasmine comparison arms and removed the predecessor package,
  command, quality workflow, tests, and stale design documents. A repository
  search over package code, entry points, comparison/quality scripts, README,
  tests, and docs found no predecessor identifiers.
- Added `playground-vision:CartpoleBalance`. It preserves HWC pixels inside the
  existing scanned adapter and converts the upstream centered grayscale stack
  to `[0, 1]`.
- Added `world-marl-runpod --job jafar-jasmine-quality`: three fixed seeds,
  equal source-sized budgets, MJX/Warp plus MJWarp rendering, utilization
  samples, random/learned-simulator/bridged-real returns, source checks, and
  normalized aggregation.
- Playground 0.2.0 requires a newer Brax than JAXMarl 0.1.0 permits. The Runpod
  job therefore performs an explicit no-dependency GPU-only upgrade of
  Playground 0.2.0, MuJoCo/MJX 3.6.0, and Warp 1.11.0 after the locked sync,
  executes directly from `.venv`, and asserts JAX 0.4.36 and Flax 0.10.4 before
  starting. This avoids changing the repository's JAXMarl/Brax dependency
  contract.

### Local evidence

- Both synthetic-pixel debug CLIs completed with staged training, full
  samplers, reward/continue heads, scanned PPO, and artifact checks.
- Focused non-integration migration suite: 92 passed, 2 deselected; exit 0.
- Optional local MuJoCo Playground vector regression also passed after its
  one-time external asset initialization.
- Ruff checks for all migration files passed.
- Repository-wide non-integration suite: 433 passed, 9 deselected; exit 0;
  486.25 seconds.
- `pre-commit run --all-files`: yamlfmt, keep-sorted, Ruff check, and Ruff
  format passed. Its first pass made only mechanical formatting changes in the
  example training YAML and existing evaluation signatures; the second pass
  was clean.

### Pending acceptance evidence

No expert calibration NPZ for `playground-vision:CartpoleBalance` exists in this
worktree. The required Runpod source-sized checks, genuine GPU smokes, and
three-seed quality runs cannot start until a separate expert-calibration file
with the required observations, actions, episode starts, environment, and
provenance metadata is supplied. No GPU completion or merge claim is made.

### Evidence policy

Future entries record the exact red/green commands, commits, local artifacts,
GPU artifacts, and quality outputs produced by each implementation phase.
Results are not marked complete until fresh command output and artifact
inspection satisfy the corresponding gate.
