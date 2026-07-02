# `train_e2e` entrypoint/core split via `TrainConfig` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `train_e2e.py` a pure training core that consumes a typed `TrainConfig` dataclass instead of `argparse.Namespace`, keeping the argparse CLI as a thin entrypoint that builds the config.

**Architecture:** A new `src/world_marl/config.py` defines `TrainConfig` (a plain, mutable dataclass mirroring all 41 argparse dests) plus `TrainConfig.from_namespace`. The entrypoint (`parse_args` + `main`) parses argv → Namespace → `TrainConfig`; the core (`run_training` and the config-taking helpers) takes `cfg: TrainConfig`. `benchmark_policy` converts its Namespace to `TrainConfig` before calling the core. No Hydra, no CLI-syntax change, `runpod.py` untouched.

**Tech Stack:** Python 3.11, `dataclasses`, argparse, OmegaConf (existing `--config` loader), pytest, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-07-01-train-config-split-design.md` (in the worktree).

## Global Constraints

- Work only in the worktree `/Users/bkaplowitz/Developer/work/feat-train-config-split` (branch `feat-train-config-split`, off `feat-wandb-omegaconf`).
- `TrainConfig` is a **plain `@dataclass`** — NOT `frozen` (mutation sites: `evaluate_checkpoint_mode` lines 407/409/410/413; `benchmark_policy._arm_train_args` lines 115/116).
- `TrainConfig` fields = the 41 argparse dests, **same names, same defaults, in argparse-declaration order**.
- Do **not** change the argparse CLI, flag names, `--config`/`--wandb` behavior, `runpod.py`, README, or `configs/`.
- Keep the helper **function name** `algorithm_config_from_args` (imported by name in a test); only its parameter is renamed `args`→`cfg`.
- Conventional commits. Commit at the end of each task. **Do not push** unless the user asks.
- Use `fd`/`rg` (not `find`/`grep`). Never `python -c` multiline; write scratch scripts under the scratchpad and run with `uv run python`.
- After user approval, copy this plan to `docs/superpowers/plans/2026-07-01-train-config-split.md` inside the worktree (plan-mode currently restricts edits to this file).

## Context

`train_e2e.run_training(args: argparse.Namespace, …)` is the training core, but it is reached two ways: `main()` (the CLI) and `benchmark_policy.run_arm()` (via a `sys.argv`-swap that reuses `train_e2e.parse_args()`). That couples the core to argparse and makes it awkward to drive from scripts/sweeps. Introducing a `TrainConfig` dataclass as the core's input decouples both callers, makes `benchmark_policy`'s reuse explicit, and pre-builds the Hydra structured-config schema so a later Hydra/Optuna migration is additive (register the dataclass + swap the entrypoint) rather than invasive. This branch does the structural split only.

## File Structure

| File | Responsibility |
|------|----------------|
| `src/world_marl/config.py` | **new** — `TrainConfig` dataclass + `from_namespace` (the typed core input / future Hydra schema) |
| `src/world_marl/scripts/train_e2e.py` | entrypoint (`parse_args`, `main`) builds `TrainConfig`; core fns retyped to `cfg: TrainConfig` |
| `src/world_marl/scripts/benchmark_policy.py` | `_arm_train_args` returns `TrainConfig`; core reuse now goes through the typed object |
| `tests/test_train_config.py` | **new** — argparse↔dataclass drift guard + `from_namespace` round-trip |
| `tests/test_train_e2e_coins_e2e.py` | direct `run_training` caller converts Namespace→`TrainConfig` |
| `tests/test_train_e2e_prefit.py` | `_args()` helper builds a partial `TrainConfig` |

---

### Task 1: `TrainConfig` dataclass + drift-guard tests

**Files:**
- Create: `src/world_marl/config.py`
- Test: `tests/test_train_config.py`

**Interfaces:**
- Produces: `world_marl.config.TrainConfig` — a mutable dataclass with the 41 fields below (all defaulted) and `@classmethod from_namespace(cls, namespace: argparse.Namespace) -> TrainConfig` returning `cls(**vars(namespace))`.

- [ ] **Step 1: Write the failing tests**

`tests/test_train_config.py`:

```python
from __future__ import annotations

import dataclasses
import sys

from world_marl.config import TrainConfig
from world_marl.scripts import train_e2e


def test_trainconfig_defaults_match_argparse(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["world-marl-train-e2e"])
    namespace = train_e2e.parse_args()
    assert dataclasses.asdict(TrainConfig()) == vars(namespace)


def test_from_namespace_round_trips(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["world-marl-train-e2e", "--num-runs", "7", "--wm-flow-type", "discrete"],
    )
    namespace = train_e2e.parse_args()
    cfg = TrainConfig.from_namespace(namespace)
    assert dataclasses.asdict(cfg) == vars(namespace)
    assert cfg.num_runs == 7
    assert cfg.wm_flow_type == "discrete"
```

Rationale: `asdict(TrainConfig()) == vars(parse_args([]))` checks BOTH field-name parity and default parity in one assertion (empty argv is safe — no required flags; the `prefit_world_model`/`wm_policy_warmup_updates` validation branches are skipped at defaults). `from_namespace` uses `cls(**vars(namespace))`, so a missing/renamed/extra field raises `TypeError` on construction — drift cannot pass silently.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/bkaplowitz/Developer/work/feat-train-config-split && uv run pytest tests/test_train_config.py -q`
Expected: FAIL / collection error — `ModuleNotFoundError: No module named 'world_marl.config'`.

- [ ] **Step 3: Create `src/world_marl/config.py`**

```python
"""Typed training configuration for the ``train_e2e`` core.

``TrainConfig`` mirrors the ``train_e2e`` argparse dests one-to-one so the pure
training core consumes a structured object instead of ``argparse.Namespace``.
It doubles as the schema for a future Hydra structured config.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class TrainConfig:
    config: str | None = None
    wandb: bool = False
    wandb_project: str = "world-marl"
    algorithm: str = "ippo"
    substrate: str = "coins"
    num_envs: int = 4
    rollout_steps: int = 128
    total_env_steps: int = 100_000
    eval_episodes: int = 50
    num_runs: int = 3
    seed: int = 0
    max_cycles: int = 1000
    observation_size: int | None = None
    append_agent_id: bool = False
    include_observation_scalars: bool = False
    stochastic_eval: bool = False
    eval_max_steps: int | None = None
    out_dir: str = "runs"
    min_improvement: float = 0.2
    negative_control: str = "freeze-policy"
    prefit_world_model: bool = False
    wm_random_rollouts: int = 1
    wm_initial_rollouts: int = 1
    wm_fit_steps: int = 10_000
    wm_learning_rate: float = 3e-4
    wm_hidden_dim: int = 256
    wm_integration_steps: int = 10
    wm_policy_warmup_updates: int = 0
    wm_flow_type: str = "linear"
    wm_num_categories: int = 9
    learning_rate: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 4
    activation: str = "relu"
    eval_checkpoint: str | None = None

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> "TrainConfig":
        return cls(**vars(namespace))
```

(Field order matches the `add_argument` order in `parse_args`, except `eval_checkpoint` sits last exactly as in argparse. `choices=` sets are intentionally left as plain strings for this branch.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_train_config.py -q`
Expected: PASS (2 passed). If `test_trainconfig_defaults_match_argparse` fails, the diff of the two dicts names the drifted field — fix the dataclass field/default to match argparse.

- [ ] **Step 5: Lint**

Run: `uv run ruff format src/world_marl/config.py tests/test_train_config.py && uv run ruff check src/world_marl/config.py tests/test_train_config.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/world_marl/config.py tests/test_train_config.py
git commit -m "feat(config): add TrainConfig dataclass mirroring train_e2e args"
```

---

### Task 2: Cut the `train_e2e` core + `benchmark_policy` over to `TrainConfig`

Atomic type cutover: `run_training`'s parameter type changes and it starts calling `dataclasses.asdict(cfg)`, so every production caller and the two direct-caller tests must move together to keep the commit green.

**Files:**
- Modify: `src/world_marl/scripts/train_e2e.py` (import; `run_training` 590-654 + 606/622 + 1008; `algorithm_config_from_args` 238-263; `evaluate_checkpoint_mode` 402-447; `_make_training_adapter` 326-340; `evaluate_random_baseline` at its def; `main` 1081-1118)
- Modify: `src/world_marl/scripts/benchmark_policy.py` (import; `_arm_train_args` 95-117)
- Test: `tests/test_train_e2e_coins_e2e.py` (76-82), `tests/test_train_e2e_prefit.py` (`_args`)

**Interfaces:**
- Consumes: `world_marl.config.TrainConfig`, `TrainConfig.from_namespace` (Task 1).
- Produces: `run_training(cfg: TrainConfig, *, run_dir, name, run_index, control) -> RunOutcome`; `algorithm_config_from_args(cfg: TrainConfig, control=None)`; `evaluate_checkpoint_mode(cfg: TrainConfig)`; `_make_training_adapter(cfg: TrainConfig, *, seed)`; `evaluate_random_baseline(cfg: TrainConfig, *, seed)`; `benchmark_policy._arm_train_args(...) -> TrainConfig`.

- [ ] **Step 1: Import `TrainConfig` in `train_e2e.py`**

Add after the `from world_marl.logging import …` line (currently line 45), grouped with the other `world_marl.*` imports:

```python
from world_marl.config import TrainConfig
```

- [ ] **Step 2: Retype the config-taking helpers and rename `args`→`cfg` in their bodies**

For each of these functions, change the parameter `args: argparse.Namespace` → `cfg: TrainConfig` and replace every `args.` with `cfg.` inside the body. The inner locals named `config` (the algo config) are a different name and stay untouched.

- `algorithm_config_from_args` (238-263): signature `def algorithm_config_from_args(cfg: TrainConfig, control: str | None = None) -> IPPOConfig | MAPPOConfig:`; body reads become `cfg.algorithm`, `getattr(cfg, "prefit_world_model", False)`, `cfg.learning_rate`, … `cfg.activation`.
- `_make_training_adapter` (326-340): signature `def _make_training_adapter(cfg: TrainConfig, *, seed: int) -> TrainingAdapter:`; body reads `cfg.substrate`, `cfg.num_envs`, `cfg.max_cycles`, `cfg.observation_size`, `cfg.include_observation_scalars`, `cfg.append_agent_id`.
- `evaluate_random_baseline` (called at line 633 as `evaluate_random_baseline(args, seed=seed + 1)`): retype its `args` parameter to `cfg: TrainConfig` and rename its body `args.`→`cfg.` (locate the def with `rg -n "def evaluate_random_baseline" src/world_marl/scripts/train_e2e.py`).
- `evaluate_checkpoint_mode` (402-447): signature `def evaluate_checkpoint_mode(cfg: TrainConfig) -> None:`; the in-place backfills stay as assignments (mutable dataclass), now on `cfg`:

```python
def evaluate_checkpoint_mode(cfg: TrainConfig) -> None:
    checkpoint_dir = Path(cfg.eval_checkpoint)
    metadata = load_metadata(checkpoint_dir)
    algorithm = metadata.get("algorithm", "ippo")

    cfg.substrate = cfg.substrate or metadata["substrate"]
    if cfg.observation_size is None:
        cfg.observation_size = metadata.get("observation_size")
    cfg.include_observation_scalars = cfg.include_observation_scalars or metadata.get(
        "include_observation_scalars", False
    )
    cfg.append_agent_id = cfg.append_agent_id or metadata.get(
        "append_agent_id", False
    )
    adapter = _make_training_adapter(cfg, seed=cfg.seed)
    try:
        # ... unchanged body, with the remaining args.* reads renamed to cfg.* ...
```

(Continue renaming the rest of `evaluate_checkpoint_mode`'s `args.` reads: `cfg.stochastic_eval`, `cfg.seed`, `cfg.eval_episodes`, `cfg.eval_max_steps`.)

- [ ] **Step 3: Retype `run_training` and switch `vars(args)` → `dataclasses.asdict(cfg)`**

Signature (590-597) → `def run_training(cfg: TrainConfig, *, run_dir: Path, name: str, run_index: int, control: str | None) -> RunOutcome:`. Rename every `args.` in the body to `cfg.` (`cfg.wandb`, `cfg.wandb_project`, `cfg.seed`, `cfg.substrate`, `cfg.prefit_world_model`, `cfg.algorithm`, `cfg.stochastic_eval`, …). Change the two `vars(args)` sites:

- Line 606 (wandb init): `config=dataclasses.asdict(cfg),`
- Line 622 (config.json): `"args": dataclasses.asdict(cfg),`

Everything else (including `config = algorithm_config_from_args(cfg, control)` at 613, `dataclasses.asdict(config)` at 628, and `wandb_run.finish()` at 1008) is unchanged.

- [ ] **Step 4: Build the config in `main()`**

Replace `main()` (1081-1118) so it parses argv then constructs the config once and drives everything with `cfg`:

```python
def main() -> None:
    cfg = TrainConfig.from_namespace(parse_args())
    if cfg.eval_checkpoint:
        evaluate_checkpoint_mode(cfg)
        return

    experiment_dir = Path(cfg.out_dir) / f"e2e_{timestamp()}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    outcomes = [
        run_training(
            cfg,
            run_dir=experiment_dir / f"run_{run_index:03d}",
            name=f"run_{run_index:03d}",
            run_index=run_index,
            control=None,
        )
        for run_index in range(cfg.num_runs)
    ]

    control_outcome = None
    if cfg.negative_control != "none":
        control_outcome = run_training(
            cfg,
            run_dir=experiment_dir / f"control_{cfg.negative_control}",
            name=f"control_{cfg.negative_control}",
            run_index=cfg.num_runs,
            control=cfg.negative_control,
        )

    summary = summarize(
        outcomes,
        control_outcome,
        min_improvement=cfg.min_improvement,
    )
    RunLogger(experiment_dir).write_json("summary.json", summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)
```

- [ ] **Step 5: Convert `benchmark_policy._arm_train_args` to return `TrainConfig`**

Add the import after `from world_marl.scripts import train_e2e` (line 15):

```python
from world_marl.config import TrainConfig
```

Change `_arm_train_args` (95-117): keep the two Namespace fixups on `parsed`, then convert on return. Update the return annotation `-> argparse.Namespace` → `-> TrainConfig`:

```python
    parsed = _parse_train_args(args)
    parsed.negative_control = "none"
    parsed.out_dir = str(out_dir)
    return TrainConfig.from_namespace(parsed)
```

`run_arm` needs no other change: `train_args` is now a `TrainConfig`, and `train_args.num_runs` / `train_args.min_improvement` / the `train_e2e.run_training(train_args, …)` call all work against the typed object.

- [ ] **Step 6: Update the two direct-caller tests**

`tests/test_train_e2e_coins_e2e.py` — convert at the call site (import + wrap). After `args = _tiny_coins_args(tmp_path, monkeypatch)` (line 73), pass a `TrainConfig`:

```python
from world_marl.config import TrainConfig  # add near the other imports (after line 20)

    # in the test body:
    cfg = TrainConfig.from_namespace(_tiny_coins_args(tmp_path, monkeypatch))
    run_dir = tmp_path / "run_000"
    outcome = train_e2e.run_training(
        cfg,
        run_dir=run_dir,
        name="run_000",
        run_index=0,
        control=None,
    )
```

`tests/test_train_e2e_prefit.py` — make `_args()` build a partial `TrainConfig` (defaults cover the unset fields). Replace the `Namespace` import and `_args` helper:

```python
from world_marl.config import TrainConfig


def _args(*, algorithm: str) -> TrainConfig:
    return TrainConfig(
        algorithm=algorithm,
        substrate="coins",
        num_envs=1,
        max_cycles=5,
        observation_size=None,
        include_observation_scalars=False,
        append_agent_id=False,
        prefit_world_model=True,
        learning_rate=5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=1,
        num_minibatches=1,
        activation="relu",
    )
```

The existing `args.prefit_world_model = False` mutation in `test_coins_selects_mlp_policy_config_without_prefit` still works (mutable dataclass). Drop the now-unused `from argparse import Namespace` import.

- [ ] **Step 7: Lint the changed files**

Run: `uv run ruff format src/world_marl/scripts/train_e2e.py src/world_marl/scripts/benchmark_policy.py tests/test_train_e2e_coins_e2e.py tests/test_train_e2e_prefit.py && uv run ruff check` (same paths)
Expected: no errors. Then sanity-check nothing else still does `vars(`/`argparse.Namespace` on the config: `rg -n "vars\(args\)|vars\(cfg\)" src/world_marl/scripts/train_e2e.py` should return nothing.

- [ ] **Step 8: Run the affected pure/light tests**

Run: `uv run pytest tests/test_train_config.py tests/test_config_and_wandb.py tests/test_train_e2e_prefit.py tests/test_runpod.py -q`
Expected: PASS. (`test_train_e2e_prefit` exercises the retyped `algorithm_config_from_args`/`_make_training_adapter` against a `TrainConfig`; `test_runpod` proves the untouched CLI contract still parses.)

- [ ] **Step 9: Run the end-to-end direct-caller test (real short compute — confirm first)**

This test runs a tiny real training loop (`--total-env-steps 8`, coins). Per the no-unilateral-heavy-runs rule, confirm it is OK to run (it is short, seconds-to-a-minute on CPU) before executing:

Run: `uv run pytest tests/test_train_e2e_coins_e2e.py -q`
Expected: PASS — the converted `run_training(cfg, …)` call completes and writes artifacts.

- [ ] **Step 10: Commit**

```bash
git add src/world_marl/scripts/train_e2e.py src/world_marl/scripts/benchmark_policy.py tests/test_train_e2e_coins_e2e.py tests/test_train_e2e_prefit.py
git commit -m "refactor(train_e2e): take TrainConfig in the training core"
```

---

## Verification (end-to-end)

Run inside the worktree `/Users/bkaplowitz/Developer/work/feat-train-config-split`:

1. **Lint:** `uv run ruff format --check` + `uv run ruff check` on all changed files → clean.
2. **Drift guard:** `uv run pytest tests/test_train_config.py -q` → both tests pass (proves the dataclass mirrors argparse exactly).
3. **Regression (light):** `uv run pytest tests/test_train_config.py tests/test_config_and_wandb.py tests/test_train_e2e_prefit.py tests/test_runpod.py -q` → pass.
4. **Core end-to-end (short compute, confirm first):** `uv run pytest tests/test_train_e2e_coins_e2e.py -q` → pass (converted direct `run_training(cfg, …)` caller).
5. **CLI unchanged:** `uv run world-marl-train-e2e --config configs/train_e2e.example.yaml --num-runs 1 --help` still parses; `--config … --num-runs 2` still overrides (existing `test_config_and_wandb` covers precedence).
6. **Grep proof of decoupling:** `rg -n "argparse.Namespace" src/world_marl/scripts/train_e2e.py` shows Namespace only in `parse_args`'s return type (the entrypoint), not in the core signatures.

## Out of scope (YAGNI / future)

- Hydra dependency, `@hydra.main`, `ConfigStore`, `key=value` CLI, Optuna wiring.
- Encoding `choices=` as `Literal`/`Enum` (a natural Hydra-time hardening step).
- Removing the transitional `config`/`eval_checkpoint` fields (Hydra will own config-loading and eval dispatch).
- `runpod.py`, README, `configs/` changes.

## Self-review notes

- **Spec coverage:** every spec section maps to a task — `config.py`/`from_namespace` (Task 1); core retype + `vars`→`asdict` + `main` (Task 2 steps 1-4); `benchmark_policy` (Task 2 step 5); test conversions + round-trip guard (Task 1 step 1, Task 2 step 6). The extra `evaluate_random_baseline` helper found during excerpt-gathering is included (Task 2 step 2).
- **Frozen decision:** mutable dataclass confirmed correct against the mutation audit — no `dataclasses.replace` needed anywhere.
- **Type consistency:** `run_training`/helpers all take `cfg: TrainConfig`; `_arm_train_args` returns `TrainConfig`; `from_namespace(namespace)` name is consistent across Task 1 and its call sites.
