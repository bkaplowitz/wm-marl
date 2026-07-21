# Design: `train_e2e` entrypoint/core split via a `TrainConfig` dataclass

**Date:** 2026-07-01
**Branch:** `feat-train-config-split` (off `feat-wandb-omegaconf`)
**Status:** design, pending user review

## Goal

Separate the *thin CLI entrypoint* of `train_e2e.py` from a *pure training core* by
introducing a `TrainConfig` dataclass as the core's typed input, replacing
`argparse.Namespace`. This makes the training core callable from scripts and sweeps
without argv gymnastics, decouples `benchmark_policy`'s reuse of the core from argparse,
and pre-builds the Hydra structured-config schema so a later Hydra migration is
"register the dataclass + swap the entrypoint" with no core changes.

This branch does the split **only** — no Hydra dependency, no CLI syntax change.

## Non-goals (out of scope for this branch)

- No Hydra dependency, `@hydra.main`, `ConfigStore`, or `key=value` CLI.
- No change to the argparse CLI surface, flag names, or `--config`/`--wandb` behavior.
- No change to `runpod.py` (it emits `--flag value` command strings; the CLI is untouched).
- No Optuna wiring (the dataclass is the future sweep surface; the wiring is a later branch).
- No README change (the CLI surface is unchanged, so there is nothing new to document).

## Current boundary (what couples the core to argparse today)

- `run_training(args: argparse.Namespace, *, run_dir, name, run_index, control)`
  (`train_e2e.py:590`) is the core. It is called from **two** places:
  - `main()` (`train_e2e.py:1090`, `:1102`), the CLI driver;
  - `benchmark_policy.run_arm()` (`benchmark_policy.py:309`), which feeds it a Namespace
    produced by a `sys.argv`-swap (`_parse_train_args` → `train_e2e.parse_args()`).
- Helpers that also take the Namespace: `algorithm_config_from_args(args, control)`
  (`:238`), `evaluate_checkpoint_mode(args)` (`:402`), `_make_training_adapter(args, seed)`.
- `vars(args)` is used twice inside `run_training` (config.json write, wandb `config=`).

### Mutation audit (decides `frozen`)

`rg` for writes to the *train* Namespace found real mutations:

- `train_e2e.py:407–413` — `evaluate_checkpoint_mode` backfills `substrate`,
  `observation_size`, `include_observation_scalars`, `append_agent_id` from checkpoint
  metadata.
- `benchmark_policy.py:116` — `_arm_train_args` sets `parsed.out_dir = str(out_dir)`.

(`benchmark_policy.py:67` writes `args.train_args`, but that is the benchmark's *own*
top-level Namespace, not the train one — irrelevant.)

**Consequence:** `TrainConfig` is a plain **mutable** `@dataclass`, *not* `frozen`. A frozen
dataclass would raise `FrozenInstanceError` on the two mutation sites. `frozen` also buys
nothing for the Hydra path — Hydra builds a mutable `DictConfig` from the schema regardless.
Mutable keeps both mutation sites working unchanged and is fully Hydra-compatible.

## Design

### New module: `src/world_marl/config.py`

A plain dataclass whose fields are **exactly** the 41 argparse dests (same names, same
defaults). Because every field has a default (argparse supplies one for all 41), there is no
"non-default after default" ordering problem.

```python
from __future__ import annotations
import argparse
from dataclasses import dataclass

@dataclass
class TrainConfig:
    # entrypoint/logging layer (transitional — Hydra will own config-loading later)
    config: str | None = None
    wandb: bool = False
    wandb_project: str = "world-marl"
    eval_checkpoint: str | None = None

    # experiment
    algorithm: str = "ippo"                 # choices: ippo, mappo
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
    negative_control: str = "freeze-policy" # choices: none, freeze-policy, shuffle-rewards, zero-advantages

    # world model
    prefit_world_model: bool = False
    wm_random_rollouts: int = 1
    wm_initial_rollouts: int = 1
    wm_fit_steps: int = 10_000
    wm_learning_rate: float = 3e-4
    wm_hidden_dim: int = 256
    wm_integration_steps: int = 10
    wm_policy_warmup_updates: int = 0
    wm_flow_type: str = "linear"            # choices: gaussian, linear, discrete, transformer
    wm_num_categories: int = 9

    # ppo
    learning_rate: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 4
    activation: str = "relu"                # choices: relu, tanh

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> "TrainConfig":
        return cls(**vars(ns))
```

**Why `from_namespace = cls(**vars(ns))` is the right conversion:** it is self-checking.
If the dataclass and the argparse dests ever drift (a renamed, missing, or extra field), the
construction raises `TypeError` on the first call — drift cannot pass silently. This is a
stronger guarantee than a hand-written field-by-field copy.

The `choices=` sets are documented as comments, not re-encoded as `Enum`/`Literal` in this
branch (keeping the dataclass a faithful, minimal mirror). The existing behavior — argparse
enforces `choices` on CLI-supplied values but not on `--config` YAML defaults — is unchanged
and already accepted in the prior wandb/OmegaConf design.

### Entrypoint vs. core boundary

**Entrypoint (thin, argv/argparse-aware) — unchanged in behavior:**
- `parse_args()` — the argparse definition, the OmegaConf `--config` merge, and the post-parse
  numeric validation (`train_e2e.py:220–234`) all stay here. Validation is a CLI-layer
  concern and continues to run on the Namespace *before* conversion.
- `main()` — parses argv, then `cfg = TrainConfig.from_namespace(args)`, then drives the run
  loop / control run / `evaluate_checkpoint_mode` with `cfg`.

**Core (pure, config-driven, no argv/argparse):**
- `run_training(cfg: TrainConfig, *, run_dir, name, run_index, control)`
- `algorithm_config_from_args(cfg: TrainConfig, control=None)`
- `evaluate_checkpoint_mode(cfg: TrainConfig)`
- `_make_training_adapter(cfg: TrainConfig, seed)`

### Naming

The core parameter is renamed `args` → `cfg` in `run_training` and the retyped helpers
(user decision: prefer clarity over minimal diff). This is a mechanical `args.` → `cfg.`
sweep inside those function bodies. The inner local `config = algorithm_config_from_args(...)`
(`train_e2e.py:613`) is a *different* name (`cfg` ≠ `config`) — no collision, left as-is.

The helper **function name** `algorithm_config_from_args` is kept (not renamed to
`_from_cfg`): it is imported by name in `tests/test_train_e2e_prefit.py`, so renaming is pure
churn with no functional benefit. The `cfg` parameter type annotation already conveys intent.

### `vars(args)` → `dataclasses.asdict(cfg)`

The two `vars(args)` sites inside `run_training` (config.json, wandb `config=`) become
`dataclasses.asdict(cfg)`. This is deliberate: `vars()` works on both a Namespace and a
non-slots dataclass, but `asdict()` works on *only* the dataclass — so switching to `asdict`
turns "core takes a `TrainConfig`" from a comment into an enforced contract. It is also why
the one direct-Namespace test caller (below) must be converted.

### `benchmark_policy.py`

`_arm_train_args` currently returns an `argparse.Namespace` (after `parsed.out_dir = ...`).
It will instead return a `TrainConfig`: set `out_dir` (on the Namespace before conversion, or
on the mutable `cfg` after — either works) and return `TrainConfig.from_namespace(parsed)`.
`run_arm` is otherwise unchanged: `train_args.num_runs` / `.min_improvement` still resolve as
attributes, and `run_training(train_args, ...)` now receives a `TrainConfig`. The argv-swap
becomes a pure "parse the passthrough flags" detail with no coupling into the core.

## Test changes

- **`tests/test_train_e2e_coins_e2e.py`** (`:76`) calls `train_e2e.run_training(args, ...)`
  directly with a full Namespace. Convert: `cfg = TrainConfig.from_namespace(args)` and pass
  `cfg`. Required, because `run_training` now calls `asdict(cfg)`, which would raise on a raw
  Namespace.
- **`tests/test_train_e2e_prefit.py`** builds a *partial* Namespace in `_args()` and passes it
  to `algorithm_config_from_args` and `_make_training_adapter`. Migrate `_args()` to build a
  partial `TrainConfig(...)` (relying on the dataclass defaults for the fields it omits). These
  helpers are read-only, so a Namespace would still duck-type at runtime, but migrating keeps
  the tests type-honest and exercises the defaults.
- **New test** (in `tests/test_config_and_wandb.py` or a new `tests/test_train_config.py`):
  round-trip fidelity — with `sys.argv` monkeypatched to `["prog"]` (no flags),
  `dataclasses.asdict(TrainConfig.from_namespace(parse_args())) == vars(parse_args())`. This
  guards against argparse↔dataclass drift. (`parse_args()` has no required args, so empty argv
  does not `SystemExit`.)

## Files touched

| File | Change |
|------|--------|
| `src/world_marl/config.py` | **new** — `TrainConfig` + `from_namespace` |
| `src/world_marl/scripts/train_e2e.py` | retype core fns to `TrainConfig`; `args`→`cfg` in bodies; `main()` builds cfg; 2× `vars`→`asdict`; import `TrainConfig` |
| `src/world_marl/scripts/benchmark_policy.py` | `_arm_train_args` returns `TrainConfig`; import `TrainConfig` |
| `tests/test_train_e2e_coins_e2e.py` | wrap the direct `run_training` call with `from_namespace` |
| `tests/test_train_e2e_prefit.py` | `_args()` builds a partial `TrainConfig` |
| `tests/test_train_config.py` (or existing) | **new** round-trip test |

`runpod.py`, README, `configs/`, and the argparse CLI are **unchanged**.

## Known transitional seams (revisit at the real Hydra migration, not now)

- `config` and `eval_checkpoint` as `TrainConfig` fields are awkward for a Hydra schema
  (Hydra owns config-loading and the eval-mode dispatch). Kept now so `from_namespace` is a
  trivial faithful mirror; flagged for cleanup when Hydra actually lands.
- `choices=` are comments, not `Literal`/`Enum`. Encoding them into the type is a natural
  Hydra-time hardening step (Hydra/OmegaConf can validate enums), deferred to keep this branch
  a pure structural split.

## Verification

Run inside the `feat-train-config-split` worktree:

1. `uv run ruff format` + `uv run ruff check` on changed files.
2. `uv run pytest tests/test_train_config.py tests/test_config_and_wandb.py -q` — the new
   round-trip guard + the existing config/wandb tests.
3. `uv run pytest tests/test_train_e2e_prefit.py tests/test_runpod.py -q` — regression proof
   that the retyped helpers and the untouched CLI contract still work for existing callers.
4. `uv run pytest tests/test_train_e2e_coins_e2e.py -q` — exercises the converted direct
   `run_training(cfg, ...)` caller end-to-end (this is a real short training run; it is an
   existing test, run per the no-unilateral-heavy-runs rule — confirm it is acceptable to run,
   or scope it out).
5. Parse smoke: `--config configs/train_e2e.example.yaml --num-runs 1` still parses and the
   override precedence is unchanged.

## Why this is the Hydra/Optuna groundwork

`TrainConfig` is precisely a Hydra *structured config* node. The later migration becomes:
`ConfigStore.instance().store(name="train", node=TrainConfig)` + an `@hydra.main` entrypoint
that yields the same dataclass (as a `DictConfig`); the core — already consuming `TrainConfig`
— is untouched. Optuna plugs into Hydra's sweeper over these same 41 typed fields. Doing the
split now, as a plain dataclass, delivers the decoupling immediately and makes the Hydra step
additive rather than invasive.
