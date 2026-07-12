"""Locate the first nondeterministic stage in the DMC JEPA pipeline."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from world_marl.checkpointing import load_metadata, load_params
from world_marl.envs.dmc_adapter import DMCVectorAdapter, dmc_env_name
from world_marl.jepa.models import JepaConfig
from world_marl.jepa.replay import SequenceReplayBuffer
from world_marl.jepa.reproducibility import (
    fingerprint_arrays,
    fingerprint_pytree,
)
from world_marl.jepa.training import (
    continuous_policy_train_step,
    create_jepa_train_state,
    reset_policy_heads,
    train_model_step,
)
from world_marl.logging import to_jsonable
from world_marl.scripts.train_dmc_jepa import (
    _configure_deterministic_compute,
    _evaluate_continuous_policy,
)


DEFAULT_HASH_UPDATES = (0, 1, 10, 100, 500, 1000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "evaluation", "environment", "world-model", "actor"),
        default="all",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="JEPA checkpoint directory containing checkpoint.msgpack.",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        help="Saved SequenceReplayBuffer NPZ for model and actor diagnostics.",
    )
    parser.add_argument("--env", default=None, help="Override checkpoint DMC env.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-seed", type=int, default=9_000_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-num-envs", type=int, default=1)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--environment-steps", type=int, default=2000)
    parser.add_argument(
        "--environment-cases",
        default="1:1,16:1,16:16",
        help="Comma-separated num_envs:num_workers cases.",
    )
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--hash-updates", default="0,1,10,100,500,1000")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--policy-batch-size", type=int, default=256)
    parser.add_argument("--chunk-length", type=int, default=64)
    parser.add_argument("--model-horizon", type=int, default=5)
    parser.add_argument("--imag-horizon", type=int, default=5)
    parser.add_argument("--actor-entropy-coef", type=float, default=3e-4)
    parser.add_argument("--value-clip", type=float, default=100.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--deterministic-compute",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_deterministic_compute(args.deterministic_compute)
    stages = (
        ("evaluation", "environment", "world-model", "actor")
        if args.stage == "all"
        else (args.stage,)
    )
    if any(stage in {"evaluation", "world-model", "actor"} for stage in stages):
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for this diagnostic stage")
    if any(stage in {"world-model", "actor"} for stage in stages):
        if args.replay is None:
            raise SystemExit("--replay is required for this diagnostic stage")

    checkpoint = _checkpoint_dir(args.checkpoint) if args.checkpoint else None
    metadata = load_metadata(checkpoint) if checkpoint else {}
    env = args.env or metadata.get("env") or "dmc:reacher/easy"
    if not env.startswith("dmc:"):
        raise SystemExit("the determinism ladder currently supports DMC only")
    config = JepaConfig(**metadata["jepa_config"]) if metadata else None
    hash_updates = _parse_hash_updates(args.hash_updates, args.updates)
    report: dict[str, Any] = {
        "env": env,
        "seed": args.seed,
        "deterministic_compute": args.deterministic_compute,
        "jax_devices": [str(device) for device in jax.devices()],
        "stages": {},
    }
    if "evaluation" in stages:
        assert checkpoint is not None and config is not None
        report["stages"]["evaluation"] = evaluation_determinism(
            checkpoint,
            config,
            env=env,
            seed=args.seed,
            eval_seed=args.eval_seed,
            episodes=args.eval_episodes,
            num_envs=args.eval_num_envs,
            max_cycles=args.max_cycles,
        )
    if "environment" in stages:
        report["stages"]["environment"] = environment_determinism(
            env,
            seed=args.seed,
            action_seed=args.seed + 100,
            steps=args.environment_steps,
            max_cycles=args.max_cycles,
            cases=_parse_environment_cases(args.environment_cases),
        )
    if "world-model" in stages:
        assert config is not None and args.replay is not None
        report["stages"]["world_model"] = world_model_determinism(
            args.replay,
            config,
            seed=args.seed,
            updates=args.updates,
            hash_updates=hash_updates,
            batch_size=args.batch_size,
            chunk_length=args.chunk_length,
            model_horizon=args.model_horizon,
        )
    if "actor" in stages:
        assert checkpoint is not None and config is not None and args.replay is not None
        report["stages"]["actor"] = actor_determinism(
            checkpoint,
            args.replay,
            config,
            env=env,
            seed=args.seed,
            updates=args.updates,
            hash_updates=hash_updates,
            batch_size=args.policy_batch_size,
            imag_horizon=args.imag_horizon,
            actor_entropy_coef=args.actor_entropy_coef,
            value_clip=args.value_clip,
        )
    report["passed"] = all(
        bool(stage_report["passed"]) for stage_report in report["stages"].values()
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(to_jsonable(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(to_jsonable(report), indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


def evaluation_determinism(
    checkpoint: Path,
    config: JepaConfig,
    *,
    env: str,
    seed: int,
    eval_seed: int,
    episodes: int,
    num_envs: int,
    max_cycles: int,
) -> dict[str, Any]:
    state = _load_checkpoint_state(checkpoint, config, seed=seed)
    adapter = DMCVectorAdapter(
        dmc_env_name(env),
        num_envs=num_envs,
        max_cycles=max_cycles,
        seed=eval_seed,
        num_workers=1,
    )
    try:
        action_low = jnp.asarray(adapter.action_low, dtype=jnp.float32)
        action_high = jnp.asarray(adapter.action_high, dtype=jnp.float32)
    finally:
        adapter.close()
    eval_args = SimpleNamespace(
        env=env,
        num_envs=num_envs,
        env_workers=1,
        max_cycles=max_cycles,
        brax_backend=None,
        policy_eval_episodes=episodes,
        quiet=True,
        wandb_video_size=64,
        wandb_video_camera=0,
        wandb_video_frame_stride=1,
        wandb_video_fps=20,
        policy_failure_return_threshold=100.0,
        policy_success_return_threshold=900.0,
    )
    runs = [
        _evaluate_continuous_policy(
            eval_args,
            state,
            config,
            seed=eval_seed,
            num_envs=num_envs,
            episodes=episodes,
            action_low=action_low,
            action_high=action_high,
            desc=f"determinism evaluation {run_index + 1}",
            stochastic_actions=False,
        )
        for run_index in range(2)
    ]
    returns = [np.asarray(run["returns"], dtype=np.float32) for run in runs]
    lengths = [np.asarray(run["lengths"], dtype=np.int32) for run in runs]
    returns_equal = np.array_equal(returns[0], returns[1])
    lengths_equal = np.array_equal(lengths[0], lengths[1])
    return {
        "passed": bool(returns_equal and lengths_equal),
        "returns_equal": bool(returns_equal),
        "lengths_equal": bool(lengths_equal),
        "first_return_mismatch": _first_mismatch(returns[0], returns[1]),
        "first_length_mismatch": _first_mismatch(lengths[0], lengths[1]),
        "runs": [
            {
                "returns": returns[index].tolist(),
                "lengths": lengths[index].tolist(),
                "sha256": fingerprint_arrays(
                    {"returns": returns[index], "lengths": lengths[index]}
                ),
            }
            for index in range(2)
        ],
    }


def environment_determinism(
    env: str,
    *,
    seed: int,
    action_seed: int,
    steps: int,
    max_cycles: int,
    cases: tuple[tuple[int, int], ...],
) -> dict[str, Any]:
    case_reports = []
    for num_envs, num_workers in cases:
        probe = DMCVectorAdapter(
            dmc_env_name(env),
            num_envs=num_envs,
            max_cycles=max_cycles,
            seed=seed,
            num_workers=num_workers,
        )
        try:
            action_low = probe.action_low.copy()
            action_high = probe.action_high.copy()
            action_dim = probe.action_dim
        finally:
            probe.close()
        actions = (
            np.random.default_rng(action_seed)
            .uniform(
                low=action_low,
                high=action_high,
                size=(steps, num_envs, action_dim),
            )
            .astype(np.float32)
        )
        case_reports.append(
            _compare_environment_pair(
                env,
                actions,
                seed=seed,
                max_cycles=max_cycles,
                num_envs=num_envs,
                num_workers=num_workers,
            )
        )
    return {
        "passed": all(case["passed"] for case in case_reports),
        "cases": case_reports,
    }


def _compare_environment_pair(
    env: str,
    actions: np.ndarray,
    *,
    seed: int,
    max_cycles: int,
    num_envs: int,
    num_workers: int,
) -> dict[str, Any]:
    adapters = [
        DMCVectorAdapter(
            dmc_env_name(env),
            num_envs=num_envs,
            max_cycles=max_cycles,
            seed=seed,
            num_workers=num_workers,
        )
        for _ in range(2)
    ]
    traces = [
        {"observations": [], "rewards": [], "dones": [], "returns": [], "lengths": []}
        for _ in range(2)
    ]
    mismatch = None
    try:
        reset_observations = [adapter.reset() for adapter in adapters]
        for index in range(2):
            traces[index]["observations"].append(reset_observations[index])
        if not np.array_equal(reset_observations[0], reset_observations[1]):
            mismatch = _array_mismatch("reset_observations", 0, *reset_observations)
        for step_index, action in enumerate(actions, start=1):
            if mismatch is not None:
                break
            results = [adapter.step(action[:, None, :]) for adapter in adapters]
            components = {
                "observations": [result.observations for result in results],
                "rewards": [result.rewards for result in results],
                "dones": [result.dones for result in results],
                "completed_returns": [
                    np.asarray(result.completed_returns, dtype=np.float32)
                    for result in results
                ],
                "completed_lengths": [
                    np.asarray(result.completed_lengths, dtype=np.int32)
                    for result in results
                ],
            }
            for name, values in components.items():
                if not np.array_equal(values[0], values[1]):
                    mismatch = _array_mismatch(name, step_index, *values)
                    break
            for index, result in enumerate(results):
                traces[index]["observations"].append(result.observations)
                traces[index]["rewards"].append(result.rewards)
                traces[index]["dones"].append(result.dones)
                traces[index]["returns"].extend(result.completed_returns)
                traces[index]["lengths"].extend(result.completed_lengths)
    finally:
        for adapter in adapters:
            adapter.close()

    trace_hashes = []
    for trace in traces:
        trace_hashes.append(
            fingerprint_arrays(
                {
                    "observations": np.asarray(trace["observations"], dtype=np.float32),
                    "rewards": np.asarray(trace["rewards"], dtype=np.float32),
                    "dones": np.asarray(trace["dones"], dtype=np.float32),
                    "returns": np.asarray(trace["returns"], dtype=np.float32),
                    "lengths": np.asarray(trace["lengths"], dtype=np.int32),
                }
            )
        )
    return {
        "num_envs": num_envs,
        "num_workers": num_workers,
        "steps": int(actions.shape[0]),
        "passed": mismatch is None and trace_hashes[0] == trace_hashes[1],
        "first_mismatch": mismatch,
        "trace_sha256": trace_hashes,
    }


def world_model_determinism(
    replay_path: Path,
    source_config: JepaConfig,
    *,
    seed: int,
    updates: int,
    hash_updates: tuple[int, ...],
    batch_size: int,
    chunk_length: int,
    model_horizon: int,
) -> dict[str, Any]:
    replay = SequenceReplayBuffer.load_npz(replay_path)
    config = dataclasses.replace(
        source_config,
        max_horizon=model_horizon,
        dynamics_ensemble_size=1,
        regularizer="none",
        regularizer_weight=0.0,
    )
    index_rng = np.random.default_rng(np.random.SeedSequence([seed, 14]))
    indices = [
        replay.sample_indices(
            index_rng,
            batch_size=batch_size,
            chunk_length=chunk_length,
            max_horizon=model_horizon,
        )
        for _ in range(updates)
    ]
    init_key = jax.random.fold_in(jax.random.PRNGKey(seed), 1)
    root_key = jax.random.fold_in(jax.random.PRNGKey(seed), 2)
    states = [create_jepa_train_state(init_key, config) for _ in range(2)]
    records = [_hash_record(0, states, mode="model")]
    first_mismatch = None
    for update_index, (starts, envs) in enumerate(indices, start=1):
        batch = replay.sample_from_indices(
            starts,
            envs,
            chunk_length=chunk_length,
            max_horizon=model_horizon,
        )
        update_key = jax.random.fold_in(root_key, update_index)
        states = [
            train_model_step(
                state,
                update_key,
                batch,
                config,
                chunk_length=chunk_length,
                control="none",
                freeze_encoder=False,
            )[0]
            for state in states
        ]
        if update_index in hash_updates:
            record = _hash_record(update_index, states, mode="model")
            records.append(record)
            if not record["matched"] and first_mismatch is None:
                first_mismatch = update_index
    return {
        "passed": first_mismatch is None,
        "first_mismatch_update": first_mismatch,
        "replay_sha256": replay.fingerprint(),
        "pre_generated_batches": updates,
        "config": dataclasses.asdict(config),
        "checkpoints": records,
    }


def actor_determinism(
    checkpoint: Path,
    replay_path: Path,
    config: JepaConfig,
    *,
    env: str,
    seed: int,
    updates: int,
    hash_updates: tuple[int, ...],
    batch_size: int,
    imag_horizon: int,
    actor_entropy_coef: float,
    value_clip: float,
) -> dict[str, Any]:
    replay = SequenceReplayBuffer.load_npz(replay_path)
    index_rng = np.random.default_rng(np.random.SeedSequence([seed, 15]))
    indices = [
        replay.sample_indices(
            index_rng,
            batch_size=batch_size,
            chunk_length=config.context_window,
            max_horizon=1,
        )
        for _ in range(updates)
    ]
    base = _load_checkpoint_state(checkpoint, config, seed=seed)
    reset_key = jax.random.fold_in(jax.random.PRNGKey(seed), 30)
    base = reset_policy_heads(base, reset_key, config)
    states = [base, base]
    adapter = DMCVectorAdapter(
        dmc_env_name(env),
        num_envs=1,
        seed=seed,
        num_workers=1,
    )
    try:
        action_low = jnp.asarray(adapter.action_low, dtype=jnp.float32)
        action_high = jnp.asarray(adapter.action_high, dtype=jnp.float32)
    finally:
        adapter.close()
    root_key = jax.random.fold_in(jax.random.PRNGKey(seed), 3)
    records = [_hash_record(0, states, mode="actor")]
    first_mismatch = None
    for update_index, (starts, envs) in enumerate(indices, start=1):
        batch = replay.sample_from_indices(
            starts,
            envs,
            chunk_length=config.context_window,
            max_horizon=1,
        )
        start_observations = batch.observations[:, : config.context_window]
        start_actions = batch.actions[:, : config.context_window]
        update_key = jax.random.fold_in(root_key, update_index)
        states = [
            continuous_policy_train_step(
                state,
                update_key,
                start_observations,
                config,
                action_low,
                action_high,
                imag_horizon=imag_horizon,
                policy_return_mode="reward-only",
                policy_actor_baseline="none",
                policy_return_normalization="none",
                policy_actor_cvar_fraction=1.0,
                policy_actor_cvar_coef=0.0,
                policy_gradient_mode="reinforce",
                value_clip=value_clip,
                start_actions=start_actions,
                actor_entropy_coef=actor_entropy_coef,
                target_critic_ema_decay=0.0,
                real_critic_loss_enabled=False,
                slow_value_regularization_coef=0.0,
            )[0]
            for state in states
        ]
        if update_index in hash_updates:
            record = _hash_record(update_index, states, mode="actor")
            records.append(record)
            if not record["matched"] and first_mismatch is None:
                first_mismatch = update_index
    return {
        "passed": first_mismatch is None,
        "first_mismatch_update": first_mismatch,
        "replay_sha256": replay.fingerprint(),
        "pre_generated_batches": updates,
        "imag_horizon": imag_horizon,
        "checkpoints": records,
    }


def _load_checkpoint_state(
    checkpoint: Path,
    config: JepaConfig,
    *,
    seed: int,
):
    state = create_jepa_train_state(jax.random.PRNGKey(seed), config)
    params = load_params(checkpoint / "checkpoint.msgpack", state.params)
    return state.replace(params=params, target_critic_params=params)


def _hash_record(update: int, states: list[Any], *, mode: str) -> dict[str, Any]:
    hashes = [_state_hashes(state, mode=mode) for state in states]
    return {
        "update": update,
        "matched": hashes[0] == hashes[1],
        "runs": hashes,
    }


def _state_hashes(state, *, mode: str) -> dict[str, str]:
    if mode == "model":
        return {
            "params_sha256": fingerprint_pytree(state.params),
            "model_optimizer_sha256": fingerprint_pytree(state.model_opt_state),
        }
    return {
        "actor_sha256": fingerprint_pytree(state.params["actor_head"]),
        "critic_sha256": fingerprint_pytree(state.params["value_head"]),
        "actor_optimizer_sha256": fingerprint_pytree(state.actor_opt_state),
        "critic_optimizer_sha256": fingerprint_pytree(state.critic_opt_state),
        "target_critic_sha256": fingerprint_pytree(
            state.target_critic_params["value_head"]
        ),
        "return_range_sha256": fingerprint_pytree(
            (state.return_range_ema, state.return_range_initialized)
        ),
    }


def _checkpoint_dir(path: Path) -> Path:
    checkpoint = path.parent if path.name == "checkpoint.msgpack" else path
    if not (checkpoint / "checkpoint.msgpack").is_file():
        raise SystemExit(f"checkpoint.msgpack not found in {checkpoint}")
    if not (checkpoint / "metadata.json").is_file():
        raise SystemExit(f"metadata.json not found in {checkpoint}")
    return checkpoint


def _parse_hash_updates(raw: str, updates: int) -> tuple[int, ...]:
    requested = {int(value.strip()) for value in raw.split(",") if value.strip()}
    requested.add(0)
    requested.add(updates)
    invalid = sorted(value for value in requested if value < 0 or value > updates)
    if invalid:
        raise SystemExit(f"hash updates outside [0, {updates}]: {invalid}")
    return tuple(sorted(requested))


def _parse_environment_cases(raw: str) -> tuple[tuple[int, int], ...]:
    cases = []
    for item in raw.split(","):
        num_envs, num_workers = (int(value) for value in item.split(":", 1))
        if num_envs < 1 or num_workers < 1:
            raise SystemExit("environment cases must contain positive integers")
        cases.append((num_envs, num_workers))
    return tuple(cases)


def _first_mismatch(left: np.ndarray, right: np.ndarray) -> int | None:
    if left.shape != right.shape:
        return 0
    differing = np.flatnonzero(left.reshape((-1,)) != right.reshape((-1,)))
    return int(differing[0]) if differing.size else None


def _array_mismatch(
    component: str,
    step: int,
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, Any]:
    left = np.asarray(left)
    right = np.asarray(right)
    max_abs_diff = None
    if left.shape == right.shape and left.size and np.issubdtype(left.dtype, np.number):
        max_abs_diff = float(
            np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))
        )
    return {
        "component": component,
        "step": step,
        "left_shape": left.shape,
        "right_shape": right.shape,
        "first_flat_index": _first_mismatch(left, right),
        "max_abs_diff": max_abs_diff,
    }


if __name__ == "__main__":
    main()
