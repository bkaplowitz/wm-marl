from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from world_marl.world_model_foundation.replay import (
    WorldModelSequenceBatch,
    synthetic_observation_batch,
)

ActionMode = str


def adapter_action_mode(adapter: Any) -> ActionMode:
    action_shape = tuple(getattr(adapter, "action_shape", ()))
    if action_shape == () and getattr(adapter, "action_low", None) is None:
        return "discrete"
    return "continuous"


def make_single_agent_adapter(
    env_name: str,
    *,
    num_envs: int,
    max_cycles: int,
    seed: int,
    brax_backend: str | None = None,
    dmc_workers: int = 1,
) -> Any:
    if env_name.startswith("dmc:"):
        from world_marl.envs.dmc_adapter import DMCVectorAdapter, dmc_env_name

        return DMCVectorAdapter(
            dmc_env_name(env_name),
            num_envs=num_envs,
            max_cycles=max_cycles,
            seed=seed,
            num_workers=dmc_workers,
        )
    if env_name.startswith("brax:"):
        from world_marl.envs.brax_adapter import BraxVectorAdapter, brax_env_name

        return BraxVectorAdapter(
            brax_env_name(env_name),
            num_envs=num_envs,
            max_cycles=max_cycles,
            seed=seed,
            backend=brax_backend,
        )
    if env_name.startswith("gymnax:"):
        from world_marl.envs.gymnax_adapter import GymnaxVectorAdapter, gymnax_env_name

        return GymnaxVectorAdapter(
            gymnax_env_name(env_name),
            num_envs=num_envs,
            max_cycles=max_cycles,
            seed=seed,
        )
    if env_name.startswith("pixels:"):
        from world_marl.envs.pixel_control_adapter import (
            PixelPointMassAdapter,
            pixel_env_name,
        )

        return PixelPointMassAdapter(
            pixel_env_name(env_name),
            num_envs=num_envs,
            max_cycles=max_cycles,
            seed=seed,
        )
    raise ValueError(
        "--env must be formatted as synthetic:<name>, brax:<env>, "
        "gymnax:<env_id>, pixels:<env_id>, or dmc:<domain>/<task>"
    )


def collect_world_model_sequence(
    *,
    env_name: str,
    time_steps: int,
    batch_size: int | None = None,
    observation_shape: tuple[int, ...] | None = None,
    action_dim: int | None = None,
    num_envs: int | None = None,
    max_cycles: int = 1000,
    seed: int = 0,
    brax_backend: str | None = None,
    dmc_workers: int = 1,
) -> WorldModelSequenceBatch:
    if env_name.startswith("synthetic:"):
        if batch_size is None:
            raise ValueError("batch_size is required for synthetic collectors")
        if observation_shape is None:
            raise ValueError("observation_shape is required for synthetic collectors")
        return synthetic_sequence_collector(
            env_name=env_name,
            time_steps=time_steps,
            batch_size=batch_size,
            observation_shape=observation_shape,
            action_dim=4 if action_dim is None else action_dim,
        )

    adapter_num_envs = batch_size if num_envs is None else num_envs
    if adapter_num_envs is None:
        raise ValueError("num_envs or batch_size is required for adapter collectors")

    adapter = make_single_agent_adapter(
        env_name,
        num_envs=adapter_num_envs,
        max_cycles=max_cycles,
        seed=seed,
        brax_backend=brax_backend,
        dmc_workers=dmc_workers,
    )
    try:
        return collect_adapter_sequence(
            adapter,
            env_name=env_name,
            time_steps=time_steps,
            seed=seed,
        )
    finally:
        close = getattr(adapter, "close", None)
        if close is not None:
            close()


def collect_adapter_sequence(
    adapter: Any,
    *,
    env_name: str | None = None,
    time_steps: int,
    seed: int = 0,
) -> WorldModelSequenceBatch:
    if time_steps <= 0:
        raise ValueError("time_steps must be positive")

    rng = np.random.default_rng(seed)
    action_mode = adapter_action_mode(adapter)
    num_envs = int(getattr(adapter, "num_envs"))
    observation_shape = tuple(int(dim) for dim in getattr(adapter, "observation_shape"))
    action_shape = tuple(int(dim) for dim in getattr(adapter, "action_shape", ()))
    action_dim = int(getattr(adapter, "action_dim"))

    observations = np.empty(
        (time_steps, num_envs, *observation_shape), dtype=np.float32
    )
    if action_mode == "discrete":
        actions = np.empty((time_steps, num_envs), dtype=np.int32)
    else:
        actions = np.empty((time_steps, num_envs, action_dim), dtype=np.float32)
    rewards = np.empty((time_steps, num_envs), dtype=np.float32)
    continues = np.empty((time_steps, num_envs), dtype=np.float32)
    is_first = np.zeros((time_steps, num_envs), dtype=bool)
    is_terminal = np.zeros((time_steps, num_envs), dtype=bool)
    is_first[0, :] = True

    current_obs = _squeeze_single_agent_axis(adapter.reset(), num_envs=num_envs)
    for step_index in range(time_steps):
        observations[step_index] = current_obs.reshape((num_envs, *observation_shape))
        sampled_actions = _sample_adapter_actions(adapter, rng, action_mode=action_mode)
        actions[step_index] = _store_actions(
            sampled_actions,
            num_envs=num_envs,
            action_dim=action_dim,
            action_mode=action_mode,
        )
        step = adapter.step(sampled_actions)
        step_rewards = _squeeze_single_agent_axis(step.rewards, num_envs=num_envs)
        step_dones = _squeeze_single_agent_axis(step.dones, num_envs=num_envs)
        rewards[step_index] = step_rewards.reshape((num_envs,)).astype(np.float32)
        terminal = step_dones.reshape((num_envs,)).astype(np.float32) > 0.5
        is_terminal[step_index] = terminal
        continues[step_index] = 1.0 - terminal.astype(np.float32)
        current_obs = _squeeze_single_agent_axis(step.observations, num_envs=num_envs)

    return WorldModelSequenceBatch(
        observations=observations,
        actions=actions,
        rewards=rewards,
        continues=continues,
        is_first=is_first,
        is_terminal=is_terminal,
        metadata={
            "collector": "adapter_sequence_collector",
            "env": env_name or getattr(adapter, "substrate", "adapter"),
            "action_mode": action_mode,
            "observation_shape": observation_shape,
            "raw_observation_shape": tuple(
                getattr(adapter, "raw_observation_shape", observation_shape)
            ),
            "action_shape": action_shape,
            "action_dim": action_dim,
            "num_envs": num_envs,
        },
    )


def _sample_adapter_actions(
    adapter: Any,
    rng: np.random.Generator,
    *,
    action_mode: ActionMode,
) -> np.ndarray:
    sample_actions = getattr(adapter, "sample_actions", None)
    if sample_actions is not None:
        return np.asarray(sample_actions(rng))

    num_envs = int(getattr(adapter, "num_envs"))
    action_dim = int(getattr(adapter, "action_dim"))
    if action_mode == "discrete":
        return rng.integers(
            low=0,
            high=action_dim,
            size=(num_envs, 1),
            dtype=np.int32,
        )

    low = np.asarray(getattr(adapter, "action_low", -1.0), dtype=np.float32).reshape(
        (action_dim,)
    )
    high = np.asarray(getattr(adapter, "action_high", 1.0), dtype=np.float32).reshape(
        (action_dim,)
    )
    return rng.uniform(low=low, high=high, size=(num_envs, action_dim)).astype(
        np.float32
    )


def _store_actions(
    actions: np.ndarray,
    *,
    num_envs: int,
    action_dim: int,
    action_mode: ActionMode,
) -> np.ndarray:
    if action_mode == "discrete":
        return np.asarray(actions, dtype=np.int32).reshape((num_envs,))
    return np.asarray(actions, dtype=np.float32).reshape((num_envs, action_dim))


def _squeeze_single_agent_axis(array: Any, *, num_envs: int) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim >= 2 and values.shape[0] == num_envs and values.shape[1] == 1:
        values = values[:, 0, ...]
    return values


def synthetic_sequence_collector(
    *,
    env_name: str,
    time_steps: int,
    batch_size: int,
    observation_shape: tuple[int, ...],
    action_dim: int,
) -> WorldModelSequenceBatch:
    batch = synthetic_observation_batch(
        time_steps=time_steps,
        batch_size=batch_size,
        observation_shape=observation_shape,
        action_dim=action_dim,
    )
    metadata = dict(batch.metadata)
    metadata.update({"collector": "synthetic_sequence_collector", "env": env_name})
    return WorldModelSequenceBatch(
        observations=batch.observations,
        actions=batch.actions,
        rewards=batch.rewards,
        continues=batch.continues,
        is_first=batch.is_first,
        is_terminal=batch.is_terminal,
        metadata=metadata,
    )


def write_json_artifact(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    return path


def write_jsonl_metrics(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    return path
