from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from world_marl.world_model_foundation.preprocess import normalize_observations
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
    image_size: int = 64,
    dmc_camera_id: int = 0,
) -> Any:
    if env_name.startswith("dmc-pixels:"):
        from world_marl.envs.dmc_pixel_adapter import (
            DMCPixelAdapter,
            dmc_pixel_env_name,
        )

        return DMCPixelAdapter(
            dmc_pixel_env_name(env_name),
            num_envs=num_envs,
            max_cycles=max_cycles,
            seed=seed,
            image_size=image_size,
            camera_id=dmc_camera_id,
            num_workers=dmc_workers,
        )
    if env_name.startswith("dmc:"):
        from world_marl.envs.playground_dmc_adapter import PlaygroundDMCAdapter

        return PlaygroundDMCAdapter(
            env_name.split(":", 1)[1],
            num_envs=num_envs,
            max_cycles=max_cycles,
            seed=seed,
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
        "gymnax:<env_id>, pixels:<env_id>, dmc:<domain>/<task>, or "
        "dmc-pixels:<domain>/<task>"
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
    image_size: int = 64,
    dmc_camera_id: int = 0,
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
        image_size=image_size,
        dmc_camera_id=dmc_camera_id,
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

    action_mode = adapter_action_mode(adapter)
    num_envs = int(getattr(adapter, "num_envs"))
    observation_shape = tuple(int(dim) for dim in getattr(adapter, "observation_shape"))
    action_shape = tuple(int(dim) for dim in getattr(adapter, "action_shape", ()))
    action_dim = int(getattr(adapter, "action_dim"))

    current_obs = _squeeze_single_agent_axis(adapter.reset(), num_envs=num_envs)
    scan_random_sequence = getattr(adapter, "scan_random_sequence", None)
    if scan_random_sequence is None:
        raise RuntimeError(
            f"adapter {getattr(adapter, 'substrate', 'unknown')!r} must implement "
            "scan_random_sequence; host-loop collection is not supported"
        )
    import jax

    scanned = scan_random_sequence(
        time_steps,
        key=jax.random.PRNGKey(seed),
        observations=current_obs,
    )
    observations, actions, rewards, is_terminal, is_last = jax.device_get(scanned)
    observations = np.asarray(observations).reshape(
        (time_steps, num_envs, *observation_shape)
    )
    action_dtype = np.int32 if action_mode == "discrete" else np.float32
    action_suffix = () if action_mode == "discrete" else (action_dim,)
    actions = np.asarray(actions, dtype=action_dtype).reshape(
        (time_steps, num_envs, *action_suffix)
    )
    rewards = np.asarray(rewards, dtype=np.float32).reshape((time_steps, num_envs))
    is_terminal = np.asarray(is_terminal, dtype=bool).reshape((time_steps, num_envs))
    is_last = np.asarray(is_last, dtype=bool).reshape((time_steps, num_envs))
    if np.any(is_terminal & ~is_last):
        raise ValueError("terminal records must also be last records")
    continues = 1.0 - is_terminal.astype(np.float32)
    is_first = np.zeros((time_steps, num_envs), dtype=bool)
    is_first[0] = True
    is_first[1:] = is_last[:-1]
    collection_execution = "jax_scan"

    if len(observation_shape) == 3:
        if observations.dtype == np.uint8:
            observations = normalize_observations(observations)
        else:
            observations = np.asarray(observations, dtype=np.float32)
            if not np.all(np.isfinite(observations)):
                raise ValueError("pixel observations must be finite")
            if np.any(observations < 0.0) or np.any(observations > 1.0):
                raise ValueError(
                    "floating-point pixel observations must already be in [0, 1]"
                )
    else:
        observations = np.asarray(observations, dtype=np.float32)

    environment_name = env_name or getattr(adapter, "substrate", "adapter")
    environment_metadata = dict(getattr(adapter, "environment_metadata", {}))
    namespace = environment_name.split(":", 1)[0]
    inferred_backends = {
        "brax": "brax",
        "dmc": "mujoco_playground",
        "dmc-pixels": "dm_control",
        "gymnax": "gymnax",
        "pixels": "synthetic",
        "synthetic": "synthetic",
    }
    environment_metadata.setdefault(
        "environment_backend", inferred_backends.get(namespace, "unknown")
    )
    environment_metadata.setdefault(
        "observation_mode", "pixels" if len(observation_shape) == 3 else "vector"
    )
    is_real_environment = environment_metadata["environment_backend"] not in {
        "synthetic",
        "unknown",
    }
    metadata = {
        "collector": "adapter_sequence_collector",
        "env": environment_name,
        "action_mode": action_mode,
        "observation_shape": observation_shape,
        "raw_observation_shape": tuple(
            getattr(adapter, "raw_observation_shape", observation_shape)
        ),
        "action_shape": action_shape,
        "action_dim": action_dim,
        "action_low": (
            np.asarray(getattr(adapter, "action_low"), dtype=np.float32).tolist()
            if getattr(adapter, "action_low", None) is not None
            else None
        ),
        "action_high": (
            np.asarray(getattr(adapter, "action_high"), dtype=np.float32).tolist()
            if getattr(adapter, "action_high", None) is not None
            else None
        ),
        "num_envs": num_envs,
        "environment_transitions": time_steps * num_envs,
        "real_env_transitions": time_steps * num_envs if is_real_environment else 0,
        "collection_execution": collection_execution,
    }
    metadata.update(environment_metadata)

    return WorldModelSequenceBatch(
        observations=observations,
        actions=actions,
        rewards=rewards,
        continues=continues,
        is_first=is_first,
        is_terminal=is_terminal,
        is_last=is_last,
        metadata=metadata,
    )


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
    metadata.update(
        {
            "collector": "synthetic_sequence_collector",
            "env": env_name,
            "environment_backend": "synthetic",
            "observation_mode": "pixels" if len(observation_shape) == 3 else "vector",
            "environment_transitions": time_steps * batch_size,
            "real_env_transitions": 0,
        }
    )
    return WorldModelSequenceBatch(
        observations=batch.observations,
        actions=batch.actions,
        rewards=batch.rewards,
        continues=batch.continues,
        is_first=batch.is_first,
        is_terminal=batch.is_terminal,
        is_last=batch.is_last,
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
