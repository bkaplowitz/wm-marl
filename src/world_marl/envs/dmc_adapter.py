"""Vector adapter for DeepMind Control Suite state observations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from world_marl.envs.meltingpot_adapter import VectorStep


DMCEnvFactory = Callable[[int], Any]


def dmc_boundary_flags(
    timestep: Any,
    *,
    max_cycles_reached: bool,
) -> tuple[bool, bool]:
    """Return DMC sequence-boundary and Bellman-terminal flags.

    DeepMind Control uses ``LAST`` with ``discount=1`` for time-limit
    truncations and ``discount=0`` for true task terminals. Older test or
    third-party timestep objects may omit ``discount``; those retain the
    historical behavior where ``LAST`` is treated as terminal.
    """

    environment_last = bool(timestep.last())
    is_last = environment_last or bool(max_cycles_reached)
    discount = getattr(timestep, "discount", None)
    is_terminal = environment_last and (
        discount is None or float(np.asarray(discount)) <= 0.0
    )
    return is_last, is_terminal


def make_dmc_env(env_id: str, *, seed: int):
    """Build a DeepMind Control Suite environment.

    ``env_id`` is formatted as ``"<domain>/<task>"``, for example
    ``"cartpole/swingup"``.
    """

    from dm_control import suite

    domain_name, task_name = _split_env_id(env_id)
    return suite.load(
        domain_name=domain_name,
        task_name=task_name,
        task_kwargs={"random": seed},
    )


class DMCVectorAdapter:
    """Wrap DeepMind Control tasks in the repo's single-agent vector contract."""

    def __init__(
        self,
        env_id: str = "cartpole/swingup",
        *,
        num_envs: int = 1,
        max_cycles: int = 1000,
        seed: int = 0,
        env_factory: DMCEnvFactory | None = None,
        auto_reset: bool = True,
        num_workers: int = 1,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if max_cycles < 1:
            raise ValueError("max_cycles must be >= 1")

        self.env_id = env_id
        self.substrate = f"dmc:{env_id}"
        self.num_envs = int(num_envs)
        self.max_cycles = int(max_cycles)
        self.auto_reset = auto_reset
        self.num_workers = int(num_workers)
        if self.num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        self.agents = ("agent_0",)
        self.num_agents = 1

        factory = env_factory or (lambda env_seed: make_dmc_env(env_id, seed=env_seed))
        self._envs = [factory(seed + index) for index in range(self.num_envs)]
        self._executor = (
            ThreadPoolExecutor(max_workers=min(self.num_workers, self.num_envs))
            if self.num_workers > 1
            else None
        )

        first_env = self._envs[0]
        self._observation_keys = _observation_keys(first_env.observation_spec())
        self.observation_shape = (
            _flatten_observation_spec(
                first_env.observation_spec(),
                self._observation_keys,
            ),
        )
        self.raw_observation_shape = self.observation_shape
        self.observation_size = None
        self.include_observation_scalars = False
        self.scalar_observation_keys: tuple[str, ...] = ()
        self.append_agent_id = False

        action_spec = first_env.action_spec()
        self.action_shape = tuple(int(dim) for dim in action_spec.shape) or (1,)
        self.action_dim = int(np.prod(self.action_shape))
        self.action_low = np.broadcast_to(
            np.asarray(action_spec.minimum, dtype=np.float32),
            self.action_shape,
        ).reshape((self.action_dim,))
        self.action_high = np.broadcast_to(
            np.asarray(action_spec.maximum, dtype=np.float32),
            self.action_shape,
        ).reshape((self.action_dim,))

        self._episode_returns = np.zeros((self.num_envs, 1), dtype=np.float32)
        self._episode_lengths = np.zeros((self.num_envs,), dtype=np.int32)

    def reset(self) -> np.ndarray:
        self._episode_returns[:] = 0.0
        self._episode_lengths[:] = 0
        if self._executor is None:
            timesteps = [env.reset() for env in self._envs]
        else:
            timesteps = list(self._executor.map(lambda env: env.reset(), self._envs))
        observations = [
            self._flatten_observation(timestep.observation) for timestep in timesteps
        ]
        return np.asarray(observations, dtype=np.float32)[:, None, :]

    def reset_indices(self, indices: np.ndarray) -> np.ndarray:
        """Reset selected vector members and return their new observations."""

        reset_indices = np.asarray(indices, dtype=np.int64).reshape((-1,))
        if np.any((reset_indices < 0) | (reset_indices >= self.num_envs)):
            raise IndexError("reset index is outside the vector environment")
        if np.unique(reset_indices).size != reset_indices.size:
            raise ValueError("reset indices must be unique")
        if reset_indices.size == 0:
            return np.empty((0, 1, *self.observation_shape), dtype=np.float32)

        self._episode_returns[reset_indices] = 0.0
        self._episode_lengths[reset_indices] = 0
        environments = [self._envs[int(index)] for index in reset_indices]
        if self._executor is None:
            timesteps = [env.reset() for env in environments]
        else:
            timesteps = list(self._executor.map(lambda env: env.reset(), environments))
        observations = [
            self._flatten_observation(timestep.observation) for timestep in timesteps
        ]
        return np.asarray(observations, dtype=np.float32)[:, None, :]

    def step(self, actions: np.ndarray) -> VectorStep:
        action_batch = np.asarray(actions, dtype=np.float32).reshape(
            (self.num_envs, self.action_dim)
        )
        observations = []
        rewards = np.zeros((self.num_envs, 1), dtype=np.float32)
        dones = np.zeros((self.num_envs, 1), dtype=np.float32)
        is_last = np.zeros((self.num_envs, 1), dtype=np.float32)
        is_terminal = np.zeros((self.num_envs, 1), dtype=np.float32)
        completed_returns: list[tuple[float, ...]] = []
        completed_lengths: list[int] = []
        infos: list[dict[str, Any]] = []

        if self._executor is None:
            timesteps = [
                env.step(flat_action.reshape(self.action_shape))
                for env, flat_action in zip(self._envs, action_batch, strict=True)
            ]
        else:
            timesteps = list(
                self._executor.map(
                    lambda item: item[0].step(item[1].reshape(self.action_shape)),
                    zip(self._envs, action_batch, strict=True),
                )
            )

        for env_index, (env, timestep) in enumerate(
            zip(self._envs, timesteps, strict=True)
        ):
            reward = 0.0 if timestep.reward is None else float(timestep.reward)
            self._episode_returns[env_index, 0] += reward
            self._episode_lengths[env_index] += 1

            last, terminal = dmc_boundary_flags(
                timestep,
                max_cycles_reached=(
                    self._episode_lengths[env_index] >= self.max_cycles
                ),
            )
            rewards[env_index, 0] = reward
            dones[env_index, 0] = float(last)
            is_last[env_index, 0] = float(last)
            is_terminal[env_index, 0] = float(terminal)

            if last:
                completed_returns.append((float(self._episode_returns[env_index, 0]),))
                completed_lengths.append(int(self._episode_lengths[env_index]))
                infos.append(
                    {
                        "env_index": int(env_index),
                        "terminated": terminal,
                        "truncated": not terminal,
                        "agent_infos": {},
                    }
                )
                if self.auto_reset:
                    timestep = env.reset()
                self._episode_returns[env_index] = 0.0
                self._episode_lengths[env_index] = 0

            observations.append(self._flatten_observation(timestep.observation))

        return VectorStep(
            observations=np.asarray(observations, dtype=np.float32)[:, None, :],
            rewards=rewards,
            dones=dones,
            completed_returns=tuple(completed_returns),
            completed_lengths=tuple(completed_lengths),
            step_infos=tuple({} for _ in range(self.num_envs)),
            infos=tuple(infos),
            is_last=is_last,
            is_terminal=is_terminal,
        )

    def sample_actions(self, rng: np.random.Generator) -> np.ndarray:
        actions = rng.uniform(
            low=self.action_low,
            high=self.action_high,
            size=(self.num_envs, self.action_dim),
        ).astype(np.float32)
        return actions[:, None, :]

    def save_state_npz(self, path: str | Path) -> Path:
        """Save simulator and task RNG state for an exact phase-boundary resume."""

        physics_states = []
        physics_times = []
        task_rng_keys = []
        task_rng_positions = []
        task_rng_has_gauss = []
        task_rng_cached_gaussian = []
        task_rng_algorithms = []
        environment_step_counts = []
        environment_reset_next_step = []
        for env in self._envs:
            physics_states.append(np.asarray(env.physics.get_state(), dtype=np.float64))
            physics_times.append(float(env.physics.data.time))
            task_rng = getattr(getattr(env, "_task", None), "_random", None)
            if task_rng is None or not hasattr(task_rng, "get_state"):
                raise RuntimeError("DMC task does not expose a restorable RandomState")
            algorithm, keys, position, has_gauss, cached_gaussian = task_rng.get_state()
            task_rng_algorithms.append(str(algorithm))
            task_rng_keys.append(np.asarray(keys, dtype=np.uint32))
            task_rng_positions.append(int(position))
            task_rng_has_gauss.append(int(has_gauss))
            task_rng_cached_gaussian.append(float(cached_gaussian))
            environment_step_counts.append(int(getattr(env, "_step_count", 0)))
            environment_reset_next_step.append(
                bool(getattr(env, "_reset_next_step", False))
            )

        snapshot_path = Path(path)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = snapshot_path.with_name(f".{snapshot_path.name}.tmp")
        with temporary_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                env_id=np.asarray(self.env_id),
                num_envs=np.asarray(self.num_envs, dtype=np.int64),
                physics_states=np.stack(physics_states),
                physics_times=np.asarray(physics_times, dtype=np.float64),
                task_rng_algorithms=np.asarray(task_rng_algorithms, dtype="U32"),
                task_rng_keys=np.stack(task_rng_keys),
                task_rng_positions=np.asarray(task_rng_positions, dtype=np.int64),
                task_rng_has_gauss=np.asarray(task_rng_has_gauss, dtype=np.int8),
                task_rng_cached_gaussian=np.asarray(
                    task_rng_cached_gaussian,
                    dtype=np.float64,
                ),
                environment_step_counts=np.asarray(
                    environment_step_counts,
                    dtype=np.int64,
                ),
                environment_reset_next_step=np.asarray(
                    environment_reset_next_step,
                    dtype=np.bool_,
                ),
                episode_returns=self._episode_returns,
                episode_lengths=self._episode_lengths,
            )
        temporary_path.replace(snapshot_path)
        return snapshot_path

    def load_state_npz(self, path: str | Path) -> None:
        """Restore a snapshot produced by :meth:`save_state_npz`."""

        with np.load(Path(path), allow_pickle=False) as data:
            saved_env_id = str(np.asarray(data["env_id"]).item())
            saved_num_envs = int(np.asarray(data["num_envs"]).item())
            if saved_env_id != self.env_id or saved_num_envs != self.num_envs:
                raise ValueError(
                    "DMC state snapshot does not match this adapter: "
                    f"expected {self.env_id} with {self.num_envs} envs, got "
                    f"{saved_env_id} with {saved_num_envs} envs"
                )
            physics_states = np.asarray(data["physics_states"], dtype=np.float64)
            physics_times = np.asarray(data["physics_times"], dtype=np.float64)
            task_rng_algorithms = np.asarray(data["task_rng_algorithms"])
            task_rng_keys = np.asarray(data["task_rng_keys"], dtype=np.uint32)
            task_rng_positions = np.asarray(data["task_rng_positions"], dtype=np.int64)
            task_rng_has_gauss = np.asarray(data["task_rng_has_gauss"], dtype=np.int8)
            task_rng_cached_gaussian = np.asarray(
                data["task_rng_cached_gaussian"],
                dtype=np.float64,
            )
            environment_step_counts = np.asarray(
                data["environment_step_counts"],
                dtype=np.int64,
            )
            environment_reset_next_step = np.asarray(
                data["environment_reset_next_step"],
                dtype=np.bool_,
            )
            episode_returns = np.asarray(data["episode_returns"], dtype=np.float32)
            episode_lengths = np.asarray(data["episode_lengths"], dtype=np.int32)

        for index, env in enumerate(self._envs):
            env.physics.set_state(physics_states[index])
            env.physics.data.time = physics_times[index]
            env.physics.forward()
            task_rng = getattr(getattr(env, "_task", None), "_random", None)
            if task_rng is None or not hasattr(task_rng, "set_state"):
                raise RuntimeError("DMC task does not expose a restorable RandomState")
            task_rng.set_state(
                (
                    str(task_rng_algorithms[index]),
                    task_rng_keys[index],
                    int(task_rng_positions[index]),
                    int(task_rng_has_gauss[index]),
                    float(task_rng_cached_gaussian[index]),
                )
            )
            env._step_count = int(environment_step_counts[index])
            env._reset_next_step = bool(environment_reset_next_step[index])
        self._episode_returns[...] = episode_returns
        self._episode_lengths[...] = episode_lengths

    def render(
        self,
        env_index: int = 0,
        *,
        height: int = 64,
        width: int = 64,
        camera_id: int = 0,
    ) -> np.ndarray:
        """Render one vector member through its dm_control physics object."""

        if not 0 <= env_index < self.num_envs:
            raise IndexError(f"env_index must be in [0, {self.num_envs})")
        if height < 1 or width < 1:
            raise ValueError("render height and width must be >= 1")
        physics = getattr(self._envs[env_index], "physics", None)
        if physics is None or not hasattr(physics, "render"):
            raise RuntimeError("DMC environment does not expose physics.render")
        frame = physics.render(
            height=int(height),
            width=int(width),
            camera_id=int(camera_id),
        )
        return np.asarray(frame, dtype=np.uint8)

    def close(self) -> None:
        for env in self._envs:
            close = getattr(env, "close", None)
            if close is not None:
                close()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _flatten_observation(self, observation: Any) -> np.ndarray:
        return _flatten_observation(observation, self._observation_keys)


def is_dmc_substrate(substrate: str) -> bool:
    return substrate.startswith("dmc:")


def dmc_env_name(substrate: str) -> str:
    if not is_dmc_substrate(substrate):
        raise ValueError(f"not a DMC substrate: {substrate!r}")
    env_id = substrate.split(":", 1)[1]
    _split_env_id(env_id)
    return env_id


def _split_env_id(env_id: str) -> tuple[str, str]:
    if "/" not in env_id:
        raise ValueError("DMC substrates must be formatted as 'dmc:<domain>/<task>'")
    domain_name, task_name = env_id.split("/", 1)
    if not domain_name or not task_name:
        raise ValueError("DMC substrates must be formatted as 'dmc:<domain>/<task>'")
    return domain_name, task_name


def _observation_keys(observation_spec: Any) -> tuple[str, ...]:
    if isinstance(observation_spec, dict):
        return tuple(observation_spec.keys())
    return ()


def _flatten_observation_spec(
    observation_spec: Any,
    keys: Sequence[str],
) -> int:
    if isinstance(observation_spec, dict):
        return int(
            sum(np.prod(tuple(observation_spec[key].shape) or (1,)) for key in keys)
        )
    return int(np.prod(tuple(observation_spec.shape) or (1,)))


def _flatten_observation(observation: Any, keys: Sequence[str]) -> np.ndarray:
    if isinstance(observation, dict):
        parts = [
            np.asarray(observation[key], dtype=np.float32).reshape((-1,))
            for key in keys
        ]
        return np.concatenate(parts, axis=0)
    return np.asarray(observation, dtype=np.float32).reshape((-1,))
