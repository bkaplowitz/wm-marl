"""Vector adapter for Brax single-agent continuous-control environments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from world_marl.envs.meltingpot_adapter import VectorStep

BraxEnvFactory = Callable[[], Any]


def make_brax_env(
    env_id: str,
    *,
    backend: str | None = None,
    episode_length: int = 1000,
):
    """Build a Brax environment by name."""

    from brax import envs

    kwargs = {}
    if backend is not None:
        kwargs["backend"] = backend
    return envs.create(env_name=env_id, episode_length=episode_length, **kwargs)


class BraxVectorAdapter:
    """Wrap Brax environments in the repo's single-agent vector contract.

    ``auto_reset=False`` is accepted for signature parity with
    ``MeltingPotVectorAdapter`` but is **not honored**: Brax's own
    ``AutoResetWrapper`` is always active and rewinds done environments to
    their first reset state internally, so episodes never freeze at a
    terminal observation.
    """

    def __init__(
        self,
        env_id: str = "reacher",
        *,
        num_envs: int = 1,
        max_cycles: int = 1000,
        seed: int = 0,
        env_factory: BraxEnvFactory | None = None,
        auto_reset: bool = True,
        backend: str | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if max_cycles < 1:
            raise ValueError("max_cycles must be >= 1")

        self.env_id = env_id
        self.substrate = f"brax:{env_id}"
        self.num_envs = int(num_envs)
        self.max_cycles = int(max_cycles)
        self.auto_reset = auto_reset
        self.backend = backend
        self.agents = ("agent_0",)
        self.num_agents = 1

        self._env = (
            env_factory
            or (
                lambda: make_brax_env(
                    env_id, backend=backend, episode_length=max_cycles
                )
            )
        )()
        self._reset = jax.jit(jax.vmap(self._env.reset))
        self._step = jax.jit(jax.vmap(self._env.step))
        self._base_key = jax.random.PRNGKey(seed)
        self._reset_counter = 0
        self._state = self._reset(self._next_reset_keys())

        observations = np.asarray(jax.device_get(self._state.obs), dtype=np.float32)
        self.observation_shape = tuple(observations.shape[1:]) or (1,)
        self.raw_observation_shape = self.observation_shape
        self.observation_size = None
        self.include_observation_scalars = False
        self.scalar_observation_keys: tuple[str, ...] = ()
        self.append_agent_id = False

        self.action_shape = (int(getattr(self._env, "action_size")),)
        self.action_dim = self.action_shape[0]
        self.action_low = -np.ones((self.action_dim,), dtype=np.float32)
        self.action_high = np.ones((self.action_dim,), dtype=np.float32)

        self._episode_returns = np.zeros((self.num_envs, 1), dtype=np.float32)
        self._episode_lengths = np.zeros((self.num_envs,), dtype=np.int32)
        # Jitted rollout scans, keyed by (id(get_action_and_value), num_steps), so
        # the compile is paid once and reused across calls (train_state flows as a
        # traced arg, so changing params does not retrigger a recompile).
        self._rollout_scan_jit: dict[tuple[int, int], Callable] = {}

    def reset(self) -> np.ndarray:
        self._episode_returns[:] = 0.0
        self._episode_lengths[:] = 0
        self._state = self._reset(self._next_reset_keys())
        return self._observations()

    def step(self, actions: np.ndarray) -> VectorStep:
        action_batch = np.asarray(actions, dtype=np.float32).reshape(
            (self.num_envs, self.action_dim)
        )
        action_batch = np.clip(action_batch, self.action_low, self.action_high)

        next_state = self._step(self._state, jnp.asarray(action_batch))
        rewards_flat = np.asarray(jax.device_get(next_state.reward), dtype=np.float32)
        env_done = np.asarray(jax.device_get(next_state.done), dtype=np.float32) > 0.5

        self._episode_returns[:, 0] += rewards_flat.reshape((self.num_envs,))
        self._episode_lengths += 1
        truncated = self._episode_lengths >= self.max_cycles
        done_mask = np.logical_or(env_done.reshape((self.num_envs,)), truncated)

        completed_returns: list[tuple[float, ...]] = []
        completed_lengths: list[int] = []
        infos: list[dict[str, Any]] = []
        for env_index, done in enumerate(done_mask):
            if not bool(done):
                continue
            completed_returns.append((float(self._episode_returns[env_index, 0]),))
            completed_lengths.append(int(self._episode_lengths[env_index]))
            infos.append(
                {
                    "env_index": int(env_index),
                    "terminated": bool(env_done[env_index]),
                    "truncated": bool(truncated[env_index]),
                    "agent_infos": {},
                }
            )

        if self.auto_reset and bool(np.any(done_mask)):
            reset_state = self._reset(self._next_reset_keys())
            next_state = _select_reset_state(
                reset_state,
                next_state,
                jnp.asarray(done_mask),
                num_envs=self.num_envs,
            )
            self._episode_returns[done_mask] = 0.0
            self._episode_lengths[done_mask] = 0

        self._state = next_state
        return VectorStep(
            observations=self._observations(),
            rewards=rewards_flat.reshape((self.num_envs, 1)).astype(np.float32),
            dones=done_mask.astype(np.float32).reshape((self.num_envs, 1)),
            completed_returns=tuple(completed_returns),
            completed_lengths=tuple(completed_lengths),
            step_infos=tuple({} for _ in range(self.num_envs)),
            infos=tuple(infos),
        )

    def scan_rollout(
        self,
        get_action_and_value: Callable,
        train_state: Any,
        num_steps: int,
        *,
        policy_key: jax.Array,
        observations: np.ndarray,
    ) -> tuple[tuple[jax.Array, ...], jax.Array]:
        """Run ``num_steps`` policy steps fully on device with ``lax.scan``.

        Same contract as ``JaxMARLCoinGameVectorAdapter.scan_rollout``: starts
        from the adapter's current carry, returns the stacked
        ``(obs, actions, log_probs, values, entropies, rewards, dones)`` plus
        the last flat observations, and advances ``_state`` — but not the
        episode accumulators, which the caller replays from the recorded dones
        (``_replay_scan_episode_bookkeeping``); the in-scan truncation timer
        starts from ``_episode_lengths`` and runs the identical recurrence, so
        the replayed lengths land on the same boundary.

        Deviations from the host loop, both intentional: recorded ``actions``
        are the raw policy outputs (a clipped copy is what steps the env,
        matching ``step``), and in-scan resets draw fresh keys from a single
        ``fold_in`` per call rather than one per reset event, so reset streams
        are distribution-equivalent — not bit-for-bit — with the host loop.
        Rewards are recorded pre-reset and ``dones`` fold in the ``max_cycles``
        truncation, exactly like ``step``. No ``scan_rewards_dones`` is
        provided: that eval path assumes lockstep fixed-horizon episodes and
        raises for early-terminating envs.
        """
        cache_key = (id(get_action_and_value), num_steps)
        run = self._rollout_scan_jit.get(cache_key)
        if run is None:
            run = self._build_rollout_scan(get_action_and_value, num_steps)
            self._rollout_scan_jit[cache_key] = run

        obs_flat0 = jnp.asarray(observations, dtype=jnp.float32).reshape(
            (self.num_envs * self.num_agents, -1)
        )
        reset_key = jax.random.fold_in(self._base_key, self._reset_counter)
        self._reset_counter += 1
        lengths0 = jnp.asarray(self._episode_lengths, dtype=jnp.int32)
        ys, last_obs_flat, final_state = run(
            train_state, self._state, reset_key, policy_key, obs_flat0, lengths0
        )
        self._state = final_state
        return ys, last_obs_flat

    def _build_rollout_scan(
        self, get_action_and_value: Callable, num_steps: int
    ) -> Callable:
        num_envs = self.num_envs
        action_dim = self.action_dim
        max_cycles = self.max_cycles
        auto_reset = self.auto_reset

        @jax.jit
        def run(
            train_state, init_state, reset_key, policy_key, init_obs_flat, init_lengths
        ):
            def step(carry, _):
                state, obs_flat, lengths, pkey, rkey = carry
                pkey, action_key = jax.random.split(pkey)
                actions, log_probs, values, entropies = get_action_and_value(
                    train_state, action_key, obs_flat
                )
                actions = actions.astype(jnp.float32)
                action_batch = jnp.clip(
                    actions.reshape((num_envs, action_dim)),
                    self.action_low,
                    self.action_high,
                )
                state_n = self._step(state, action_batch)
                reward = state_n.reward.reshape((num_envs,)).astype(jnp.float32)
                lengths_n = lengths + 1
                done_mask = jnp.logical_or(
                    state_n.done.reshape((num_envs,)) > 0.5,
                    lengths_n >= max_cycles,
                )
                if auto_reset:
                    rkey, step_reset_key = jax.random.split(rkey)
                    reset_keys = jax.random.split(step_reset_key, num_envs)

                    def with_reset(operand):
                        stepped, keys = operand
                        return _select_reset_state(
                            self._reset(keys), stepped, done_mask, num_envs=num_envs
                        )

                    state_n = jax.lax.cond(
                        jnp.any(done_mask),
                        with_reset,
                        lambda operand: operand[0],
                        (state_n, reset_keys),
                    )
                    lengths_n = jnp.where(done_mask, 0, lengths_n)
                obs_flat_n = state_n.obs.reshape((num_envs, -1)).astype(jnp.float32)
                ys = (
                    obs_flat,
                    actions,
                    log_probs,
                    values,
                    entropies,
                    reward,
                    done_mask,
                )
                return (state_n, obs_flat_n, lengths_n, pkey, rkey), ys

            init = (init_state, init_obs_flat, init_lengths, policy_key, reset_key)
            (final_state, last_obs_flat, _, _, _), ys = jax.lax.scan(
                step, init, None, length=num_steps
            )
            return ys, last_obs_flat, final_state

        return run

    def sample_actions(self, rng: np.random.Generator) -> np.ndarray:
        actions = rng.uniform(
            low=self.action_low,
            high=self.action_high,
            size=(self.num_envs, self.action_dim),
        ).astype(np.float32)
        return actions[:, None, :]

    def close(self) -> None:
        return None

    def _next_reset_keys(self) -> jax.Array:
        key = jax.random.fold_in(self._base_key, self._reset_counter)
        self._reset_counter += 1
        return jax.random.split(key, self.num_envs)

    def _observations(self) -> np.ndarray:
        observations = np.asarray(jax.device_get(self._state.obs), dtype=np.float32)
        return observations.reshape((self.num_envs, -1))[:, None, :]


def is_brax_substrate(substrate: str) -> bool:
    return substrate.startswith("brax:")


def brax_env_name(substrate: str) -> str:
    if not is_brax_substrate(substrate):
        raise ValueError(f"not a Brax substrate: {substrate!r}")
    env_id = substrate.split(":", 1)[1]
    if not env_id:
        raise ValueError("Brax substrates must be formatted as 'brax:<env_name>'")
    return env_id


def _select_reset_state(
    reset_state, step_state, done_mask: jax.Array, *, num_envs: int
):
    def select(reset_leaf, step_leaf):
        if not hasattr(step_leaf, "shape") or not hasattr(reset_leaf, "shape"):
            return step_leaf
        if not step_leaf.shape or step_leaf.shape[0] != num_envs:
            return step_leaf
        mask = done_mask
        while mask.ndim < step_leaf.ndim:
            mask = mask[..., None]
        return jnp.where(mask, reset_leaf, step_leaf)

    return jax.tree_util.tree_map(select, reset_state, step_state)
