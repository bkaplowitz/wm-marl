"""Vector adapter for single-agent Gymnax environments.

Gymnax environments are native JAX environments. This adapter exposes them with
the same small vector-env contract used by the Melting Pot and JaxMARL CoinGame
adapters: observations are shaped ``[env, agent, ...]`` and actions are shaped
``[env, agent]``. For Gymnax, ``agent`` is always a singleton axis.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from world_marl.envs.meltingpot_adapter import VectorStep


GymnaxFactory = Callable[[], tuple[Any, Any]]


class GymnaxVectorAdapter:
    """Wrap a single-agent Gymnax environment as a vectorized training adapter.

    ``auto_reset`` is accepted for signature parity with
    ``MeltingPotVectorAdapter`` but is **not honored**: Gymnax's ``step``
    always resets done environments internally, so episodes never freeze at
    a terminal observation.
    """

    def __init__(
        self,
        env_name: str = "CartPole-v1",
        *,
        num_envs: int = 1,
        max_cycles: int = 500,
        seed: int = 0,
        env_factory: GymnaxFactory | None = None,
        auto_reset: bool = True,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if max_cycles < 1:
            raise ValueError("max_cycles must be >= 1")

        self.substrate = f"gymnax:{env_name}"
        self.env_name = env_name
        self.num_envs = num_envs
        self.max_cycles = max_cycles
        self.auto_reset = auto_reset
        self.agents = ("agent_0",)
        self.num_agents = 1

        if env_factory is None:
            import gymnax

            self.env, self.env_params = gymnax.make(env_name)
        else:
            self.env, self.env_params = env_factory()
        self.env_params = _with_max_cycles(self.env_params, max_cycles)

        action_space = self.env.action_space(self.env_params)
        if not hasattr(action_space, "n"):
            raise TypeError("only discrete Gymnax action spaces are supported")
        self.action_dim = int(action_space.n)
        # Discrete contract: actions are int32 scalars, so there is no action
        # vector shape and no box bounds (unlike Brax/DMC).
        self.action_shape: tuple[int, ...] = ()
        self.action_low = None
        self.action_high = None

        observation_shape = tuple(
            int(dim) for dim in self.env.observation_space(self.env_params).shape
        )
        self.observation_shape = observation_shape or (1,)
        self.raw_observation_shape = self.observation_shape
        self.observation_size = None
        self.include_observation_scalars = False
        self.scalar_observation_keys: tuple[str, ...] = ()
        self.append_agent_id = False

        self._split = jax.vmap(jax.random.split)
        self._reset = jax.jit(jax.vmap(self.env.reset, in_axes=(0, None)))
        self._step = jax.jit(
            jax.vmap(self.env.step, in_axes=(0, 0, 0, None)),
        )

        self._keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
        self._state = None
        self._episode_returns = np.zeros((num_envs, 1), dtype=np.float32)
        self._episode_lengths = np.zeros((num_envs,), dtype=np.int32)
        # Jitted rollout scans, keyed by (id(get_action_and_value), num_steps), so
        # the compile is paid once and reused across PPO updates (train_state flows
        # as a traced arg, so changing params does not retrigger a recompile).
        self._rollout_scan_jit: dict[tuple[int, int], Callable] = {}

    def reset(self) -> np.ndarray:
        split_keys = self._split(self._keys)
        self._keys = split_keys[:, 0]
        observations, self._state = self._reset(split_keys[:, 1], self.env_params)
        self._episode_returns[:] = 0.0
        self._episode_lengths[:] = 0
        return self._stack_observations(observations)

    def step(self, actions: np.ndarray) -> VectorStep:
        actions = np.asarray(actions, dtype=np.int32).reshape((self.num_envs, 1))
        split_keys = self._split(self._keys)
        self._keys = split_keys[:, 0]
        observations, self._state, reward, done, _ = self._step(
            split_keys[:, 1],
            self._state,
            actions[:, 0],
            self.env_params,
        )

        rewards = np.asarray(reward, dtype=np.float32).reshape((self.num_envs, 1))
        dones = np.asarray(done, dtype=np.float32).reshape((self.num_envs, 1))
        done_all = np.asarray(done, dtype=bool)
        self._episode_returns += rewards
        self._episode_lengths += 1

        completed_returns: list[tuple[float, ...]] = []
        completed_lengths: list[int] = []
        infos: list[dict[str, Any]] = []
        for env_index in np.flatnonzero(done_all):
            completed_returns.append((float(self._episode_returns[env_index, 0]),))
            completed_lengths.append(int(self._episode_lengths[env_index]))
            infos.append(
                {
                    "env_index": int(env_index),
                    "terminated": True,
                    "truncated": False,
                    "agent_infos": {},
                }
            )
            self._episode_returns[env_index] = 0.0
            self._episode_lengths[env_index] = 0

        return VectorStep(
            observations=self._stack_observations(observations),
            rewards=rewards,
            dones=dones,
            completed_returns=tuple(completed_returns),
            completed_lengths=tuple(completed_lengths),
            step_infos=tuple({} for _ in range(self.num_envs)),
            infos=tuple(infos),
        )

    def scan_rewards_dones(
        self,
        action_fn: Callable,
        num_steps: int,
        *,
        policy_key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Fully-jitted on-device eval rollout via ``lax.scan``.

        ``action_fn(obs[E,1,d], key) -> actions[E,1]`` is applied each step. The
        rollout starts from a fresh reset using the constructor PRNG state and
        does not touch the host-side episode bookkeeping, so it reads the same
        env-key stream that ``reset``/``step`` would consume. Returns
        ``(rewards[T,E,1] float32, dones_all[T,E] bool)`` as device arrays.
        Unlike coins, episodes may terminate early; ``max_steps_in_episode``
        still caps them at ``max_cycles``.
        """
        num_envs = self.num_envs
        env_params = self.env_params

        def stack_obs(obs):
            return obs.reshape((num_envs, 1, -1)).astype(jnp.float32)

        @jax.jit
        def run(init_keys, pkey):
            split0 = self._split(init_keys)
            obs0, state0 = self._reset(split0[:, 1], env_params)

            def body(carry, _):
                keys, state, obs, pkey = carry
                split = self._split(keys)
                pkey, action_key = jax.random.split(pkey)
                actions = action_fn(obs, action_key).astype(jnp.int32)
                obs_n, state_n, reward, done, _ = self._step(
                    split[:, 1], state, actions[:, 0], env_params
                )
                carry_n = (split[:, 0], state_n, stack_obs(obs_n), pkey)
                return carry_n, (
                    reward.reshape((num_envs, 1)).astype(jnp.float32),
                    done.reshape((num_envs,)),
                )

            init = (split0[:, 0], state0, stack_obs(obs0), pkey)
            _, outputs = jax.lax.scan(body, init, None, length=num_steps)
            return outputs

        return run(self._keys, policy_key)

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

        Mirrors ``JaxMARLCoinGameVectorAdapter.scan_rollout``: starts from the
        adapter's current carry, splits the policy key then the env keys in the
        same order as the Python loop (so it reproduces ``collect_rollout``
        bit-for-bit on integer actions), returns the stacked
        ``(obs, actions, log_probs, values, entropies, rewards, dones)`` plus
        the last flat observations, and advances ``_state``/``_keys`` — but not
        the episode accumulators, which the caller replays from the dones.
        """
        cache_key = (id(get_action_and_value), num_steps)
        run = self._rollout_scan_jit.get(cache_key)
        if run is None:
            run = self._build_rollout_scan(get_action_and_value, num_steps)
            self._rollout_scan_jit[cache_key] = run

        obs_flat0 = jnp.asarray(observations, dtype=jnp.float32).reshape(
            (self.num_envs * self.num_agents, -1)
        )
        ys, last_obs_flat, final_state, final_keys = run(
            train_state, self._state, self._keys, policy_key, obs_flat0
        )
        self._state = final_state
        self._keys = final_keys
        return ys, last_obs_flat

    def _build_rollout_scan(
        self, get_action_and_value: Callable, num_steps: int
    ) -> Callable:
        num_envs = self.num_envs
        env_params = self.env_params

        @jax.jit
        def run(train_state, init_state, init_keys, policy_key, init_obs_flat):
            def step(carry, _):
                state, keys, obs_flat, pkey = carry
                pkey, action_key = jax.random.split(pkey)
                actions, log_probs, values, entropies = get_action_and_value(
                    train_state, action_key, obs_flat
                )
                actions = actions.astype(jnp.int32)
                split = self._split(keys)
                obs_n, state_n, reward, done, _ = self._step(
                    split[:, 1],
                    state,
                    actions.reshape((num_envs,)),
                    env_params,
                )
                ys = (
                    obs_flat,
                    actions,
                    log_probs,
                    values,
                    entropies,
                    reward.reshape((num_envs,)).astype(jnp.float32),
                    done.reshape((num_envs,)),
                )
                obs_flat_n = obs_n.reshape((num_envs, -1)).astype(jnp.float32)
                carry_n = (state_n, split[:, 0], obs_flat_n, pkey)
                return carry_n, ys

            init = (init_state, init_keys, init_obs_flat, policy_key)
            (final_state, final_keys, last_obs_flat, _), ys = jax.lax.scan(
                step, init, None, length=num_steps
            )
            return ys, last_obs_flat, final_state, final_keys

        return run

    def sample_actions(self, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(
            low=0,
            high=self.action_dim,
            size=(self.num_envs, 1),
            dtype=np.int32,
        )

    def close(self) -> None:
        return None

    def _stack_observations(self, observations: Any) -> np.ndarray:
        array = np.asarray(observations, dtype=np.float32)
        array = array.reshape((self.num_envs, *self.observation_shape))
        return array[:, None, ...]


def is_gymnax_substrate(substrate: str) -> bool:
    return substrate.startswith("gymnax:")


def gymnax_env_name(substrate: str) -> str:
    if not is_gymnax_substrate(substrate):
        raise ValueError(f"not a Gymnax substrate: {substrate!r}")
    env_name = substrate.split(":", 1)[1]
    if not env_name:
        raise ValueError("Gymnax substrates must be formatted as 'gymnax:<env_id>'")
    return env_name


def _with_max_cycles(env_params: Any, max_cycles: int) -> Any:
    """Align Gymnax episode horizon with the adapter's max_cycles when possible."""
    if dataclasses.is_dataclass(env_params) and hasattr(
        env_params,
        "max_steps_in_episode",
    ):
        return dataclasses.replace(env_params, max_steps_in_episode=max_cycles)
    return env_params
