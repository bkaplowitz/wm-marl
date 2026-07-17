"""Vector adapter for Brax single-agent continuous-control environments."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from world_marl.envs.meltingpot_adapter import VectorStep

BraxEnvFactory = Callable[[], Any]
BraxObservationFn = Callable[[Any], jax.Array]


def _uniform_random_policy(policy_state, key, obs_flat, is_first):
    del is_first
    action_low, action_high = policy_state
    key, action_key = jax.random.split(key)
    actions = jax.random.uniform(
        action_key,
        (obs_flat.shape[0], action_low.shape[0]),
        minval=action_low,
        maxval=action_high,
    )
    return key, actions


def make_brax_env(
    env_id: str,
    *,
    backend: str | None = None,
    episode_length: int = 1000,
    auto_reset: bool = True,
):
    """Build a Brax environment by name."""

    from brax import envs

    kwargs = {}
    if backend is not None:
        kwargs["backend"] = backend
    return envs.create(
        env_name=env_id,
        episode_length=episode_length,
        auto_reset=auto_reset,
        **kwargs,
    )


@functools.lru_cache(maxsize=None)
def _shared_env_and_fns(
    env_id: str, backend: str | None, episode_length: int
) -> tuple[Any, Callable, Callable]:
    """Share the (stateless) Brax env and its jitted reset/step across adapters.

    Per-instance jit closures each carry their own compile cache, so building a
    fresh adapter per evaluation used to recompile reset/step every time.
    """
    env = make_brax_env(env_id, backend=backend, episode_length=episode_length)
    return env, jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))


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
        observation_fn: BraxObservationFn | None = None,
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
        self._observation_fn = observation_fn or (lambda state: state.obs)
        self.backend = backend or ("generalized" if env_factory is None else "unknown")
        self.environment_metadata = {
            "environment_backend": "brax",
            "physics_backend": self.backend,
            "observation_mode": "vector",
        }
        self.agents = ("agent_0",)
        self.num_agents = 1

        if env_factory is None:
            self._env, self._reset, self._step = _shared_env_and_fns(
                env_id, backend, self.max_cycles
            )
        else:
            self._env = env_factory()
            self._reset = jax.jit(jax.vmap(self._env.reset))
            self._step = jax.jit(jax.vmap(self._env.step))
        self._base_key = jax.random.PRNGKey(seed)
        self._reset_counter = 0
        self._state = self._reset(self._next_reset_keys())

        observations = np.asarray(
            jax.device_get(self._observation_fn(self._state)), dtype=np.float32
        )
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
        self._recurrent_rollout_scan_jit: dict[tuple[int, int], Callable] = {}
        self._online_rollout_scan_jit: dict[tuple[int, int], Callable] = {}

    def reset(self) -> np.ndarray:
        self._episode_returns[:] = 0.0
        self._episode_lengths = np.zeros((self.num_envs,), dtype=np.int32)
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

        Deviations from the Python loop, both intentional: recorded ``actions``
        are the raw policy outputs (a clipped copy is what steps the env,
        matching ``step``), and in-scan resets draw fresh keys from a single
        ``fold_in`` per call rather than one per reset event, so reset streams
        are distribution-equivalent — not bit-for-bit — with the Python loop.
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
                obs_flat_n = self._observation_fn(state_n).astype(jnp.float32)
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

    def scan_recurrent_rollout(
        self,
        policy_step: Callable,
        policy_state: Any,
        policy_carry: Any,
        num_steps: int,
        *,
        observations: np.ndarray,
    ) -> tuple[tuple[jax.Array, ...], jax.Array, Any]:
        cache_key = (id(policy_step), num_steps)
        run = self._recurrent_rollout_scan_jit.get(cache_key)
        if run is None:
            run = self._build_recurrent_rollout_scan(policy_step, num_steps)
            self._recurrent_rollout_scan_jit[cache_key] = run

        obs_flat0 = jnp.asarray(observations, dtype=jnp.float32).reshape(
            (self.num_envs, *self.observation_shape)
        )
        reset_key = jax.random.fold_in(self._base_key, self._reset_counter)
        self._reset_counter += 1
        lengths0 = jnp.asarray(self._episode_lengths, dtype=jnp.int32)
        ys, last_obs, final_state, final_policy_carry, final_lengths = run(
            policy_state,
            policy_carry,
            self._state,
            reset_key,
            obs_flat0,
            lengths0,
        )
        self._state = final_state
        self._episode_lengths = final_lengths
        return ys, last_obs, final_policy_carry

    def _build_recurrent_rollout_scan(
        self,
        policy_step: Callable,
        num_steps: int,
    ) -> Callable:
        num_envs = self.num_envs
        action_dim = self.action_dim
        max_cycles = self.max_cycles
        auto_reset = self.auto_reset

        @jax.jit
        def run(
            policy_state,
            policy_carry,
            init_state,
            reset_key,
            init_obs_flat,
            init_lengths,
        ):
            def step(carry, _):
                state, obs_flat, lengths, policy_carry, is_first, rkey = carry
                is_terminal = state.done.reshape((num_envs,)) > 0.5
                is_last = jnp.logical_or(
                    is_terminal,
                    lengths >= max_cycles,
                )
                policy_carry, actions = policy_step(
                    policy_state,
                    policy_carry,
                    obs_flat,
                    is_first,
                )
                actions = actions.astype(jnp.float32).reshape((num_envs, action_dim))
                actions = jnp.where(is_last[:, None], jnp.zeros_like(actions), actions)
                clipped_actions = jnp.clip(actions, self.action_low, self.action_high)
                stepped_state = self._step(state, clipped_actions)
                lengths_n = lengths + 1
                if auto_reset:
                    rkey, step_reset_key = jax.random.split(rkey)
                    reset_keys = jax.random.split(step_reset_key, num_envs)
                    state_n = jax.lax.cond(
                        jnp.any(is_last),
                        lambda operand: _select_reset_state(
                            self._reset(operand[1]),
                            operand[0],
                            is_last,
                            num_envs=num_envs,
                        ),
                        lambda operand: operand[0],
                        (stepped_state, reset_keys),
                    )
                    lengths_n = jnp.where(is_last, 0, lengths_n)
                    is_first_n = is_last
                else:
                    state_n = stepped_state
                    is_first_n = jnp.zeros_like(is_last)
                obs_flat_n = self._observation_fn(state_n).astype(jnp.float32)
                reward = state.reward.reshape((num_envs,)).astype(jnp.float32)
                outputs = (obs_flat, actions, reward, is_terminal, is_last)
                carry = (
                    state_n,
                    obs_flat_n,
                    lengths_n,
                    policy_carry,
                    is_first_n,
                    rkey,
                )
                return carry, outputs

            initial = (
                init_state,
                init_obs_flat,
                init_lengths,
                policy_carry,
                jnp.ones((num_envs,), dtype=bool),
                reset_key,
            )
            final, ys = jax.lax.scan(step, initial, None, length=num_steps)
            final_state, last_obs, final_lengths, final_policy_carry, _, _ = final
            return ys, last_obs, final_state, final_policy_carry, final_lengths

        return run

    def scan_online_rollout(
        self,
        learner_step: Callable,
        learner_carry: Any,
        num_steps: int,
        *,
        observations: np.ndarray,
    ) -> tuple[tuple[jax.Array, ...], jax.Array, Any]:
        cache_key = (id(learner_step), num_steps)
        run = self._online_rollout_scan_jit.get(cache_key)
        if run is None:
            run = self._build_online_rollout_scan(learner_step, num_steps)
            self._online_rollout_scan_jit[cache_key] = run

        obs_flat0 = jnp.asarray(observations, dtype=jnp.float32).reshape(
            (self.num_envs, -1)
        )
        reset_key = jax.random.fold_in(self._base_key, self._reset_counter)
        self._reset_counter += 1
        lengths0 = jnp.asarray(self._episode_lengths, dtype=jnp.int32)
        ys, last_obs, final_state, final_learner_carry, final_lengths = run(
            learner_carry,
            self._state,
            reset_key,
            obs_flat0,
            lengths0,
        )
        self._state = final_state
        self._episode_lengths = final_lengths
        return ys, last_obs, final_learner_carry

    def _build_online_rollout_scan(
        self,
        learner_step: Callable,
        num_steps: int,
    ) -> Callable:
        num_envs = self.num_envs
        action_dim = self.action_dim
        max_cycles = self.max_cycles
        auto_reset = self.auto_reset

        @jax.jit
        def run(
            learner_carry,
            init_state,
            reset_key,
            init_obs_flat,
            init_lengths,
        ):
            def step(carry, _):
                state, obs_flat, lengths, learner_carry, is_first, rkey = carry
                is_terminal = state.done.reshape((num_envs,)) > 0.5
                is_last = jnp.logical_or(is_terminal, lengths >= max_cycles)
                reward = state.reward.reshape((num_envs,)).astype(jnp.float32)
                learner_carry, actions, learner_metrics = learner_step(
                    learner_carry,
                    obs_flat,
                    reward,
                    is_terminal,
                    is_last,
                    is_first,
                )
                actions = actions.astype(jnp.float32).reshape((num_envs, action_dim))
                actions = jnp.where(is_last[:, None], jnp.zeros_like(actions), actions)
                clipped_actions = jnp.clip(actions, self.action_low, self.action_high)
                stepped_state = self._step(state, clipped_actions)
                lengths_n = lengths + 1
                if auto_reset:
                    rkey, step_reset_key = jax.random.split(rkey)
                    reset_keys = jax.random.split(step_reset_key, num_envs)
                    state_n = jax.lax.cond(
                        jnp.any(is_last),
                        lambda operand: _select_reset_state(
                            self._reset(operand[1]),
                            operand[0],
                            is_last,
                            num_envs=num_envs,
                        ),
                        lambda operand: operand[0],
                        (stepped_state, reset_keys),
                    )
                    lengths_n = jnp.where(is_last, 0, lengths_n)
                    is_first_n = is_last
                else:
                    state_n = stepped_state
                    is_first_n = jnp.zeros_like(is_last)
                obs_flat_n = self._observation_fn(state_n).astype(jnp.float32)
                outputs = (
                    obs_flat,
                    actions,
                    reward,
                    is_terminal,
                    is_last,
                    is_first,
                    learner_metrics,
                )
                return (
                    state_n,
                    obs_flat_n,
                    lengths_n,
                    learner_carry,
                    is_first_n,
                    rkey,
                ), outputs

            initial = (
                init_state,
                init_obs_flat,
                init_lengths,
                learner_carry,
                jnp.ones((num_envs,), dtype=bool),
                reset_key,
            )
            final, ys = jax.lax.scan(step, initial, None, length=num_steps)
            final_state, last_obs, final_lengths, final_learner_carry, _, _ = final
            return ys, last_obs, final_state, final_learner_carry, final_lengths

        return run

    def scan_random_sequence(
        self,
        num_steps: int,
        *,
        key: jax.Array,
        observations: np.ndarray,
    ) -> tuple[jax.Array, ...]:
        action_low = jnp.asarray(self.action_low, dtype=jnp.float32)
        action_high = jnp.asarray(self.action_high, dtype=jnp.float32)
        ys, _, _ = self.scan_recurrent_rollout(
            _uniform_random_policy,
            (action_low, action_high),
            key,
            num_steps,
            observations=observations,
        )
        return ys

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
        observations = np.asarray(
            jax.device_get(self._observation_fn(self._state)), dtype=np.float32
        )
        return observations.reshape((self.num_envs, *self.observation_shape))[:, None]


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
