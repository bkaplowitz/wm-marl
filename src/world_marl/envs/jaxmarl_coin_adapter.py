"""Expose JaxMARL's CoinGame through the MeltingPot vector-adapter interface."""

from __future__ import annotations

import inspect
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np
from jaxmarl.environments.coin_game.coin_game import MOVES, CoinGame

from world_marl.envs.meltingpot_adapter import VectorStep

# Reuse CoinGame's own default payoff matrix as the single source of truth so the
# analytic reward stays in lock-step with the environment if its defaults change.
# Layout mirrors CoinGame._step: [[rr, rb, r_penalty], [br, bb, b_penalty]].
_COIN_PAYOFF = inspect.signature(CoinGame.__init__).parameters["payoff_matrix"].default


# NB: This uses a compatible interface with MeltingPot. As a result it comes at the cost of complete vectorization/avoiding transferring from gppu to cu
# That would require something like jaxmarl.envs.coingame.
class JaxMARLCoinGameVectorAdapter:
    """Wrap JaxMARL CoinGame as a synchronous, numpy-batched vector environment.

    Mirrors ``MeltingPotVectorAdapter``'s duck-typed interface so the same
    IPPO/MAPPO training and world-model collection code runs unchanged. Because
    CoinGame is pure JAX, every env is stepped together under ``jax.vmap``: the
    per-env PRNG keys are a single ``(num_envs, 2)`` array, the ``EnvState`` is a
    single batched pytree, and the dict-keyed JaxMARL observations/rewards/dones
    are stacked into ``[env, agent, ...]`` arrays.

    ``auto_reset`` is accepted for signature parity with
    ``MeltingPotVectorAdapter`` but is **not honored**: CoinGame's ``step`` always
    resets the grid internally at the episode boundary, so the environment is
    inherently continuing and the returned boundary observation is the env's own
    fresh-episode observation.
    """

    def __init__(
        self,
        *,
        num_envs: int = 1,
        max_cycles: int = 10,
        seed: int = 0,
        auto_reset: bool = True,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if max_cycles < 1:
            raise ValueError("max_cycles must be >= 1")

        self.substrate = "coins"
        self.num_envs = num_envs
        self.max_cycles = max_cycles
        self.auto_reset = auto_reset
        self.env = jaxmarl.make("coin_game", num_inner_steps=max_cycles)
        self.agents = list(self.env.agents)
        self.num_agents = len(self.agents)
        self.action_dim = int(self.env.action_space(self.agents[0]).n)

        probe_obs, _ = self.env.reset(jax.random.PRNGKey(seed))
        self.observation_shape = (
            int(np.asarray(probe_obs[self.agents[0]]).reshape(-1).shape[0]),
        )
        # Parity attributes mirroring MeltingPotVectorAdapter so checkpointing and
        # metadata logging stay duck-type compatible. CoinGame has no RGB
        # downsampling, scalar channels, or agent-id channels, so these are inert.
        self.raw_observation_shape = self.observation_shape
        self.observation_size = None
        self.include_observation_scalars = False
        self.scalar_observation_keys: tuple[str, ...] = ()
        self.append_agent_id = False

        self._split = jax.vmap(jax.random.split)
        self._reset = jax.vmap(self.env.reset)
        self._step = jax.vmap(self.env.step)

        self._keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
        self._state = None
        self._episode_returns = np.zeros((num_envs, self.num_agents), dtype=np.float32)
        self._episode_lengths = np.zeros((num_envs,), dtype=np.int32)
        # Jitted rollout scans, keyed by (id(get_action_and_value), num_steps), so the
        # compile is paid once and reused across PPO updates (train_state flows
        # as a traced arg, so changing params does not retrigger a recompile).
        self._rollout_scan_jit: dict[tuple[int, int], Callable] = {}

    def reset(self) -> np.ndarray:
        split_keys = self._split(self._keys)
        self._keys = split_keys[:, 0]
        observations, self._state = self._reset(split_keys[:, 1])
        self._episode_returns[:] = 0.0
        self._episode_lengths[:] = 0
        return self._stack_agents(observations)

    def step(self, actions: np.ndarray) -> VectorStep:
        actions = np.asarray(actions, dtype=np.int32).reshape(
            (self.num_envs, self.num_agents)
        )
        action_dict = {
            agent: jnp.asarray(actions[:, agent_index], dtype=jnp.int32)
            for agent_index, agent in enumerate(self.agents)
        }

        split_keys = self._split(self._keys)
        self._keys = split_keys[:, 0]
        obs, self._state, reward, done, _ = self._step(
            split_keys[:, 1], self._state, action_dict
        )

        observations = self._stack_agents(obs)
        rewards = np.stack(
            [np.asarray(reward[a], dtype=np.float32) for a in self.agents], axis=1
        )
        dones = np.stack(
            [np.asarray(done[a], dtype=np.float32) for a in self.agents], axis=1
        )
        done_all = np.asarray(done["__all__"])

        self._episode_returns += rewards
        self._episode_lengths += 1

        completed_returns: list[tuple[float, ...]] = []
        completed_lengths: list[int] = []
        for env_index in np.flatnonzero(done_all):
            completed_returns.append(
                tuple(float(x) for x in self._episode_returns[env_index])
            )
            completed_lengths.append(int(self._episode_lengths[env_index]))
            self._episode_returns[env_index] = 0.0
            self._episode_lengths[env_index] = 0

        return VectorStep(
            observations=observations,
            rewards=rewards,
            dones=dones,
            completed_returns=tuple(completed_returns),
            completed_lengths=tuple(completed_lengths),
            step_infos=tuple({} for _ in range(self.num_envs)),
            infos=tuple({} for _ in range(self.num_envs)),
        )

    def scan_rewards_dones(self, action_fn, num_steps, *, policy_key):
        """Fully-jitted on-device rollout for CoinGame via ``lax.scan``.

        ``action_fn(obs[E,A,d], key) -> actions[E,A]`` is applied each step. The
        rollout starts from a fresh reset using the constructor PRNG state and
        does not touch the host-side episode bookkeeping, so it reads the same
        env-key stream that ``reset``/``step`` would consume. Returns
        ``(rewards[T,E,A] float32, dones_all[T,E] bool)`` as device arrays.
        """
        agents = self.agents

        def stack_obs(obs):
            return jnp.stack(
                [obs[a].reshape((self.num_envs, -1)) for a in agents], axis=1
            )

        def stack_reward(reward):
            return jnp.stack([reward[a] for a in agents], axis=1)

        @jax.jit
        def run(init_keys, pkey):
            split0 = self._split(init_keys)
            obs0, state0 = self._reset(split0[:, 1])

            def body(carry, _):
                keys, state, obs, pkey = carry
                split = self._split(keys)
                pkey, action_key = jax.random.split(pkey)
                actions = action_fn(obs, action_key).astype(jnp.int32)
                action_dict = {a: actions[:, i] for i, a in enumerate(agents)}
                obs_n, state_n, reward, done, _ = self._step(
                    split[:, 1], state, action_dict
                )
                carry_n = (split[:, 0], state_n, stack_obs(obs_n), pkey)
                return carry_n, (stack_reward(reward), done["__all__"])

            init = (split0[:, 0], state0, stack_obs(obs0), pkey)
            _, outputs = jax.lax.scan(body, init, None, length=num_steps)
            return outputs

        return run(self._keys, policy_key)

    def scan_rollout(
        self, get_action_and_value, train_state, num_steps, *, policy_key, observations
    ):
        """On-device PPO rollout from the adapter's CURRENT carry (mid-stream).

        Mirrors ``collect_rollout``'s per-step PRNG order -- split the policy key
        first (action sampling), then the env keys (``env.step``) -- so a jitted
        ``lax.scan`` reproduces the host loop bit-for-bit.
        ``get_action_and_value(train_state, action_key, obs_flat[E*A, d]) ->
        (actions[E*A] int, log_probs, values, entropies)`` is applied each
        step; ``actions`` and the aux arrays are
        recorded verbatim. Starts from ``(self._state, self._keys, observations)``
        and advances ``self._state`` / ``self._keys`` to the post-rollout carry,
        matching how ``step`` threads them. Returns ``(ys, last_obs_flat[E*A, d])``
        where ``ys`` stacks ``(obs, actions, log_probs, values, entropies,
        rewards, dones)`` over ``num_steps``.
        """
        run = self._rollout_scan_jit.get((id(get_action_and_value), num_steps))
        if run is None:
            run = self._build_rollout_scan(get_action_and_value, num_steps)
            self._rollout_scan_jit[(id(get_action_and_value), num_steps)] = run

        obs_flat0 = jnp.asarray(observations, dtype=jnp.float32).reshape(
            (self.num_envs * self.num_agents, -1)
        )
        ys, last_obs_flat, final_state, final_keys = run(
            train_state, self._state, self._keys, policy_key, obs_flat0
        )
        self._state = final_state
        self._keys = final_keys
        return ys, last_obs_flat

    def _build_rollout_scan(self, get_action_and_value, num_steps):
        agents = self.agents
        num_envs = self.num_envs
        num_agents = self.num_agents
        num_all_env_agents = num_envs * num_agents

        def stack_obs_flat(obs):
            return (
                jnp.stack([obs[a].reshape((num_envs, -1)) for a in agents], axis=1)
                .reshape((num_all_env_agents, -1))
                .astype(jnp.float32)
            )

        def stack_all_agents_vals(values):
            return jnp.stack([values[a] for a in agents], axis=1).reshape(
                (num_all_env_agents,)
            )

        @jax.jit
        def run(train_state, init_state, init_keys, policy_key, init_obs_flat):
            def step(carry, _):
                state, keys, obs_flat, pkey = carry
                pkey, action_key = jax.random.split(pkey)
                actions, log_probs, values, entropies = get_action_and_value(
                    train_state, action_key, obs_flat
                )
                actions = actions.astype(jnp.int32)
                action_2d = actions.reshape((num_envs, num_agents))
                action_dict = {a: action_2d[:, i] for i, a in enumerate(agents)}
                split = self._split(keys)
                obs_n, state_n, reward, done, _ = self._step(
                    split[:, 1], state, action_dict
                )
                ys = (
                    obs_flat,
                    actions,
                    log_probs,
                    values,
                    entropies,
                    stack_all_agents_vals(reward),
                    stack_all_agents_vals(done),
                )
                carry_n = (state_n, split[:, 0], stack_obs_flat(obs_n), pkey)
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
            size=(self.num_envs, self.num_agents),
        ).astype(np.int32)

    def close(self) -> None:
        return None

    def _stack_agents(self, obs) -> np.ndarray:
        return np.stack(
            [
                np.asarray(obs[a], dtype=np.float32).reshape((self.num_envs, -1))
                for a in self.agents
            ],
            axis=1,
        )


def coin_game_reward_done(
    states: jnp.ndarray,
    env_actions: jnp.ndarray,
    next_states: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Analytic CoinGame reward/done -- the env's true ``R(state, action)``.

    Mirrors the reward branch of ``CoinGame._step`` (lines applying ``MOVES``):
    use agent's *current* state, move both players
    with ``MOVES[action] % 3``, and pay out via the env's own payoff matrix
    (pick up either coin: +1; let the opponent take your coin: -2). The reward
    is a deterministic function of the current state and the actions taken.

    ``states`` is ``[env, agent, 36]`` flattened ``(3, 3, 4)`` grids with
    values ``[red_player, blue_player, red_coin, blue_coin]`` in agent 0's
    absolute frame; agent 1's grid is the same colours
    swapped. ``env_actions`` is
    ``[env, agent]`` with column 0 the red (agent-0) action, matching the
    ``action_0, action_1`` unpacking in ``_step``.
    """
    # reward is R(state, action); model prediction not used
    states = jnp.asarray(states)
    env_actions = jnp.asarray(env_actions, dtype=jnp.int32)
    num_envs, num_agents = states.shape[0], states.shape[1]

    grid = states[:, 0].reshape((num_envs, 3, 3, 4))

    def _pos(channel: int) -> jnp.ndarray:
        # Decode the occupied 2D cell (row, col) for an entity. argmax handles
        # both one-hot env states and soft model states
        flat = jnp.argmax(grid[..., channel].reshape((num_envs, 9)), axis=-1)
        return jnp.stack([flat // 3, flat % 3], axis=-1)

    red_pos, blue_pos = _pos(0), _pos(1)
    red_coin, blue_coin = _pos(2), _pos(3)

    new_red = (red_pos + MOVES[env_actions[:, 0]]) % 3
    new_blue = (blue_pos + MOVES[env_actions[:, 1]]) % 3

    red_red = jnp.all(new_red == red_coin, axis=-1)
    red_blue = jnp.all(new_red == blue_coin, axis=-1)
    blue_red = jnp.all(new_blue == red_coin, axis=-1)
    blue_blue = jnp.all(new_blue == blue_coin, axis=-1)

    (rr, rb, r_pen), (br, bb, b_pen) = _COIN_PAYOFF
    red_reward = rr * red_red + rb * red_blue + r_pen * blue_red
    blue_reward = br * blue_red + bb * blue_blue + b_pen * red_blue

    rewards = jnp.stack([red_reward, blue_reward], axis=1).astype(jnp.float32)
    dones = jnp.zeros((num_envs, num_agents), dtype=jnp.float32)
    return rewards, dones
