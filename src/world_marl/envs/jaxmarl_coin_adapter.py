"""Expose JaxMARL's CoinGame through the MeltingPot vector-adapter interface."""

from __future__ import annotations

import inspect

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
