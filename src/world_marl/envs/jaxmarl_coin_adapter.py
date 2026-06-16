"""Expose JaxMARL's CoinGame through the MeltingPot vector-adapter interface."""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np
from jaxmarl.environments.coin_game.coin_game import CoinGame

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
    """Analytic CoinGame reward/done evaluated on world-model states.

    The world model predicts only next observations, so rewards come from
    CoinGame's *known* reward rule applied to the model's states. That reward is
    a deterministic function of the coin positions *before* the move and the
    player positions *after* it, both of which are encoded directly in the
    observation grid -- so we decode them and apply the payoff matrix, exactly
    mirroring the reward branch of ``CoinGame._step`` (pick up either coin: +1;
    let the opponent take your coin: -2).

    ``states``/``next_states`` are ``[env, agent, 36]`` flattened ``(3, 3, 4)``
    grids whose values are ``[red_player, blue_player, red_coin, blue_coin]``
    in agent 0's absolute frame (agent 1 sees the same grid colour-swapped, so
    agent 0 is the canonical view). ``env_actions`` is unused: the predicted
    next-state player values already encode where each move landed.

    Returns ``(rewards, dones)``, each shaped ``[env, agent]``. ``dones`` is all
    zeros: the observation carries no episode clock (``inner_t``), and CoinGame
    is a continuing task that auto-resets internally, so the imagined rollout is
    treated as non-terminating.
    """
    del env_actions  # next-state player positions already reflect the moves
    states = jnp.asarray(states)
    next_states = jnp.asarray(next_states)
    num_envs, num_agents = states.shape[0], states.shape[1]

    current = states[:, 0].reshape((num_envs, 3, 3, 4))
    nxt = next_states[:, 0].reshape((num_envs, 3, 3, 4))

    def _cell(plane: jnp.ndarray) -> jnp.ndarray:
        # Most-likely occupied grid cell for an entity (model states are
        # soft, not one-hot), as a flat index in [0, 9).
        return jnp.argmax(plane.reshape((num_envs, 9)), axis=-1)

    red_next = _cell(nxt[..., 0])
    blue_next = _cell(nxt[..., 1])
    red_coin = _cell(current[..., 2])
    blue_coin = _cell(current[..., 3])

    red_takes_red = red_next == red_coin
    red_takes_blue = red_next == blue_coin
    blue_takes_red = blue_next == red_coin
    blue_takes_blue = blue_next == blue_coin

    (rr, rb, r_pen), (br, bb, b_pen) = _COIN_PAYOFF
    red_reward = red_takes_red * rr + red_takes_blue * rb + blue_takes_red * r_pen
    blue_reward = blue_takes_red * br + blue_takes_blue * bb + red_takes_blue * b_pen

    rewards = jnp.stack([red_reward, blue_reward], axis=1).astype(jnp.float32)
    dones = jnp.zeros((num_envs, num_agents), dtype=jnp.float32)
    return rewards, dones
