"""Contract collectors used before learner integration."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np

from world_marl.dreamarl.contracts import MultiAgentSequenceBatch


def collect_coin_game_sequence(
    *,
    time_steps: int,
    num_envs: int,
    max_cycles: int,
    seed: int,
) -> MultiAgentSequenceBatch:
    """Collect a deterministic CoinGame contract sequence.

    JaxMARL's public ``step`` auto-resets and replaces terminal observations.
    This gate deliberately calls ``step_env`` and resets afterward so replay
    retains the real successor separately from the next episode's first state.
    The trainable collector will use the same ordering inside one JAX scan.
    """

    if time_steps < 1:
        raise ValueError("time_steps must be >= 1")
    if num_envs < 1:
        raise ValueError("num_envs must be >= 1")
    if max_cycles < 1:
        raise ValueError("max_cycles must be >= 1")

    # CoinGame resets inside its transition function and replaces the final
    # observation. Keep its own horizon one step out of reach and apply the
    # requested contract boundary externally after preserving the successor.
    env = jaxmarl.make("coin_game", num_inner_steps=max_cycles + 1)
    agent_ids = tuple(str(agent) for agent in env.agents)
    num_agents = len(agent_ids)
    action_dim = int(env.action_space(agent_ids[0]).n)
    reset = jax.jit(jax.vmap(env.reset))
    step_env = jax.jit(jax.vmap(env.step))
    split = jax.jit(jax.vmap(lambda key: jax.random.split(key, 3)))
    keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
    observations_by_agent, state = reset(keys)
    rng = np.random.default_rng(seed)

    def stack_observations(values) -> np.ndarray:
        return np.stack(
            [
                np.asarray(values[agent], dtype=np.float32).reshape(
                    (num_envs, -1)
                )
                for agent in agent_ids
            ],
            axis=1,
        )

    current_observation = stack_observations(observations_by_agent)
    first = np.ones((num_envs,), dtype=bool)
    observations = []
    next_observations = []
    actions = []
    rewards = []
    team_rewards = []
    is_first = []
    is_last = []
    is_terminal = []
    agent_alive = []
    next_agent_alive = []
    action_masks = []

    for step_index in range(time_steps):
        action = rng.integers(
            0,
            action_dim,
            size=(num_envs, num_agents),
            dtype=np.int32,
        )
        action_dict = {
            agent: jnp.asarray(action[:, index])
            for index, agent in enumerate(agent_ids)
        }
        split_keys = split(keys)
        final_obs, final_state, reward, _, _ = step_env(
            split_keys[:, 0], state, action_dict
        )
        reset_obs, reset_state = reset(split_keys[:, 1])
        last = np.full(
            (num_envs,),
            (step_index + 1) % max_cycles == 0,
            dtype=bool,
        )
        keep_reset = jnp.asarray(last)
        state = jax.tree.map(
            lambda reset_value, final_value: jnp.where(
                keep_reset.reshape(
                    (num_envs,) + (1,) * (final_value.ndim - 1)
                ),
                reset_value,
                final_value,
            ),
            reset_state,
            final_state,
        )
        observations_by_agent = jax.tree.map(
            lambda reset_value, final_value: jnp.where(
                keep_reset.reshape(
                    (num_envs,) + (1,) * (final_value.ndim - 1)
                ),
                reset_value,
                final_value,
            ),
            reset_obs,
            final_obs,
        )
        final_observation = stack_observations(final_obs)
        reward_array = np.stack(
            [np.asarray(reward[agent], dtype=np.float32) for agent in agent_ids],
            axis=1,
        )
        alive = np.ones((num_envs, num_agents), dtype=bool)
        legal = np.ones((num_envs, num_agents, action_dim), dtype=bool)

        observations.append(current_observation)
        next_observations.append(final_observation)
        actions.append(action)
        rewards.append(reward_array)
        team_rewards.append(np.mean(reward_array, axis=1))
        is_first.append(first)
        is_last.append(last)
        # CoinGame's fixed horizon is a time-limit cut, not an environment
        # terminal. The preserved final observation permits value bootstrap.
        is_terminal.append(np.zeros_like(last))
        agent_alive.append(alive)
        next_agent_alive.append(alive)
        action_masks.append(legal)

        keys = split_keys[:, 2]
        current_observation = stack_observations(observations_by_agent)
        first = last

    masks = np.stack(action_masks)
    return MultiAgentSequenceBatch(
        observations=np.stack(observations),
        next_observations=np.stack(next_observations),
        actions=np.stack(actions),
        rewards=np.stack(rewards),
        team_rewards=np.stack(team_rewards),
        is_first=np.stack(is_first),
        is_last=np.stack(is_last),
        is_terminal=np.stack(is_terminal),
        agent_alive=np.stack(agent_alive),
        next_agent_alive=np.stack(next_agent_alive),
        action_mask=masks,
        next_action_mask=masks.copy(),
        agent_ids=agent_ids,
    )
