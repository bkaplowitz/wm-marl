"""Small contract-first collectors used before DreaMARL learner integration."""

from __future__ import annotations

import numpy as np

from world_marl.dreamarl.contracts import MultiAgentSequenceBatch
from world_marl.envs.jaxmarl_coin_adapter import JaxMARLCoinGameVectorAdapter


def collect_coin_game_sequence(
    *,
    time_steps: int,
    num_envs: int,
    max_cycles: int,
    seed: int,
) -> MultiAgentSequenceBatch:
    """Collect a deterministic random-policy contract smoke from CoinGame."""

    if time_steps < 1:
        raise ValueError("time_steps must be >= 1")
    adapter = JaxMARLCoinGameVectorAdapter(
        num_envs=num_envs,
        max_cycles=max_cycles,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    try:
        observation = adapter.reset()
        first = np.ones((num_envs,), dtype=bool)
        observations = []
        actions = []
        rewards = []
        team_rewards = []
        is_first = []
        is_last = []
        is_terminal = []
        agent_alive = []
        for _ in range(time_steps):
            action = adapter.sample_actions(rng)
            step = adapter.step(action)
            last = np.all(np.asarray(step.dones, dtype=bool), axis=1)

            observations.append(observation)
            actions.append(action)
            rewards.append(step.rewards)
            team_rewards.append(np.sum(step.rewards, axis=1))
            is_first.append(first)
            is_last.append(last)
            # CoinGame boundaries are fixed-length time-limit cuts. They reset
            # temporal state but retain value bootstrap semantics.
            is_terminal.append(np.zeros_like(last))
            agent_alive.append(
                np.ones((num_envs, adapter.num_agents), dtype=bool)
            )

            observation = step.observations
            first = last

        return MultiAgentSequenceBatch(
            observations=np.stack(observations),
            actions=np.stack(actions),
            rewards=np.stack(rewards),
            team_rewards=np.stack(team_rewards),
            is_first=np.stack(is_first),
            is_last=np.stack(is_last),
            is_terminal=np.stack(is_terminal),
            agent_alive=np.stack(agent_alive),
            agent_ids=adapter.agents,
        )
    finally:
        adapter.close()
