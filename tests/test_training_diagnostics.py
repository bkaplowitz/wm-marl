from __future__ import annotations

import jax

from world_marl.algs.ippo import IPPOConfig, create_train_state as create_ippo_state
from world_marl.algs.mappo import MAPPOConfig, create_train_state as create_mappo_state
from world_marl.envs.meltingpot_adapter import MeltingPotVectorAdapter
from world_marl.training import (
    central_observation_shape,
    collect_mappo_rollout,
    collect_rollout,
)


def test_ippo_rollout_writes_diagnostics(dummy_env_factory):
    adapter = MeltingPotVectorAdapter(num_envs=2, env_factory=dummy_env_factory)
    try:
        config = IPPOConfig()
        state = create_ippo_state(
            jax.random.PRNGKey(0),
            adapter.observation_shape,
            adapter.action_dim,
            config,
        )
        rollout = collect_rollout(
            adapter,
            state,
            adapter.reset(),
            jax.random.PRNGKey(1),
            rollout_steps=3,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        metrics = rollout.metrics
        assert sum(metrics["action_counts"]) == 12
        assert len(metrics["action_counts_by_agent"]) == adapter.num_agents
        assert len(metrics["policy_entropy_by_agent"]) == adapter.num_agents
        assert len(metrics["rollout_reward_mean_by_agent"]) == adapter.num_agents
        assert "value_explained_variance" in metrics
        assert "coin_related_info_items" in metrics
    finally:
        adapter.close()


def test_mappo_rollout_writes_diagnostics(dummy_env_factory):
    adapter = MeltingPotVectorAdapter(num_envs=1, env_factory=dummy_env_factory)
    try:
        config = MAPPOConfig()
        state = create_mappo_state(
            jax.random.PRNGKey(0),
            adapter.observation_shape,
            central_observation_shape(adapter.observation_shape, adapter.num_agents),
            adapter.action_dim,
            config,
        )
        rollout = collect_mappo_rollout(
            adapter,
            state,
            adapter.reset(),
            jax.random.PRNGKey(1),
            rollout_steps=3,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        metrics = rollout.metrics
        assert sum(metrics["action_counts"]) == 6
        assert len(metrics["action_frequencies"]) == adapter.action_dim
        assert len(metrics["episode_return_mean_by_agent"]) == adapter.num_agents
        assert "policy_entropy_mean" in metrics
        assert "value_target_std" in metrics
    finally:
        adapter.close()
