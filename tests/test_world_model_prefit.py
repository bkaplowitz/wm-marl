from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from world_marl.algs.ippo import IPPOConfig, create_train_state as create_ippo_state
from world_marl.algs.mappo import MAPPOConfig, create_train_state as create_mappo_state
from world_marl.envs.meltingpot_adapter import MeltingPotVectorAdapter
from world_marl.world_model import (
    VectorWorldModelConfig,
    create_world_model_state,
    simulate_mappo_model_rollout,
)
from world_marl.world_model_training import (
    collect_random_transition_batch,
    flatten_state_observations,
)


def test_random_prefit_collection_uses_flat_vector_states_and_zero_policy_ids(
    dummy_env_factory,
):
    adapter = MeltingPotVectorAdapter(num_envs=1, env_factory=dummy_env_factory)
    try:
        observations = adapter.reset()
        batch, next_observations, start_states = collect_random_transition_batch(
            adapter,
            observations,
            np.random.default_rng(0),
            rollout_steps=2,
        )

        state_dim = int(np.prod(adapter.observation_shape))
        assert batch.states.shape == (
            2 * adapter.num_envs,
            adapter.num_agents,
            state_dim,
        )
        assert batch.actions.shape == (2 * adapter.num_envs, adapter.num_agents)
        assert batch.next_states.shape == batch.states.shape
        assert batch.rewards.shape == (2 * adapter.num_envs, adapter.num_agents)
        assert batch.dones.shape == (2 * adapter.num_envs, adapter.num_agents)
        assert batch.policy_ids.shape == (2 * adapter.num_envs,)
        np.testing.assert_array_equal(np.asarray(batch.policy_ids), 0)
        assert next_observations.shape == observations.shape
        assert start_states.shape == (
            2 * adapter.num_envs,
            adapter.num_agents,
            state_dim,
        )
    finally:
        adapter.close()


def test_flatten_state_observations_preserves_env_agent_axes():
    observations = np.arange(2 * 3 * 4, dtype=np.float32).reshape((2, 3, 4))

    states = flatten_state_observations(observations)

    assert states.shape == (2, 3, 4)
    np.testing.assert_allclose(states, observations)


def test_simulate_mappo_model_rollout_returns_vector_central_batches():
    config = VectorWorldModelConfig(
        state_dim=4,
        num_agents=2,
        action_dim=3,
        hidden_dims=(8,),
        integration_steps=1,
    )
    model_state = create_world_model_state(jax.random.PRNGKey(0), config)
    policy_config = MAPPOConfig(network_arch="mlp", num_minibatches=1)
    policy_state = create_mappo_state(
        jax.random.PRNGKey(1),
        (4,),
        (10,),
        3,
        policy_config,
    )
    initial_states = jnp.zeros((3, 2, 4), dtype=jnp.float32)

    rollout = simulate_mappo_model_rollout(
        model_state,
        policy_state,
        initial_states,
        jax.random.PRNGKey(2),
        rollout_steps=2,
        config=config,
    )

    assert rollout.batch.observations.shape == (2, 6, 4)
    assert rollout.batch.central_observations.shape == (2, 6, 10)
    assert rollout.batch.actions.shape == (2, 6)
    assert rollout.batch.rewards.shape == (2, 6)
    assert rollout.last_values.shape == (6,)
    assert "rollout_mean_reward" in rollout.metrics
    assert (
        rollout.metrics["rollout_mean_reward"]
        == rollout.metrics["model_rollout_mean_reward"]
    )


def test_simulate_ippo_model_rollout_supports_mlp_vector_policy():
    from world_marl.world_model import simulate_ippo_model_rollout

    config = VectorWorldModelConfig(
        state_dim=4,
        num_agents=2,
        action_dim=3,
        hidden_dims=(8,),
        integration_steps=1,
    )
    model_state = create_world_model_state(jax.random.PRNGKey(0), config)
    policy_state = create_ippo_state(
        jax.random.PRNGKey(1),
        (4,),
        3,
        IPPOConfig(network_arch="mlp", num_minibatches=1),
    )

    rollout = simulate_ippo_model_rollout(
        model_state,
        policy_state,
        jnp.zeros((3, 2, 4), dtype=jnp.float32),
        jax.random.PRNGKey(2),
        rollout_steps=2,
        config=config,
    )

    assert rollout.batch.observations.shape == (2, 6, 4)
    assert rollout.batch.actions.shape == (2, 6)
    assert rollout.last_values.shape == (6,)
    assert "rollout_mean_reward" in rollout.metrics
    assert (
        rollout.metrics["rollout_mean_reward"]
        == rollout.metrics["model_rollout_mean_reward"]
    )
