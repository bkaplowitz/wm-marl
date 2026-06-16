from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from flow_matching.models import MLPVectorField
from world_marl.algs.ippo import IPPOConfig, create_train_state as create_ippo_state
from world_marl.algs.mappo import MAPPOConfig, create_train_state as create_mappo_state
from world_marl.envs.meltingpot_adapter import MeltingPotVectorAdapter
from world_marl.world_model import (
    VectorTransitionBatch,
    VectorWorldModelConfig,
    _pack_context,
    _pack_transition,
    _transition_dim,
    _unpack_transition,
    create_world_model_state,
    simulate_mappo_model_rollout,
    train_world_model_step,
    world_model_loss,
)
from world_marl.world_model_training import (
    collect_random_transition_batch,
    flatten_state_observations,
)


def test_random_prefit_collection_uses_flat_vector_states(
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
        assert not hasattr(batch, "policy_ids")
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


def test_create_world_model_state_uses_mlp_vector_field():
    config = VectorWorldModelConfig(
        state_dim=4,
        num_agents=2,
        action_dim=3,
        hidden_dims=(8,),
        integration_steps=1,
    )

    model_state = create_world_model_state(jax.random.PRNGKey(0), config)
    model = model_state.apply_fn.__self__
    input_dim = (config.num_agents * config.state_dim) + (
        config.num_agents * config.state_dim + config.num_agents * config.action_dim
    )
    output = model_state.apply_fn(
        {"params": model_state.params},
        jnp.zeros((1, input_dim), dtype=jnp.float32),
        jnp.zeros((1, 1), dtype=jnp.float32),
    )

    assert isinstance(model, MLPVectorField)
    assert output.shape == (1, input_dim)


def test_world_model_pack_context_order_is_state_then_agent_major_actions():
    config = VectorWorldModelConfig(state_dim=2, num_agents=2, action_dim=3)
    states = jnp.asarray([[[1.0, 2.0], [3.0, 4.0]]])
    actions = jnp.asarray([[0, 2]], dtype=jnp.int32)

    packed = _pack_context(states, actions, config)

    np.testing.assert_allclose(
        np.asarray(packed),
        np.asarray([[1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
    )


def test_world_model_pack_transition_round_trips_next_state_only():
    config = VectorWorldModelConfig(state_dim=2, num_agents=2, action_dim=3)
    next_states = jnp.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )

    packed = _pack_transition(next_states, config)
    unpacked = _unpack_transition(packed, config)

    assert packed.shape == (2, _transition_dim(config))
    assert packed.shape == (2, 4)
    np.testing.assert_allclose(np.asarray(unpacked), np.asarray(next_states))


def test_world_model_loss_ignores_reward_and_done_fields():
    config = VectorWorldModelConfig(
        state_dim=2,
        num_agents=2,
        action_dim=3,
        hidden_dims=(8,),
        integration_steps=1,
    )
    model_state = create_world_model_state(jax.random.PRNGKey(0), config)
    batch = VectorTransitionBatch(
        states=jnp.zeros((4, 2, 2), dtype=jnp.float32),
        actions=jnp.zeros((4, 2), dtype=jnp.int32),
        next_states=jnp.ones((4, 2, 2), dtype=jnp.float32),
        rewards=jnp.zeros((4, 2), dtype=jnp.float32),
        dones=jnp.zeros((4, 2), dtype=jnp.float32),
    )
    changed_rewards_dones = batch._replace(
        rewards=jnp.full((4, 2), 100.0, dtype=jnp.float32),
        dones=jnp.ones((4, 2), dtype=jnp.float32),
    )

    key = jax.random.PRNGKey(1)
    base_loss = world_model_loss(
        model_state.params,
        model_state.apply_fn,
        key,
        batch,
        config,
    )
    changed_loss = world_model_loss(
        model_state.params,
        model_state.apply_fn,
        key,
        changed_rewards_dones,
        config,
    )

    np.testing.assert_allclose(np.asarray(base_loss), np.asarray(changed_loss))


def test_train_world_model_step_returns_finite_loss():
    config = VectorWorldModelConfig(
        state_dim=2,
        num_agents=2,
        action_dim=3,
        hidden_dims=(8,),
        integration_steps=1,
    )
    model_state = create_world_model_state(jax.random.PRNGKey(0), config)
    batch = VectorTransitionBatch(
        states=jnp.zeros((4, 2, 2), dtype=jnp.float32),
        actions=jnp.zeros((4, 2), dtype=jnp.int32),
        next_states=jnp.ones((4, 2, 2), dtype=jnp.float32),
        rewards=jnp.zeros((4, 2), dtype=jnp.float32),
        dones=jnp.zeros((4, 2), dtype=jnp.float32),
    )

    _, loss = train_world_model_step(
        model_state,
        jax.random.PRNGKey(1),
        batch,
        config,
    )

    assert jnp.isfinite(loss)


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


def test_simulate_ippo_model_rollout_uses_reward_done_provider():
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

    def reward_done_fn(states, actions, next_states):
        del states, next_states
        return actions.astype(jnp.float32), (actions == 2).astype(jnp.float32)

    rollout = simulate_ippo_model_rollout(
        model_state,
        policy_state,
        jnp.zeros((3, 2, 4), dtype=jnp.float32),
        jax.random.PRNGKey(2),
        rollout_steps=2,
        config=config,
        reward_done_fn=reward_done_fn,
    )

    np.testing.assert_allclose(
        np.asarray(rollout.batch.rewards),
        np.asarray(rollout.batch.actions, dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(rollout.batch.dones),
        np.asarray(rollout.batch.actions == 2, dtype=np.float32),
    )
