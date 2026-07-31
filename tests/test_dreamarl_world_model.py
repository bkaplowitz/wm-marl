from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from world_marl.dreamarl.config import (
    DreaMARLConfig,
    DynamicsConfig,
    EncoderConfig,
)
from world_marl.dreamarl.contracts import JaxMultiAgentSequenceBatch
from world_marl.dreamarl.world_model import DreaMARLWorldModel


def _config() -> DreaMARLConfig:
    return DreaMARLConfig(
        max_agents=2,
        action_dim=5,
        encoder=EncoderConfig(
            kind="vector",
            embedding_dim=16,
            vector_hidden_dim=32,
            vector_layers=2,
        ),
        dynamics=DynamicsConfig(
            model_dim=16,
            num_layers=2,
            num_heads=4,
            mlp_ratio=2,
            context_length=8,
            stochastic_variables=4,
            stochastic_classes=4,
            cross_agent_layers=1,
            cross_agent_heads=4,
        ),
    )


def _batch() -> JaxMultiAgentSequenceBatch:
    time, envs, agents, obs_dim, actions = 6, 3, 2, 12, 5
    observations = jax.random.normal(
        jax.random.PRNGKey(1), (time, envs, agents, obs_dim)
    )
    return JaxMultiAgentSequenceBatch(
        observations=observations,
        next_observations=jnp.roll(observations, -1, axis=0),
        actions=jnp.zeros((time, envs, agents), jnp.int32),
        rewards=jnp.zeros((time, envs, agents), jnp.float32),
        team_rewards=jnp.zeros((time, envs), jnp.float32),
        is_first=jnp.zeros((time, envs), bool).at[0].set(True),
        is_last=jnp.zeros((time, envs), bool),
        is_terminal=jnp.zeros((time, envs), bool),
        valid=jnp.ones((time, envs), bool),
        agent_alive=jnp.ones((time, envs, agents), bool),
        next_agent_alive=jnp.ones((time, envs, agents), bool),
        action_mask=jnp.ones((time, envs, agents, actions), bool),
        next_action_mask=jnp.ones((time, envs, agents, actions), bool),
    )


def test_world_model_sequence_shapes_and_finite_outputs() -> None:
    config = _config()
    model = DreaMARLWorldModel(config)
    batch = _batch()
    key = jax.random.PRNGKey(2)
    variables = model.init(key, batch, key)
    inference = model.apply(variables, batch, key)
    predictions = inference.predictions
    assert predictions.hidden.shape == (6, 3, 2, 16)
    assert predictions.stochastic.shape == (6, 3, 2, 4, 4)
    assert predictions.next_embedding.shape == (6, 3, 2, 16)
    assert predictions.team_reward.shape == (6, 3)
    assert predictions.next_action_mask_logit.shape == (6, 3, 2, 5)
    assert inference.final_state.temporal.keys.shape[:3] == (6, 2, 8)
    for leaf in jax.tree.leaves(predictions):
        assert bool(jnp.all(jnp.isfinite(leaf)))


def test_untrained_lifecycle_heads_preserve_imagined_agents_and_actions() -> None:
    config = _config()
    model = DreaMARLWorldModel(config)
    batch = _batch()
    key = jax.random.PRNGKey(20)
    variables = model.init(key, batch, key)
    state = model.apply(variables, batch, key).final_state
    next_state, prediction = model.apply(
        variables,
        state,
        jnp.zeros((batch.observations.shape[1], config.max_agents), jnp.int32),
        key,
        method=model.imagine_step,
    )
    assert bool(jnp.all(next_state.agent_alive))
    assert bool(jnp.all(next_state.action_mask))
    np.testing.assert_allclose(
        jax.nn.sigmoid(prediction.continuation_logit),
        config.dynamics.initial_continuation,
        atol=1e-6,
    )

    inactive = state._replace(
        agent_alive=state.agent_alive.at[:, 1].set(False),
        action_mask=state.action_mask.at[:, 1].set(False),
    )
    respawned, _ = model.apply(
        variables,
        inactive,
        jnp.zeros((batch.observations.shape[1], config.max_agents), jnp.int32),
        key,
        method=model.imagine_step,
    )
    assert bool(jnp.all(respawned.agent_alive[:, 1]))


def test_teammate_action_changes_focal_transition_context() -> None:
    config = _config()
    model = DreaMARLWorldModel(config)
    batch = _batch()
    key = jax.random.PRNGKey(3)
    variables = model.init(key, batch, key)
    inference = model.apply(variables, batch, key)
    state = inference.final_state
    first = jnp.zeros((3, 2), jnp.int32)
    changed = first.at[:, 1].set(4)
    prediction_a = model.apply(
        variables, state, first, method=model.transition
    )
    prediction_b = model.apply(
        variables, state, changed, method=model.transition
    )
    difference = jnp.abs(
        prediction_a.context[:, 0] - prediction_b.context[:, 0]
    ).sum()
    assert float(difference) > 1e-5


def test_inactive_agent_cannot_change_active_agent_context() -> None:
    config = _config()
    model = DreaMARLWorldModel(config)
    batch = _batch()
    key = jax.random.PRNGKey(4)
    variables = model.init(key, batch, key)
    state = model.apply(variables, batch, key).final_state
    state = state._replace(
        agent_alive=state.agent_alive.at[:, 1].set(False),
        hidden=state.hidden.at[:, 1].set(100.0),
        stochastic=state.stochastic.at[:, 1].set(100.0),
    )
    action_a = jnp.zeros((3, 2), jnp.int32)
    action_b = action_a.at[:, 1].set(4)
    context_a = model.apply(
        variables, state, action_a, method=model.transition
    ).context
    context_b = model.apply(
        variables, state, action_b, method=model.transition
    ).context
    np.testing.assert_allclose(context_a[:, 0], context_b[:, 0], atol=1e-6)
    np.testing.assert_allclose(context_a[:, 1], 0.0, atol=1e-6)
