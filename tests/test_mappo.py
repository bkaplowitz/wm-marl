from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from world_marl.algs.ippo import tree_l2_distance
from world_marl.algs.mappo import (
    MAPPOConfig,
    MAPPORolloutBatch,
    create_train_state,
    mappo_update,
    select_actions,
)
from world_marl.training import build_central_observations, central_observation_shape


def _synthetic_mappo_batch(
    state,
    key,
    *,
    steps=4,
    actors=4,
    obs_shape=(8, 8, 3),
    central_obs_shape=(8, 8, 8),
):
    obs = jax.random.uniform(key, (steps, actors, *obs_shape))
    central_obs = jax.random.uniform(key, (steps, actors, *central_obs_shape))
    flat_obs = obs.reshape((steps * actors, *obs_shape))
    flat_central_obs = central_obs.reshape((steps * actors, *central_obs_shape))
    action_key, reward_key = jax.random.split(key)
    actions, log_probs, values = select_actions(
        state,
        action_key,
        flat_obs,
        flat_central_obs,
    )
    rewards = jax.random.normal(reward_key, (steps, actors))
    return MAPPORolloutBatch(
        observations=obs,
        central_observations=central_obs,
        actions=actions.reshape((steps, actors)),
        log_probs=log_probs.reshape((steps, actors)),
        rewards=rewards,
        dones=jnp.zeros((steps, actors)),
        values=values.reshape((steps, actors)),
    )


def test_build_central_observations_adds_all_agents_and_target_id():
    observations = np.zeros((1, 2, 4, 4, 3), dtype=np.float32)
    observations[:, 0, :, :, :] = 1.0
    observations[:, 1, :, :, :] = 2.0

    central = build_central_observations(observations)

    assert central.shape == (1, 2, 4, 4, 8)
    assert central_observation_shape((4, 4, 3), 2) == (4, 4, 8)
    np.testing.assert_allclose(central[0, 0, :, :, :3], 1.0)
    np.testing.assert_allclose(central[0, 0, :, :, 3:6], 2.0)
    np.testing.assert_allclose(central[0, 0, :, :, 6], 1.0)
    np.testing.assert_allclose(central[0, 0, :, :, 7], 0.0)
    np.testing.assert_allclose(central[0, 1, :, :, 6], 0.0)
    np.testing.assert_allclose(central[0, 1, :, :, 7], 1.0)


def test_mappo_forward_and_update_change_parameters():
    config = MAPPOConfig(update_epochs=2, num_minibatches=2, learning_rate=1e-3)
    obs_shape = (8, 8, 3)
    central_obs_shape = (8, 8, 8)
    state = create_train_state(
        jax.random.PRNGKey(0),
        obs_shape,
        central_obs_shape,
        3,
        config,
    )
    obs = jnp.ones((5, *obs_shape), dtype=jnp.float32)
    central_obs = jnp.ones((5, *central_obs_shape), dtype=jnp.float32)
    actions, log_probs, values = select_actions(
        state,
        jax.random.PRNGKey(1),
        obs,
        central_obs,
    )
    assert actions.shape == (5,)
    assert log_probs.shape == (5,)
    assert values.shape == (5,)

    batch = _synthetic_mappo_batch(
        state,
        jax.random.PRNGKey(2),
        obs_shape=obs_shape,
        central_obs_shape=central_obs_shape,
    )
    new_state, metrics = mappo_update(
        state,
        batch,
        jnp.zeros((4,)),
        jax.random.PRNGKey(3),
        config,
    )
    assert tree_l2_distance(state.params, new_state.params) > 0.0
    assert "total_loss" in metrics
