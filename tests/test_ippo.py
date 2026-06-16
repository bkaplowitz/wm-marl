from __future__ import annotations

import jax
import jax.numpy as jnp

from world_marl.algs.gae import compute_gae
from world_marl.algs.ippo import (
    IPPOConfig,
    RolloutBatch,
    action_nll,
    create_train_state,
    ppo_update,
    select_actions,
    tree_l2_distance,
)


def _synthetic_batch(state, key, *, steps=4, actors=4, obs_shape=(8, 8, 3)):
    obs = jax.random.uniform(key, (steps, actors, *obs_shape))
    flat_obs = obs.reshape((steps * actors, *obs_shape))
    action_key, reward_key = jax.random.split(key)
    actions, log_probs, values = select_actions(state, action_key, flat_obs)
    rewards = jax.random.normal(reward_key, (steps, actors))
    return RolloutBatch(
        observations=obs,
        actions=actions.reshape((steps, actors)),
        log_probs=log_probs.reshape((steps, actors)),
        rewards=rewards,
        dones=jnp.zeros((steps, actors)),
        values=values.reshape((steps, actors)),
    )


def test_ppo_update_changes_parameters():
    config = IPPOConfig(update_epochs=2, num_minibatches=2, learning_rate=1e-3)
    state = create_train_state(jax.random.PRNGKey(0), (8, 8, 3), 3, config)
    batch = _synthetic_batch(state, jax.random.PRNGKey(1))
    new_state, metrics = ppo_update(
        state,
        batch,
        jnp.zeros((4,)),
        jax.random.PRNGKey(2),
        config,
    )
    assert tree_l2_distance(state.params, new_state.params) > 0.0
    assert "total_loss" in metrics


def test_ppo_update_reduces_synthetic_surrogate_action_loss():
    config = IPPOConfig(
        update_epochs=1,
        num_minibatches=1,
        learning_rate=5e-3,
        ent_coef=0.0,
        vf_coef=0.0,
        clip_eps=10.0,
        max_grad_norm=10.0,
    )
    state = create_train_state(jax.random.PRNGKey(0), (8, 8, 3), 3, config)
    batch = _synthetic_batch(state, jax.random.PRNGKey(3), steps=2, actors=8)
    last_values = jnp.zeros((8,))

    advantages, _ = compute_gae(
        batch.rewards,
        batch.values,
        batch.dones,
        last_values,
        config.gamma,
        config.gae_lambda,
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def weighted_action_loss(train_state):
        flat_obs = batch.observations.reshape((16, 8, 8, 3))
        flat_actions = batch.actions.reshape((16,))
        policy, _ = train_state.apply_fn({"params": train_state.params}, flat_obs)
        return -jnp.mean(policy.log_prob(flat_actions) * advantages.reshape((16,)))

    before = weighted_action_loss(state)
    new_state, _ = ppo_update(
        state,
        batch,
        last_values,
        jax.random.PRNGKey(4),
        config,
    )
    after = weighted_action_loss(new_state)
    assert after < before


def test_policy_forward_on_synthetic_observations():
    config = IPPOConfig()
    state = create_train_state(jax.random.PRNGKey(0), (8, 8, 3), 3, config)
    observations = jnp.ones((5, 8, 8, 3), dtype=jnp.float32)
    actions, log_probs, values = select_actions(
        state,
        jax.random.PRNGKey(1),
        observations,
    )
    assert actions.shape == (5,)
    assert log_probs.shape == (5,)
    assert values.shape == (5,)
    assert action_nll(state, observations, actions).shape == ()
