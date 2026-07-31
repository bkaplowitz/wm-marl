from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from world_marl.dreamarl.control import (
    CentralizedCritic,
    SharedActor,
    actor_loss,
    critic_loss,
    lambda_returns,
    survival_weights,
)
from world_marl.dreamarl.losses import encoder_params, world_model_loss
from world_marl.dreamarl.world_model import DreaMARLWorldModel

from test_dreamarl_world_model import _batch, _config


def _squared_norm(tree) -> jax.Array:
    return sum(jnp.sum(jnp.square(value)) for value in jax.tree.leaves(tree))


def _initialized_control():
    config = _config()
    batch = _batch()
    model = DreaMARLWorldModel(config)
    key = jax.random.PRNGKey(40)
    params = model.init(key, batch, key)["params"]
    inference = world_model_loss(
        model, params, encoder_params(params), batch, key, config
    ).inference
    state = inference.final_state
    belief = model.apply({"params": params}, state, method=model.belief)
    actor = SharedActor(config)
    actor_params = actor.init(
        key, belief, state.agent_alive, state.action_mask
    )["params"]
    critic = CentralizedCritic(config)
    critic_params = critic.init(key, belief, state.agent_alive)["params"]
    return config, model, params, state, actor, actor_params, critic, critic_params


def test_actor_masks_illegal_and_inactive_actions() -> None:
    config, model, params, state, actor, actor_params, _, _ = _initialized_control()
    del model, params
    belief = jnp.zeros((3, config.max_agents, config.belief_dim))
    alive = state.agent_alive.at[:, 1].set(False)
    mask = jnp.ones((3, config.max_agents, config.action_dim), bool)
    mask = mask.at[:, 0, 2:].set(False)
    logits = actor.apply({"params": actor_params}, belief, alive, mask)
    assert bool(jnp.all(logits[:, 0, 2:] < -1e20))
    assert bool(jnp.all(logits[:, 1, 1:] < -1e20))
    assert bool(jnp.all(jnp.isfinite(logits[:, 1, 0])))


def test_joint_imagination_has_finite_actor_and_critic_gradients() -> None:
    config, model, params, state, actor, actor_params, critic, critic_params = (
        _initialized_control()
    )
    key = jax.random.PRNGKey(41)
    output = actor_loss(
        actor,
        actor_params,
        critic,
        critic_params,
        model,
        params,
        state,
        key,
        config,
    )
    horizon = config.imagination.horizon
    assert output.imagination.actions.shape == (horizon, 3, config.max_agents)
    assert output.imagination.returns.shape == (horizon, 3)
    assert bool(jnp.isfinite(output.loss))

    actor_grad = jax.grad(
        lambda current: actor_loss(
            actor,
            current,
            critic,
            critic_params,
            model,
            params,
            state,
            key,
            config,
        ).loss
    )(actor_params)
    value_loss, _ = critic_loss(critic, critic_params, output.imagination)
    critic_grad = jax.grad(
        lambda current: critic_loss(critic, current, output.imagination)[0]
    )(critic_params)
    assert bool(jnp.isfinite(value_loss))
    assert float(_squared_norm(actor_grad)) > 0.0
    assert float(_squared_norm(critic_grad)) > 0.0


def test_lambda_returns_and_survival_weights_match_manual_values() -> None:
    rewards = jnp.array([[1.0], [2.0], [3.0]])
    discounts = jnp.array([[0.9], [0.8], [0.0]])
    values = jnp.array([[10.0], [20.0], [30.0]])
    bootstrap = jnp.array([40.0])
    actual = lambda_returns(rewards, discounts, values, bootstrap, 0.5)
    expected_2 = 3.0
    expected_1 = 2.0 + 0.8 * (0.5 * 30.0 + 0.5 * expected_2)
    expected_0 = 1.0 + 0.9 * (0.5 * 20.0 + 0.5 * expected_1)
    np.testing.assert_allclose(
        actual[:, 0], [expected_0, expected_1, expected_2], atol=1e-6
    )
    np.testing.assert_allclose(
        survival_weights(discounts)[:, 0], [1.0, 0.9, 0.72], atol=1e-6
    )
