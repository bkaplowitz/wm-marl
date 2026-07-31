from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from world_marl.dreamarl.losses import (
    encoder_params,
    update_ema,
    world_model_loss,
)
from world_marl.dreamarl.world_model import DreaMARLWorldModel

from test_dreamarl_world_model import _batch, _config


def _squared_norm(tree) -> jax.Array:
    return sum(jnp.sum(jnp.square(value)) for value in jax.tree.leaves(tree))


def test_world_model_objective_is_finite_and_routes_encoder_gradients() -> None:
    config = _config()
    batch = _batch()
    model = DreaMARLWorldModel(config)
    key = jax.random.PRNGKey(30)
    params = model.init(key, batch, key)["params"]
    target = encoder_params(params)

    output = world_model_loss(model, params, target, batch, key, config)
    assert bool(jnp.isfinite(output.loss))
    assert set(output.metrics) >= {
        "loss",
        "loss/jepa",
        "loss/dynamics_kl",
        "loss/representation_kl",
        "loss/team_reward",
        "loss/continuation",
    }

    online_grad = jax.grad(
        lambda online: world_model_loss(
            model, online, target, batch, key, config
        ).loss
    )(params)
    target_grad = jax.grad(
        lambda target_params: world_model_loss(
            model, params, target_params, batch, key, config
        ).loss
    )(target)
    assert float(_squared_norm(online_grad)) > 0.0
    np.testing.assert_allclose(float(_squared_norm(target_grad)), 0.0, atol=0.0)


def test_masked_targets_do_not_change_loss() -> None:
    config = _config()
    batch = _batch()._replace(valid=jnp.ones((6, 3), bool).at[-1].set(False))
    model = DreaMARLWorldModel(config)
    key = jax.random.PRNGKey(31)
    params = model.init(key, batch, key)["params"]
    target = encoder_params(params)
    baseline = world_model_loss(model, params, target, batch, key, config).loss
    changed = batch._replace(
        next_observations=batch.next_observations.at[-1].set(1e6),
        rewards=batch.rewards.at[-1].set(1e6),
        team_rewards=batch.team_rewards.at[-1].set(1e6),
        is_terminal=batch.is_terminal.at[-1].set(True),
    )
    perturbed = world_model_loss(model, params, target, changed, key, config).loss
    np.testing.assert_allclose(perturbed, baseline, atol=1e-6)


def test_ema_endpoints_are_exact() -> None:
    online = {"weight": jnp.array([2.0, 4.0])}
    target = {"weight": jnp.array([0.0, 2.0])}
    np.testing.assert_allclose(update_ema(target, online, 0.0)["weight"], online["weight"])
    np.testing.assert_allclose(update_ema(target, online, 1.0)["weight"], target["weight"])
