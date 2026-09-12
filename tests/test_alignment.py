"""Integration and gradient boundaries for the two isolated latent treatments."""

import jax
import jax.numpy as jnp
import ninjax as nj

from test_training import _tiny_learner, _synthetic_replay, _assert_finite


def test_dense_alignment_updates_joint_producer_only():
    key = "ctde_posterior_alignment"
    learner, observations, actions = _tiny_learner(
        {
            f"agent.loss_scales.{key}": 0.1,
            "agent.marl.ctde.self_fed.horizons": [2],
            "agent.marl.ctde.self_fed.anchors": 2,
            "agent.marl.ctde.self_fed.scale": 0.1,
            "agent.marl.ctde.self_fed.trajectory_kl_scale": 0.1,
            "agent.marl.ctde.self_fed.bptt_steps": 2,
        }
    )
    data = _synthetic_replay(learner, observations, actions)
    data["_environment_step"] = jnp.full((2, 8), 10, jnp.int32)
    carry = learner.init_train(2)
    state = nj.init(learner.train)({}, carry, data, seed=961)
    primary = {k: v for k, v in data.items() if not k.startswith("_behavior_replay/")}
    local_data = learner.team.local_sequence_data(primary)
    local_carry = learner.team.fold_tree_batch(carry)

    def objective():
        replay_carry, obs, prevact, _ = learner._apply_replay_context(
            local_carry, local_data
        )
        _, entries, tokens, features, _, _, ema = learner._world_model_terms(
            replay_carry, obs, prevact, training=False
        )
        losses, _ = learner._ctde_replay_losses(
            tokens, features, entries[1], ema, obs, prevact, training=False
        )
        return losses[key].mean()

    pure = nj.pure(objective)
    value, grads = jax.jit(
        jax.value_and_grad(
            lambda params: pure(params, seed=962, create=False)[1], allow_int=True
        )
    )(state)
    assert float(value) > 0
    magnitude = {}
    for name, grad in grads.items():
        if str(grad.dtype) == "[('float0', 'V')]":
            continue
        root = name.split("/")[0]
        magnitude[root] = magnitude.get(root, 0.0) + float(jnp.abs(grad).sum())
    assert magnitude["ctde_joint"] > 0
    assert all(x == 0 for name, x in magnitude.items() if name != "ctde_joint"), (
        magnitude
    )

    updated, (_, _, metrics) = jax.jit(nj.pure(learner.train))(
        state, carry, data, seed=963
    )
    _assert_finite((updated, metrics))
    assert float(metrics[f"loss/{key}"]) > 0
    assert float(metrics["ppo/active"]) == 1
    assert float(metrics["opt/actor/updates"]) > 0
    assert float(metrics["ctde/self_fed_train_h2/trajectory_kl_loss"]) > 0
