"""Gradient routing for optional value and joint representation supervision."""

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from majepa.main import _load_configs, _resolve_config_profiles
from test_training import _tiny_learner, _synthetic_replay, _assert_finite


def _setup(value_scale=0.0, joint=False):
    learner, observations, actions = _tiny_learner(
        {
            "agent.world_model_gradients.critic_value_scale": value_scale,
            "agent.world_model_gradients.joint_prediction": joint,
            "agent.marl.ctde.critic.outscale": 1.0,
            "agent.loss_scales.ctde_posterior_alignment": 0.05,
            "agent.marl.ctde.self_fed.bptt_steps": 2,
            "agent.marl.ctde.self_fed.trajectory_kl_scale": 0.1,
        }
    )
    data = _synthetic_replay(learner, observations, actions)
    if value_scale:
        data["behavior_logprob"] = jnp.where(
            data["controllable_alive"], -jnp.log(7.0), 0.0
        )
        data["_behavior_replay/behavior_logprob"] = data["behavior_logprob"]
    data["_environment_step"] = jnp.full((2, 8), 10, jnp.int32)
    carry = learner.init_train(2)
    state = nj.init(learner.train)({}, carry, data, seed=1201)
    return learner, data, carry, state


def _loss_inputs(learner, data, carry):
    data = {k: v for k, v in data.items() if not k.startswith("_behavior_replay/")}
    carry, obs, prevact, _ = learner._apply_replay_context(
        learner.team.fold_tree_batch(carry), learner.team.local_sequence_data(data)
    )
    _, entries, tokens, features, _, _, targets = learner._world_model_terms(
        carry, obs, prevact, training=True
    )
    return tokens, features, entries[1], targets, obs, prevact


def _gradient_norms(grads):
    norms = {}
    for key, value in grads.items():
        if value.dtype == jax.dtypes.float0:
            continue
        root = key.split("/")[0]
        norms[root] = norms.get(root, 0.0) + float(jnp.abs(value).sum())
    return norms


def test_gradient_options_default_off():
    cfg = _resolve_config_profiles(
        _load_configs(), ("smac_vector", "ma_jepa", "localmask_reference")
    )
    assert cfg.agent.world_model_gradients.critic_value_scale == 0.0
    assert cfg.agent.world_model_gradients.joint_prediction is False


@pytest.mark.parametrize("scale", [-1.0, float("nan"), float("inf")])
def test_invalid_value_scale_fails_before_training(scale):
    with pytest.raises(ValueError, match="critic_value_scale"):
        _tiny_learner({"agent.world_model_gradients.critic_value_scale": scale})


@pytest.mark.parametrize(
    "value_scale,joint", [(0.0, False), (0.1, False), (0.0, True), (0.1, True)]
)
def test_gradient_treatments_train_with_finite_updates(value_scale, joint):
    learner, data, carry, state = _setup(value_scale, joint)
    updated, (_, _, metrics) = jax.jit(nj.pure(learner.train))(
        state, carry, data, seed=1202
    )
    _assert_finite((updated, metrics))
    assert float(metrics["opt/finite"]) == 1
    assert float(metrics["opt/local_world/updates"]) == 1
    assert float(metrics["opt/critic/updates"]) == 2
    assert ("loss/world_model_value" in metrics) == bool(value_scale)
    if value_scale:
        assert float(metrics["loss/world_model_value"]) > 0


@pytest.mark.parametrize("joint", [False, True])
def test_joint_losses_reach_local_model_only_when_enabled(joint):
    learner, data, carry, state = _setup(joint=joint)

    def objective():
        inputs = _loss_inputs(learner, data, carry)
        losses, _ = learner._ctde_replay_losses(*inputs, training=True)
        return jnp.stack(
            [
                losses[k].mean()
                for k in (
                    "ctde_embedding",
                    "ctde_posterior_alignment",
                    "ctde_self_fed_trajectory_kl",
                )
            ]
        )

    pure = nj.pure(objective)
    grads = jax.jit(
        jax.jacrev(lambda p: pure(p, seed=1203, create=False)[1], allow_int=True)
    )(state)
    for component in range(3):
        norms = _gradient_norms(jax.tree.map(lambda x: x[component], grads))
        assert norms["ctde_joint"] > 0
        assert (norms["enc"] > 0) == (joint and component != 2)
        assert (norms["dyn"] > 0) == joint
        assert norms.get("target_enc", 0) == 0
        assert norms.get("pol", 0) == 0
        assert norms.get("ctde_val", 0) == 0
        temporal = sum(
            float(jnp.abs(g[component]).sum())
            for k, g in grads.items()
            if k.startswith("dyn/temporal/")
        )
        assert (temporal > 0) == joint
        categorical = sum(
            float(jnp.abs(g[component]).sum())
            for k, g in grads.items()
            if k.startswith("dyn/obs")
        )
        assert (categorical > 0) == joint


def test_value_loss_updates_local_features_but_not_critic_parameters():
    learner, data, carry, state = _setup(value_scale=0.1)

    def objective():
        _, features, _, _, obs, _ = _loss_inputs(learner, data, carry)
        return learner._world_model_value_loss(features, obs)[0]

    pure = nj.pure(objective)
    loss, grads = jax.jit(
        jax.value_and_grad(
            lambda p: pure(p, seed=1204, create=False)[1], allow_int=True
        )
    )(state)
    assert float(loss) > 0
    norms = _gradient_norms(grads)
    assert norms["enc"] > 0 and norms["dyn"] > 0
    assert norms.get("pol", 0) == norms.get("ctde_joint", 0) == 0
    assert norms.get("target_enc", 0) == norms.get("slowctde_val", 0) == 0

    def update_world():
        return learner.opt(objective, skip_groups=("actor", "critic", "joint_world"))

    updated, metrics = jax.jit(nj.pure(update_world))(state, seed=1204)
    assert float(metrics["opt/local_world/grad_norm"]) > 0
    for key, value in state.items():
        if key.startswith(("pol/", "ctde_val/", "opt/actor", "opt/critic")):
            np.testing.assert_array_equal(updated[key], value)
    assert any(
        not np.array_equal(updated[k], v)
        for k, v in state.items()
        if k.startswith("enc/")
    )

    behavior = learner.team.local_sequence_data(
        {
            k.removeprefix("_behavior_replay/"): v
            for k, v in data.items()
            if k.startswith("_behavior_replay/")
        }
    )

    def ppo_only():
        batch, _ = learner._prepare_ppo_batch(behavior, jnp.float32(0.003))
        learner.opt.step_group("actor", learner._ppo_actor_loss, batch, has_aux=True)
        learner.opt.step_group("critic", learner._ppo_critic_loss, batch, has_aux=True)
        return batch

    ppo_updated, _ = jax.jit(nj.pure(ppo_only))(updated, seed=1207)
    world_prefixes = tuple(
        module.path + "/"
        for name in ("local_world", "joint_world")
        for module in learner.opt.groups[name][0]
    )
    for key, value in updated.items():
        if key.startswith((*world_prefixes, "opt/local_world", "opt/joint_world")):
            np.testing.assert_array_equal(ppo_updated[key], value)


def test_joint_gradient_changes_derivatives_without_changing_rollout_samples():
    from majepa.training.self_fed import local_transition, truncate_self_fed_state

    learner, _, _, state = _setup()

    def transition(enabled):
        return local_transition(
            learner.dyn,
            learner.dyn.initial(4),
            {"action": jnp.ones(4, jnp.int32)},
            jnp.ones((4, 8)),
            jnp.ones(4, bool),
            parameter_gradient=enabled,
        )

    frozen = nj.pure(lambda: transition(False))(state, seed=1205)[1]
    live = nj.pure(lambda: transition(True))(state, seed=1205)[1]
    for a, b in zip(jax.tree.leaves(frozen), jax.tree.leaves(live)):
        np.testing.assert_array_equal(a, b)
    for offset, expected in [(1, 0.0), (2, 1.0), (3, 0.0), (4, 1.0)]:
        gradient = jax.grad(lambda x: truncate_self_fed_state(x, offset, 2))(1.0)
        assert float(gradient) == expected


def test_campaign_resolves_independent_gradient_treatments():
    import json
    from pathlib import Path
    from majepa.campaign import plan

    spec = json.loads(
        (
            Path(__file__).parents[1] / "experiments/world-model-gradients.json"
        ).read_text()
    )
    resolved = plan(spec, "gradient-test")
    assert len(resolved["runs"]) == 12
    modes = {}
    baseline = resolved["runs"][0]["config"]
    for run in resolved["runs"]:
        config = run["config"]
        mode = (
            config["agent.world_model_gradients.critic_value_scale"],
            config["agent.world_model_gradients.joint_prediction"],
        )
        modes.setdefault(mode, []).append(config["seed"])
        ignored = {
            "seed",
            "logdir",
            "agent.world_model_gradients.critic_value_scale",
            "agent.world_model_gradients.joint_prediction",
        }
        assert {k: v for k, v in config.items() if k not in ignored} == {
            k: v for k, v in baseline.items() if k not in ignored
        }
    assert modes == {
        (0.0, False): [0, 1, 2],
        (0.1, False): [0, 1, 2],
        (0.0, True): [0, 1, 2],
        (0.1, True): [0, 1, 2],
    }


def test_collection_records_actual_mixed_probability_and_outgoing_action():
    from majepa.training.policy import apply_legal_unimix

    learner, data, carry, state = _setup(value_scale=0.1)
    key = "pol/head/action/logits/bias"
    state[key] = jnp.arange(8, dtype=state[key].dtype) * 0.3
    obs = {k: data[k][:, 5] for k in learner.obs_space}
    _, (_, action, output) = nj.pure(learner.policy)(state, carry, obs, seed=1206)

    def expected():
        enc, dyn, _, prevact = learner.team.fold_tree_batch(carry)
        local = learner.team.local_policy_data(obs)
        reset = local["is_first"]
        _, _, tokens = learner.enc(enc, local, reset, training=False, single=True)
        _, _, features, _ = learner.observe_dynamics(
            dyn, tokens, prevact, reset, local, training=False, single=True
        )
        policy = learner.policy_distribution(
            learner.feat2tensor(features), 1, action_mask=local["action_mask"]
        )
        policy = apply_legal_unimix(
            policy, learner.action_mask_key, learner.config.collection_unimix
        )
        return learner.team.unfold_batch(
            -policy["action"].loss(learner.team.fold_batch(action["action"]))
        )

    _, logprob = nj.pure(expected)(state, seed=1206)
    np.testing.assert_allclose(output["behavior_logprob"], logprob, atol=1e-6)
    primary = {k: v for k, v in data.items() if not k.startswith("_behavior_replay/")}
    local_data = learner.team.local_sequence_data(primary)
    local_data["action"] = jnp.broadcast_to(jnp.arange(8), local_data["action"].shape)
    replay = learner._replay_observations(local_data)
    np.testing.assert_array_equal(replay["_replay_action"], local_data["action"])
    assert "behavior_logprob" not in learner.obs_space
    assert "_replay_action" not in learner.obs_space
    baseline, _, _ = _tiny_learner()
    assert "behavior_logprob" not in baseline.ext_space
