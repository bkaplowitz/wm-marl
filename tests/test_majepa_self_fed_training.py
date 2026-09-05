"""Last-step recurrent CTDE supervision: boundaries, gradients, and integration."""

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest
import elements
import embodied.jax.nets as nn
import embodied.jax.outs as outs
from types import SimpleNamespace

from majepa.marl.core import MARLCore
from majepa.main import _load_configs, _resolve_config_profiles
from majepa.training import self_fed as self_fed_module
from majepa.training.self_fed import (
    endpoint_validity,
    gather_at,
    mixed_posterior_kl,
    self_fed_losses,
    frozen_posterior,
    scatter_sample_mean,
)
from majepa.training.ctde import TwoStepAnchors
from majepa.marl.axes import TeamAxis
from test_majepa_learner_smoke import _tiny_learner, _synthetic_replay, _assert_finite


def test_endpoint_masks_keep_terminal_arrival_and_post_death_corrections():
    first = jnp.array([[1, 0, 0, 0, 1, 0, 0]], bool)
    last = jnp.array([[0, 0, 0, 1, 0, 0, 1]], bool)
    present = jnp.ones((1, 7, 2), bool)
    alive = present.at[0, 2:4, 0].set(False)
    team, local = endpoint_validity(first, last, present, alive, 5)
    np.testing.assert_array_equal(team[0, 0, :, 0], [1, 1, 1, 0, 0])
    np.testing.assert_array_equal(local[0, 0, :, 0], [1, 1, 1, 0, 0])
    # Root1's H2 is terminal, even though H4/H5 cross a reset. Keep its sample.
    assert bool(local[0, 1, 1].all())
    assert not bool(local[0, 1, 2:].any())
    # A dead root slot still shares team reward but is not a local cohort member.
    assert bool(team[0, 2, 0, 0]) and not bool(local[0, 2, 0, 0])
    assert not bool(team[0, 3].any())
    np.testing.assert_array_equal(team[0, 4, :, 0], [1, 1, 0, 0, 0])
    assert not bool(team[0, 6].any())


def test_endpoint_absence_cannot_reenter_same_path_and_safe_gather_is_aligned():
    first = jnp.array([[1, 0, 0, 0, 0, 0]], bool)
    last = jnp.zeros_like(first)
    present = jnp.ones((1, 6, 2), bool).at[0, 2, 0].set(False)
    team, _ = endpoint_validity(first, last, present, present, 5)
    np.testing.assert_array_equal(team[0, 0, :, 0], [1, 0, 0, 0, 0])
    assert bool(team[0, 0, :, 1].all())
    anchors = TwoStepAnchors(jnp.array([0, 0]), jnp.array([1, 5]), jnp.ones(2, bool))
    values = jnp.array([[11, 22, 33, 44, 55, 66]])
    np.testing.assert_array_equal(gather_at(values, anchors, 2), [44, 66])
    assert not bool(team[0, 5].any())  # Clipped gather is not a valid target.


def test_consumer_kl_uses_executable_unimix_and_stops_teacher():
    logits = jnp.array([[[10.0, -10.0], [-10.0, 10.0]]])
    target = -logits
    assert float(mixed_posterior_kl(logits, logits, 0.01)[0]) == 0
    assert float(mixed_posterior_kl(logits, target, 0.01)[0]) < float(
        mixed_posterior_kl(logits, target, 0.0)[0]
    )
    teacher_grad = jax.grad(
        lambda value: mixed_posterior_kl(logits, value, 0.01).sum()
    )(target)
    np.testing.assert_array_equal(teacher_grad, 0)


class ToyJoint(nj.Module):
    per_agent: bool = False
    dimension: int = 1

    def sequence(self, cache, states, actions, present, alive, reset, training):
        assert not training  # Must not reuse the training/dropout snapshot path.
        batch, length, agents = actions.shape
        return cache, {}, {"memory": jnp.zeros((batch * agents, length, 1))}

    def step(self, cache, state, action, present, alive, reset, training):
        assert not training
        weight = self.value(
            "weight",
            lambda: jnp.ones(action.shape[-1]) if self.per_agent else jnp.float32(1),
        )
        if self.per_agent:
            weight = weight[..., None]
        embedding = state[..., :1].astype(jnp.float32) + weight * action[..., None]
        if self.dimension > 1:
            embedding = jnp.concatenate(
                [embedding, jnp.ones((*embedding.shape[:-1], self.dimension - 1))], -1
            )
        return cache, {"embedding": embedding, "hidden": embedding}


class ToyDynamics(nj.Module):
    unimix = 0.01

    def advance(self, carry, action, training, active):
        assert not training
        gain = self.value("gain", lambda: jnp.float32(1))
        return carry, gain * carry["deter"]

    def posterior(self, embedding, deter):
        gain = self.value("gain", lambda: jnp.float32(1))
        x = gain * embedding.astype(jnp.float32) + deter.astype(jnp.float32)
        return jnp.stack([x, -x], -1)

    def complete_from_observation(self, cache, deter, embedding, sample):
        assert sample is True
        logits = self.posterior(embedding, deter)
        stoch = jax.nn.one_hot(jax.random.categorical(nj.seed(), logits), 2)
        gain = self.value("gain", lambda: jnp.float32(1))
        carry = nn.cast(dict(cache, deter=gain * embedding, stoch=stoch))
        return carry, {"logit": logits}


class ToyHead(nj.Module):
    binary: bool = False
    vector: int = 0
    per_agent: bool = False

    def __call__(self, hidden, bdims):
        del bdims
        weight = self.value(
            "weight",
            lambda: jnp.ones(hidden.shape[-2]) if self.per_agent else jnp.float32(1),
        )
        value = weight * hidden[..., 0].astype(jnp.float32)
        if self.vector:
            value = jnp.broadcast_to(value[..., None], (*value.shape, self.vector))
        if self.binary:
            return outs.Binary(value)
        return SimpleNamespace(
            pred=lambda: value, loss=lambda target: (value - target) ** 2
        )


def toy_case(length=3, horizons=(2,), *, per_agent=False, dimension=1):
    team = TeamAxis(2)
    agent = SimpleNamespace(
        team=team,
        dyn=ToyDynamics(name="dyn"),
        ctde_joint=ToyJoint(per_agent=per_agent, dimension=dimension, name="joint"),
        ctde_rew=ToyHead(per_agent=per_agent, name="rew"),
        ctde_con=ToyHead(binary=True, per_agent=per_agent, name="con"),
        ctde_mask=ToyHead(binary=True, vector=8, per_agent=per_agent, name="mask"),
        ctde_alive=ToyHead(binary=True, per_agent=per_agent, name="alive"),
        ctde_action_key="action",
        action_mask_reduction="balanced",
        config=elements.Config(
            marl={
                "ctde": {
                    "self_fed": {
                        "horizons": list(horizons),
                        "anchors": 1,
                        "consumer_kl_scale": 0,
                    }
                }
            },
            contdisc=True,
            horizon=10,
        ),
    )
    agent._present = lambda obs: obs["agent_present"]
    agent._controllable = lambda obs: obs["controllable_alive"]
    agent.validity = agent._present
    agent.feat2tensor = lambda features: jnp.concatenate(
        [
            features["deter"],
            features["stoch"].reshape((*features["deter"].shape[:-1], -1)),
        ],
        -1,
    )
    flags = jnp.ones((2, length), bool)
    first = jnp.zeros_like(flags).at[:, 0].set(True)
    obs = {
        "agent_present": flags,
        "controllable_alive": flags,
        "is_first": first,
        "is_last": jnp.zeros_like(flags).at[:, -1].set(True),
        "is_terminal": jnp.zeros_like(flags).at[:, -1].set(True),
        "reward": jnp.zeros((2, length)).at[:, 1].set(10),
        "action_mask": jnp.ones((2, length, 8), bool),
    }
    entries = {
        "deter": jnp.zeros((2, length, dimension)),
        "stoch": jnp.zeros((2, length, dimension, 2)),
        "keys": jnp.zeros((2, length, 1, 1)),
        "values": jnp.zeros((2, length, 1, 1)),
        "valid": flags[..., None],
        "position": jnp.broadcast_to(jnp.arange(length), (2, length)),
        "ctde_joint_carry": {"memory": jnp.zeros((2, 1))},
    }
    features = {key: entries[key] for key in ("deter", "stoch")}
    tokens = jnp.ones((2, length, dimension)) * 3
    action = {
        "action": jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (2, length))
    }
    return agent, tokens, features, entries, obs, action


def test_factual_actions_and_arrival_reward_align_and_gradient_is_last_step_only():
    agent, tokens, features, entries, obs, action = toy_case()

    def loss():
        losses, _ = self_fed_losses(
            agent, tokens, features, entries, tokens, obs, action
        )
        return losses["ctde_self_fed_reward"].mean()

    state = nj.init(loss)({}, seed=21)
    pure = nj.pure(loss)
    value, grad = jax.value_and_grad(lambda params: pure(params, seed=22)[1])(state)
    # a_t=1, a_(t+1)=2 gives imagined value3; arrival reward at t+2 is0.
    assert float(value) == 9.0
    # Last-step derivative is2*3*a_(t+1)=12. Backprop through first feed gives18.
    assert float(grad["joint/weight"]) == 12.0
    assert float(grad["dyn/gain"]) == 0.0


def test_team_loss_scatter_preserves_dead_head_gradient_and_mean():
    anchors = TwoStepAnchors(jnp.array([0]), jnp.array([0]), jnp.ones(1, bool))
    destination = jnp.array([[[False, True], [False, True], [False, True]]])
    valid = jnp.ones((1, 2), bool)

    def objective(head_outputs, team_loss):
        grid = scatter_sample_mean(
            head_outputs**2, anchors, valid, destination, team_loss=team_loss
        )
        return (grid * destination).sum() / destination.sum()

    outputs = jnp.array([[2.0, 3.0]])
    assert float(objective(outputs, True)) == 6.5  # Both heads' mean, not just9.
    np.testing.assert_array_equal(jax.grad(objective)(outputs, True), [[2, 3]])
    assert float(objective(outputs, False)) == 9
    np.testing.assert_array_equal(jax.grad(objective)(outputs, False), [[0, 6]])


def test_dead_root_outcome_heads_train_but_local_embedding_and_mask_do_not():
    agent, tokens, features, entries, obs, action = toy_case(
        per_agent=True, dimension=2
    )
    obs = dict(obs, controllable_alive=obs["controllable_alive"].at[0].set(False))
    agent.validity = lambda data: data["controllable_alive"]

    def objective(name):
        losses, _ = self_fed_losses(
            agent, tokens, features, entries, tokens, obs, action
        )
        valid = agent.validity(obs)
        return (losses[f"ctde_self_fed_{name}"] * valid).sum() / valid.sum()

    state = nj.init(lambda: objective("reward"))({}, seed=91)
    pure = nj.pure(objective)
    for loss_name, head_name in (("reward", "rew"), ("continuation", "con")):
        gradients = jax.grad(lambda params: pure(params, loss_name, seed=92)[1])(state)
        assert bool((jnp.abs(gradients[f"{head_name}/weight"]) > 0).all())
    for loss_name, head_name in (("embedding", "joint"), ("action_mask", "mask")):
        gradients = jax.grad(lambda params: pure(params, loss_name, seed=93)[1])(state)
        assert float(gradients[f"{head_name}/weight"][0]) == 0
        assert float(jnp.abs(gradients[f"{head_name}/weight"][1])) > 0


def test_factual_death_does_not_remove_later_alive_or_mask_corrections():
    agent, tokens, features, entries, obs, action = toy_case(per_agent=True)
    dead = obs["controllable_alive"].at[0, 1:].set(False)
    mask = obs["action_mask"].at[0, 1:, 1:].set(False)
    obs = dict(obs, controllable_alive=dead, action_mask=mask)

    def objective(name):
        losses, _ = self_fed_losses(
            agent, tokens, features, entries, tokens, obs, action
        )
        return losses[f"ctde_self_fed_{name}"].mean()

    state = nj.init(lambda: objective("alive"))({}, seed=94)
    pure = nj.pure(objective)
    for loss_name, head_name in (("alive", "alive"), ("action_mask", "mask")):
        gradients = jax.grad(lambda params: pure(params, loss_name, seed=95)[1])(state)
        # The root-live unit actually died at H1; its still-positive predicted
        # liveness/mask at H2 must continue receiving a downward correction.
        assert float(gradients[f"{head_name}/weight"][0]) > 0


def test_future_reset_data_cannot_change_valid_endpoint_losses_or_gradients(
    monkeypatch,
):
    monkeypatch.setattr(
        self_fed_module,
        "sample_two_step_anchors",
        lambda *_: TwoStepAnchors(
            jnp.array([0]),
            jnp.array([0]),
            jnp.ones(1, bool),
        ),
    )
    agent, tokens, features, entries, obs, action = toy_case(
        length=6, horizons=(2, 4, 5)
    )
    obs = dict(
        obs,
        is_first=obs["is_first"].at[:, 3].set(True),
        is_last=obs["is_last"].at[:, 2].set(True),
        is_terminal=obs["is_terminal"].at[:, 2].set(True),
        reward=obs["reward"].at[:, 2].set(4),
    )

    def objective(data, tokens, action):
        losses, metrics = self_fed_losses(
            agent, tokens, features, entries, tokens, data, action
        )
        return sum(value.mean() for value in losses.values()), metrics

    state = nj.init(objective)({}, obs, tokens, action, seed=96)
    pure = nj.pure(objective)
    differentiate = jax.value_and_grad(
        lambda params, data, tok, act: pure(
            params,
            data,
            tok,
            act,
            seed=97,
        )[1],
        has_aux=True,
    )
    (value, metrics), gradients = differentiate(state, obs, tokens, action)
    changed_obs = dict(
        obs,
        reward=obs["reward"].at[:, 3:].set(100),
        action_mask=obs["action_mask"].at[:, 3:, 1:].set(False),
        controllable_alive=obs["controllable_alive"].at[:, 3:].set(False),
    )
    (changed_value, _), changed_gradients = differentiate(
        state,
        changed_obs,
        tokens.at[:, 3:].set(-100),
        {"action": action["action"].at[:, 3:].set(7)},
    )
    np.testing.assert_array_equal(value, changed_value)
    for key in gradients:
        np.testing.assert_array_equal(gradients[key], changed_gradients[key])
    assert float(metrics["ctde/self_fed_train_h2/team_count"]) == 2
    assert float(metrics["ctde/self_fed_train_h2/reward_loss"]) == 1
    assert float(metrics["ctde/self_fed_train_h4/team_count"]) == 0
    assert float(metrics["ctde/self_fed_train_h5/team_count"]) == 0


def test_frozen_consumer_keeps_input_gradient_and_blocks_params_state_and_teacher():
    dyn = ToyDynamics(name="dyn")
    embedding, deter, target = (
        jnp.array([[0.5]]),
        jnp.array([[0.1]]),
        jnp.array([[1.0]]),
    )
    state = nj.init(dyn.posterior)({}, embedding, deter, seed=1)

    def objective(embedding, deter, target):
        return mixed_posterior_kl(
            frozen_posterior(dyn, embedding, deter),
            frozen_posterior(dyn, target, deter),
            dyn.unimix,
        ).sum()

    pure = nj.pure(objective)
    gradients = jax.grad(
        lambda params, e, d, t: pure(params, e, d, t, seed=2)[1], argnums=(0, 1, 2, 3)
    )(
        state,
        embedding,
        deter,
        target,
    )
    assert float(jnp.abs(gradients[1]).sum()) > 0
    for tree in (gradients[0], gradients[2], gradients[3]):
        for value in jax.tree.leaves(tree):
            np.testing.assert_array_equal(value, 0)


def make_tiny(*, enabled, consumer=0.0, scale=0.1, length=4, anchors=2):
    original, observations, actions = _tiny_learner()
    config = original.config.update(
        {
            "marl.ctde.self_fed.enabled": enabled,
            "marl.ctde.self_fed.anchors": anchors,
            "marl.ctde.self_fed.consumer_kl_scale": consumer,
            "marl.ctde.self_fed.scale": scale,
            "batch_length": length,
        }
    )
    model = object.__new__(MARLCore)
    MARLCore.__init__(model, observations, actions, config)
    return model, observations, actions


@pytest.mark.parametrize("name", ["scale", "consumer"])
@pytest.mark.parametrize("value", [-0.1, float("inf"), float("nan")])
def test_nonfinite_or_negative_scales_are_rejected(name, value):
    with pytest.raises(ValueError, match="finite nonnegative scales"):
        make_tiny(enabled=True, **{name: value})


def test_calibrated_mask_configuration_cannot_silently_change_exposure_path():
    learner, observations, actions = make_tiny(enabled=True)
    config = learner.config.update({"marl.ctde.mask_calibration.enabled": True})
    model = object.__new__(MARLCore)
    with pytest.raises(ValueError, match="baseline hard CTDE mask path"):
        MARLCore.__init__(model, observations, actions, config)


@pytest.mark.parametrize("values", [["2,4,5"], ["2", "4", "5"]])
def test_experiment_horizons_parse_with_maintained_config_loader(values):
    config = _resolve_config_profiles(
        _load_configs(), ("smac_vector", "ma_jepa", "debug")
    )
    parsed = elements.Flags(config).parse(
        [
            "--agent.marl.ctde.self_fed.enabled",
            "True",
            "--agent.marl.ctde.self_fed.horizons",
            *values,
            "--agent.marl.ctde.self_fed.scale",
            "0.1",
        ]
    )
    assert parsed.agent.marl.ctde.self_fed.enabled
    assert tuple(parsed.agent.marl.ctde.self_fed.horizons) == (2, 4, 5)


def test_full_tiny_learner_jit_with_optional_recurrent_auxiliary():
    learner, observations, actions = make_tiny(
        enabled=True, consumer=0.1, length=6, anchors=8
    )
    data = _synthetic_replay(learner, observations, actions)
    data = {
        key: jnp.concatenate([value, jnp.repeat(value[:, -1:], 2, axis=1)], axis=1)
        for key, value in data.items()
    }
    # Six optimized states after the four-state prefix. Team0 has a valid H5
    # arrival at the final terminal state, including rewards after unit0 dies.
    for key in data:
        if key.endswith(("is_last", "is_terminal")):
            data[key] = data[key].at[:, 7:].set(False).at[:, 9].set(True)
    data = dict(data, _environment_step=jnp.zeros((2, 10), jnp.int32))
    carry = learner.init_train(2)
    state = nj.init(learner.train)({}, carry, data, seed=981)
    train = jax.jit(nj.pure(learner.train))
    updated, (carry, _, metrics) = train(state, carry, data, seed=982)
    _assert_finite((updated, metrics))
    assert 0 < float(metrics["ctde/self_fed_train/anchors"]) <= 8
    for horizon in (2, 4, 5):
        assert float(metrics[f"ctde/self_fed_train_h{horizon}/team_count"]) > 0
        assert float(metrics[f"ctde/self_fed_train_h{horizon}/embedding_loss"]) > 0
    assert "loss/ctde_self_fed_consumer_kl" in metrics
    assert float(metrics["ppo/active"]) == 0  # Auxiliary is a world-model update.
    assert float(metrics["opt/actor/updates"]) == 0
    data = dict(data, _environment_step=jnp.full((2, 10), 10, jnp.int32))
    updated, (_, _, active_metrics) = train(updated, carry, data, seed=983)
    _assert_finite((updated, active_metrics))
    assert float(active_metrics["ppo/active"]) == 1
    assert float(active_metrics["opt/actor/updates"]) > 0
    assert float(active_metrics["ctde/self_fed_train_h5/team_count"]) > 0


def test_default_off_and_enabled_zero_scale_preserve_initialization_and_full_train():
    baseline, observations, actions = _tiny_learner()
    # Historical configs lacking this option remain valid and executable.
    config = elements.Config(
        {
            key: value
            for key, value in baseline.config.flat.items()
            if not key.startswith("marl.ctde.self_fed.")
        }
    )
    absent = object.__new__(MARLCore)
    MARLCore.__init__(absent, observations, actions, config)
    zero, _, _ = make_tiny(enabled=True, scale=0, consumer=0.3)
    data = _synthetic_replay(baseline, observations, actions)
    data = dict(data, _environment_step=jnp.full((2, 8), 10, jnp.int32))
    carry = baseline.init_train(2)

    def exact_tree(left, right):
        left_values, left_def = jax.tree.flatten(left)
        right_values, right_def = jax.tree.flatten(right)
        assert left_def == right_def
        for left_value, right_value in zip(left_values, right_values):
            np.testing.assert_array_equal(left_value, right_value)

    reference_state, reference_result = None, None
    for learner in (baseline, absent, zero):
        assert not learner.ctde_self_fed_enabled
        assert not any(name.startswith("ctde_self_fed_") for name in learner.scales)
        initialized = nj.init(learner.train)({}, carry, data, seed=988)
        result = jax.jit(nj.pure(learner.train))(initialized, carry, data, seed=989)
        _assert_finite(result)
        assert not any(
            "self_fed_train" in name or "loss/ctde_self_fed_" in name
            for name in result[1][2]
        )
        if reference_state is None:
            reference_state, reference_result = initialized, result
        else:
            exact_tree(reference_state, initialized)
            exact_tree(reference_result, result)


def test_auxiliary_gradients_are_joint_only_including_frozen_consumer_posterior():
    learner, observations, actions = make_tiny(enabled=True, consumer=0.1)
    data = _synthetic_replay(learner, observations, actions)
    carry = learner.init_train(2)
    state = nj.init(learner.train)({}, carry, data, seed=983)
    primary = {
        key: value
        for key, value in data.items()
        if not key.startswith("_behavior_replay/")
    }
    local_data = learner.team.local_sequence_data(primary)
    local_carry = learner.team.fold_tree_batch(carry)

    def objective():
        replay_carry, obs, prevact, _ = learner._apply_replay_context(
            local_carry, local_data
        )
        _, entries, tokens, features, _, _, ema = learner._world_model_terms(
            replay_carry,
            obs,
            prevact,
            training=False,
        )
        losses, metrics = self_fed_losses(
            learner, tokens, features, entries[1], ema, obs, prevact
        )
        return sum(
            value.mean() * learner.scales[name] for name, value in losses.items()
        ), metrics

    pure = nj.pure(objective)

    def differentiate(params):
        return pure(params, seed=984, create=False)[1]

    (_, metrics), gradients = jax.jit(
        jax.value_and_grad(differentiate, has_aux=True, allow_int=True)
    )(state)
    _assert_finite(metrics)
    magnitude = {}
    for key, value in gradients.items():
        if str(value.dtype) == "[('float0', 'V')]":
            continue
        norm = float(jnp.abs(value).sum())
        magnitude[key.split("/")[0]] = magnitude.get(key.split("/")[0], 0) + norm
    for prefix in (
        "enc",
        "dyn",
        "pol",
        "ctde_val",
        "ctde_teammate_actor",
        "ctde_teammate_belief",
        "slowenc",
        "slowctde_val",
    ):
        assert magnitude.get(prefix, 0.0) == 0.0, (prefix, magnitude)
    for prefix in ("ctde_joint", "ctde_rew", "ctde_con", "ctde_mask", "ctde_alive"):
        assert magnitude[prefix] > 0.0, (prefix, magnitude)
