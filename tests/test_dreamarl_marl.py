from __future__ import annotations

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from dreamarl.agent import Agent as LocalAgent
from dreamarl.main import _load_configs
from dreamarl.marl.axes import TeamAxis
from dreamarl.marl.core import MARLCore
from dreamarl.marl.spaces import add_agent_axis
from dreamarl.models.heads import MLPHead
from dreamarl.models.heads import apply_action_mask, apply_predicted_action_mask
from dreamarl.models.normalize import Normalize
from dreamarl.models.team import (
    AgentContextEncoder,
    TeamContentPredictor,
    TeamSlotEncoder,
    TeamSlotPredictor,
    masked_agent_coverage_loss,
    scale_gradient,
    team_set_matching_loss,
    team_slot_jepa_loss,
)
from dreamarl.training.learner import masked_mean


GLOBAL_KEYS = {"is_first", "is_last", "is_terminal", "consec", "stepid"}


def _assert_tree_equal(actual, expected) -> None:
    actual_leaves, actual_tree = jax.tree.flatten(actual)
    expected_leaves, expected_tree = jax.tree.flatten(expected)
    assert actual_tree == expected_tree
    for left, right in zip(actual_leaves, expected_leaves):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def _local_model_state(state):
    return {key: value for key, value in state.items() if not key.startswith("opt/")}


def _assert_mapping_subset_equal(actual, expected) -> None:
    assert set(expected).issubset(actual)
    _assert_tree_equal(
        {key: actual[key] for key in expected},
        expected,
    )


def _agent_config(num_agents: int = 1, marl_stage: str = "b0"):
    configs = _load_configs()
    resolved = elements.Config(configs["defaults"])
    resolved = resolved.update(configs["debug"])
    resolved = resolved.update(
        {
            "replay_context": 0,
            "agent.num_agents": num_agents,
            "agent.marl.stage": marl_stage,
            "agent.imag_length": 2,
            "agent.imag_last": 2,
        }
    )
    return elements.Config(
        **resolved.agent,
        logdir="/tmp/dreamarl-marl-core-test",
        seed=0,
        jax=resolved.jax,
        batch_size=resolved.batch_size,
        batch_length=resolved.batch_length,
        replay_context=0,
        report_length=resolved.report_length,
        replica=0,
        replicas=1,
    )


def _local_spaces(*, metadata: bool = True):
    observations = {
        "image": elements.Space(np.uint8, (64, 64, 3), 0, 256),
        "reward": elements.Space(np.float32, ()),
        "is_first": elements.Space(bool, ()),
        "is_last": elements.Space(bool, ()),
        "is_terminal": elements.Space(bool, ()),
    }
    if metadata:
        observations.update(
            agent_present=elements.Space(bool, ()),
            agent_alive=elements.Space(bool, ()),
            action_mask=elements.Space(bool, (4,)),
        )
    return observations, {"action": elements.Space(np.int32, (), 0, 4)}


def _team_spaces(team_size: int):
    observations, actions = _local_spaces(metadata=True)
    observations = {
        key: space if key in GLOBAL_KEYS else add_agent_axis(space, team_size)
        for key, space in observations.items()
    }
    actions = {key: add_agent_axis(space, team_size) for key, space in actions.items()}
    return observations, actions


def _local_batch(batch: int = 1, length: int = 4):
    return {
        "image": jax.random.randint(
            jax.random.key(1),
            (batch, length, 64, 64, 3),
            0,
            256,
            dtype=jnp.uint8,
        ),
        "reward": jax.random.normal(jax.random.key(2), (batch, length)),
        "agent_present": jnp.ones((batch, length), bool),
        "agent_alive": jnp.ones((batch, length), bool),
        "action_mask": jnp.ones((batch, length, 4), bool),
        "is_first": jnp.zeros((batch, length), bool).at[:, 0].set(True),
        "is_last": jnp.zeros((batch, length), bool),
        "is_terminal": jnp.zeros((batch, length), bool),
        "action": jax.random.randint(jax.random.key(3), (batch, length), 0, 4),
        "stepid": jnp.zeros((batch, length, 20), jnp.uint8),
        "consec": jnp.zeros((batch, length), jnp.int32),
    }


def _add_team_axis(data, team_size: int):
    return {
        key: (
            value
            if key in GLOBAL_KEYS
            else jnp.repeat(value[:, :, None, ...], team_size, axis=2)
        )
        for key, value in data.items()
    }


def test_team_axis_round_trip_preserves_identity() -> None:
    team = TeamAxis(3)
    values = np.arange(2 * 5 * 3 * 4).reshape(2, 5, 3, 4)
    np.testing.assert_array_equal(
        team.unfold_sequence(team.fold_sequence(values)), values
    )


def test_categorical_policy_head_applies_configured_unimix() -> None:
    features = jnp.array([[100.0, -100.0]], jnp.float32)
    action_space = {"action": elements.Space(np.int32, (), 0, 4)}

    def distribution(unimix):
        return MLPHead(
            action_space,
            {"action": "categorical"},
            layers=0,
            units=8,
            outscale=1.0,
            unimix=unimix,
            name="pol",
        )(features, 1)["action"]

    params = nj.init(lambda: distribution(0.5))({}, seed=80)
    mixed = nj.pure(lambda: distribution(0.5))(params, seed=81)[1]
    plain = nj.pure(lambda: distribution(0.0))(params, seed=81)[1]
    probabilities = jax.nn.softmax(mixed.logits, axis=-1)

    assert float(probabilities.min()) >= 0.5 / 4 - 1e-6
    assert not np.allclose(np.asarray(mixed.logits), np.asarray(plain.logits))


def test_imagination_start_grouping_preserves_environment_and_agent_identity() -> None:
    team = TeamAxis(3)
    batch, starts = 2, 4
    logical = np.arange(batch * team.size * starts * 2).reshape(
        batch, team.size, starts, 2
    )
    folded = logical.reshape(batch * team.size * starts, 2)
    grouped = team.group_starts(folded, starts)
    expected = logical.transpose(0, 2, 1, 3).reshape(batch * starts, team.size, 2)
    np.testing.assert_array_equal(grouped, expected)
    np.testing.assert_array_equal(team.ungroup_starts(grouped, starts), folded)


def test_shared_local_mapping_is_permutation_equivariant() -> None:
    team = TeamAxis(4)
    values = np.arange(2 * 4 * 3).reshape(2, 4, 3)
    permutation = np.array([2, 0, 3, 1])
    output = team.unfold_batch(3 * team.fold_batch(values) - 7)
    permuted = team.unfold_batch(3 * team.fold_batch(values[:, permutation]) - 7)
    np.testing.assert_array_equal(permuted, output[:, permutation])


def test_decentralized_policy_is_invariant_to_peer_histories() -> None:
    team_size = 3
    observations, actions = _team_spaces(team_size)
    agent = object.__new__(MARLCore)
    MARLCore.__init__(agent, observations, actions, _agent_config(team_size))
    focal = jax.random.randint(
        jax.random.key(91), (1, 2, 64, 64, 3), 0, 256, dtype=jnp.uint8
    )
    peers = jax.random.randint(
        jax.random.key(92),
        (1, 2, team_size - 1, 64, 64, 3),
        0,
        256,
        dtype=jnp.uint8,
    )

    def observation(images, first):
        return {
            "image": images,
            "reward": jnp.zeros((1, team_size), jnp.float32),
            "agent_present": jnp.ones((1, team_size), bool),
            "agent_alive": jnp.ones((1, team_size), bool),
            "action_mask": jnp.ones((1, team_size, 4), bool),
            "is_first": jnp.full((1,), first, bool),
            "is_last": jnp.zeros((1,), bool),
            "is_terminal": jnp.zeros((1,), bool),
        }

    def rollout(peer_state_intervention):
        carry = agent.init_policy(1)
        first_images = jnp.concatenate([focal[:, :1, None], peers[:, :1]], axis=2)[:, 0]
        second_images = jnp.concatenate([focal[:, 1:, None], peers[:, 1:]], axis=2)[
            :, 0
        ]
        carry, _, _ = agent.policy(carry, observation(first_images, True), mode="eval")
        dyn = carry[1]
        intervened = jnp.roll(dyn["stoch"], 1, axis=-1)
        peer_mask = (jnp.arange(team_size) > 0)[None, :, None, None]
        peer_stoch = jnp.where(peer_mask, intervened, dyn["stoch"])
        dyn = dict(
            dyn,
            stoch=jnp.where(
                peer_state_intervention,
                peer_stoch,
                dyn["stoch"],
            ),
        )
        carry = (carry[0], dyn, carry[2], carry[3])
        second_obs = observation(second_images, False)
        local_carry = agent.team.fold_tree_batch(carry)
        local_obs = agent.team.local_policy_data(second_obs)
        enc_carry, dyn_carry, dec_carry, previous_action = local_carry
        reset = local_obs["is_first"]
        enc_carry, _, tokens = agent.enc(
            enc_carry, local_obs, reset, training=False, single=True
        )
        dyn_carry, _, feat, _ = agent.observe_dynamics(
            dyn_carry,
            tokens,
            previous_action,
            reset,
            local_obs,
            training=False,
            single=True,
        )
        tensor = agent.feat2tensor(feat)
        policy = agent.policy_distribution(
            tensor,
            1,
            action_mask=local_obs["action_mask"],
        )
        logits = agent.team.unfold_batch(policy["action"].logits)[:, 0]
        world = agent.team.unfold_batch(dyn_carry["deter"])[:, 0]
        del enc_carry, dec_carry
        return logits, world

    state = nj.init(rollout)({}, jnp.asarray(False), seed=93)
    _, (baseline_logits, baseline_world) = nj.pure(rollout)(
        state, jnp.asarray(False), seed=94
    )
    _, (changed_logits, changed_world) = nj.pure(rollout)(
        state, jnp.asarray(True), seed=94
    )
    np.testing.assert_array_equal(changed_logits, baseline_logits)
    np.testing.assert_array_equal(changed_world, baseline_world)


def test_inactive_agents_are_excluded_from_loss() -> None:
    values = jnp.array([[1.0, 2.0], [1_000.0, 2_000.0]])
    valid = jnp.array([[True, True], [False, False]])
    assert float(masked_mean(values, valid)) == 1.5


def test_replay_value_mask_aligns_with_source_states() -> None:
    values = jnp.array([[2.0, 100.0]])
    valid = jnp.array([[True, False, True]])
    assert float(masked_mean(values, valid, alignment="replay_value")) == 2.0
    assert float(masked_mean(values, valid, alignment="tail")) == 100.0


def test_masked_return_normalization_ignores_inactive_samples() -> None:
    def update(values, valid):
        norm = Normalize(
            "perc",
            rate=1.0,
            perclo=0.0,
            perchi=100.0,
            debias=False,
            name="norm",
        )
        return norm(values, update=True, mask=valid)

    active = jnp.array([[1.0, 3.0]])
    active_mask = jnp.ones_like(active, bool)
    contaminated = jnp.array([[1.0, 3.0, 10_000.0]])
    contaminated_mask = jnp.array([[True, True, False]])
    reference_state = nj.init(update)({}, active, active_mask, seed=31)
    masked_state = nj.init(update)({}, contaminated, contaminated_mask, seed=31)
    _assert_tree_equal(reference_state, masked_state)


def test_action_mask_assigns_no_probability_to_invalid_actions() -> None:
    distribution = MLPHead(
        {"action": elements.Space(np.int32, (), 0, 4)},
        {"action": "categorical"},
        layers=0,
        units=8,
        unimix=0.01,
        name="masked_policy",
    )
    features = jnp.array([[1.0, -1.0]], jnp.float32)

    def probabilities():
        policy = distribution(features, 1)
        policy = apply_action_mask(
            policy,
            jnp.array([[False, True, False, True]]),
            "action",
        )
        return jax.nn.softmax(policy["action"].logits, axis=-1)

    state = nj.init(probabilities)({}, seed=32)
    probs = nj.pure(probabilities)(state, seed=33)[1]
    np.testing.assert_array_equal(np.asarray(probs)[0, [0, 2]], 0.0)
    np.testing.assert_allclose(np.asarray(probs).sum(), 1.0)


def test_action_mask_loss_scale_honors_canonical_loss_scales() -> None:
    observations, actions = _team_spaces(3)
    config = _agent_config(3).update({"loss_scales.action_mask": 0.25})
    agent = object.__new__(MARLCore)
    MARLCore.__init__(agent, observations, actions, config)
    assert agent.scales["action_mask"] == 0.25


def test_predicted_action_mask_keeps_actor_log_probabilities_finite() -> None:
    distribution = MLPHead(
        {"action": elements.Space(np.int32, (), 0, 4)},
        {"action": "categorical"},
        layers=0,
        units=8,
        unimix=0.01,
        name="imagined_policy",
    )
    features = jnp.array([[1.0, -1.0]], jnp.float32)

    def log_probabilities():
        policy = distribution(features, 1)
        policy = apply_predicted_action_mask(
            policy,
            jnp.array([[-1e30, 1e30, -1e30, 1e30]], jnp.float32),
            "action",
        )
        actions = jnp.arange(4, dtype=jnp.int32)[None]
        return jax.vmap(policy["action"].logp, in_axes=1, out_axes=1)(actions)

    state = nj.init(log_probabilities)({}, seed=34)
    logps = nj.pure(log_probabilities)(state, seed=35)[1]
    assert np.isfinite(np.asarray(logps)).all()
    assert float(np.asarray(logps).min()) > -30.0


def test_singleton_core_matches_local_training_exactly() -> None:
    local_obs_space, local_act_space = _local_spaces(metadata=True)
    team_obs_space, team_act_space = _team_spaces(1)
    config = _agent_config(1)
    local = object.__new__(LocalAgent)
    LocalAgent.__init__(local, local_obs_space, local_act_space, config)
    team = object.__new__(MARLCore)
    MARLCore.__init__(team, team_obs_space, team_act_space, config)

    full_data = _local_batch()
    local_data = {
        key: value
        for key, value in full_data.items()
        if key in local_obs_space or key in local_act_space or key in GLOBAL_KEYS
    }
    team_data = _add_team_axis(full_data, 1)
    local_carry = local.init_train(1)
    team_carry = team.init_train(1)

    def local_step():
        return local.train(local_carry, local_data)

    def team_step():
        return team.train(team_carry, team_data)

    local_state = nj.init(local_step)({}, seed=44)
    team_state = nj.init(team_step)({}, seed=44)
    _assert_tree_equal(_local_model_state(team_state), _local_model_state(local_state))
    local_next, local_output = nj.pure(local_step)(local_state, seed=45)
    team_next, team_output = nj.pure(team_step)(team_state, seed=45)
    _assert_tree_equal(_local_model_state(team_next), _local_model_state(local_next))
    _assert_tree_equal(team.team.fold_tree_batch(team_output[0]), local_output[0])
    _assert_tree_equal(team_output[1], local_output[1])
    local_metrics = {
        key: value
        for key, value in local_output[2].items()
        if not key.startswith("opt/")
    }
    _assert_mapping_subset_equal(team_output[2], local_metrics)


def test_multi_agent_core_completes_shared_local_update() -> None:
    observations, actions = _team_spaces(3)
    agent = object.__new__(MARLCore)
    config = _agent_config(3).update({"opt.warmup": 0})
    MARLCore.__init__(agent, observations, actions, config)
    data = _add_team_axis(_local_batch(), 3)
    carry = agent.init_train(1)

    def step():
        return agent.train(carry, data)

    state = nj.init(step)({}, seed=52)
    _, (_, _, metrics) = nj.pure(step)(state, seed=53)
    assert [module.name for module in agent.modules] == [
        "dyn",
        "enc",
        "rew",
        "con",
        "pol",
        "val",
        "actmask",
    ]
    assert "loss/policy" in metrics


def test_b1_completes_agent_masked_team_jepa_update() -> None:
    observations, actions = _team_spaces(3)
    config = _agent_config(3, "b1").update(
        {
            "opt.warmup": 0,
            "marl.agent_jepa.slots": 2,
            "marl.agent_jepa.width": 32,
            "marl.agent_jepa.heads": 4,
            "marl.agent_jepa.ffup": 2,
            "marl.agent_jepa.predictor_hidden": 32,
            "marl.agent_jepa.utility_probe": True,
        }
    )
    agent = object.__new__(MARLCore)
    MARLCore.__init__(agent, observations, actions, config)
    data = _add_team_axis(_local_batch(), 3)
    carry = agent.init_train(1)

    def step():
        return agent.train(carry, data)

    state = nj.init(step)({}, seed=54)
    next_state, (_, _, metrics) = nj.pure(step)(state, seed=55)
    assert [module.name for module in agent.modules[-7:]] == [
        "team_encoder",
        "team_history_encoder",
        "team_predictor",
        "team_content_predictor",
        "team_action_conditioner",
        "team_transition_encoder",
        "team_transition_predictor",
    ]
    assert np.isfinite(float(metrics["loss/agent_jepa"]))
    assert float(metrics["agent_jepa/eligible_fraction"]) == 1.0
    assert float(metrics["agent_jepa/team_target_std"]) > 0.0
    assert float(metrics["agent_jepa/future_valid_fraction"]) > 0.0
    assert np.isfinite(float(metrics["agent_jepa/future_cosine"]))
    assert any(key.startswith("target_team_encoder/") for key in next_state)
    assert not any(module.name == "target_team_encoder" for module in agent.modules)

    def report_step():
        return agent.report(agent.init_report(1), data)

    _, (_, report_metrics) = nj.pure(report_step)(next_state, seed=56)
    for key in (
        "agent_jepa/probe/future_cross_batch_action_gap",
        "agent_jepa/probe/future_agent_pairing_gap",
        "agent_jepa/probe/future_vs_persistence_gap",
    ):
        assert np.isfinite(float(report_metrics[key]))


def test_b1_masked_agent_target_has_no_hidden_context_leakage() -> None:
    batch, length, agents = 2, 3, 3
    histories = jax.random.normal(jax.random.key(61), (batch, length, agents, 16))
    targets = jax.random.normal(jax.random.key(62), (batch, length, agents, 12))
    hidden = jax.nn.one_hot(jnp.full((batch, length), 2), agents, dtype=bool)
    visible = ~hidden
    eligible = jnp.ones((batch, length), bool)

    active = jnp.ones((batch, length, agents), bool)

    def predict(members, local_histories):
        content = TeamSlotEncoder(
            slots=2,
            width=16,
            heads=4,
            layers=2,
            ffup=2,
            name="content",
        )(members, visible, active)
        context = AgentContextEncoder(
            slots=2,
            width=16,
            heads=4,
            ffup=2,
            name="context",
        )(local_histories, visible)
        return TeamSlotPredictor(
            width=16,
            heads=4,
            layers=2,
            ffup=2,
            name="predictor",
        )(content, context)

    state = nj.init(lambda: predict(targets, histories))({}, seed=63)
    _, baseline = nj.pure(lambda: predict(targets, histories))(state, seed=64)
    hidden_members = targets.at[:, :, 2].add(100.0)
    hidden_histories = histories.at[:, :, 2].add(100.0)
    _, hidden_prediction = nj.pure(
        lambda: predict(hidden_members, hidden_histories)
    )(state, seed=64)
    visible_members = targets.at[:, :, 0].add(100.0)
    _, visible_prediction = nj.pure(lambda: predict(visible_members, histories))(
        state, seed=64
    )
    np.testing.assert_array_equal(hidden_prediction, baseline)
    assert not np.array_equal(np.asarray(visible_prediction), np.asarray(baseline))

    def encode_target(members):
        return TeamSlotEncoder(
            slots=2,
            width=16,
            heads=4,
            layers=2,
            ffup=2,
            name="teacher",
        )(members, active, active)

    teacher_state = nj.init(lambda: encode_target(targets))({}, seed=65)
    _, target_slots = nj.pure(lambda: encode_target(targets))(
        teacher_state, seed=66
    )
    _, changed_slots = nj.pure(lambda: encode_target(hidden_members))(
        teacher_state, seed=66
    )
    assert not np.array_equal(np.asarray(changed_slots), np.asarray(target_slots))
    team_loss, _ = team_slot_jepa_loss(baseline, target_slots, eligible)
    assert np.isfinite(np.asarray(team_loss)).all()

    anchor_state = nj.init(
        lambda: TeamContentPredictor(12, hidden=16, name="anchor")(baseline)
    )({}, seed=67)
    _, content_prediction = nj.pure(
        lambda: TeamContentPredictor(12, hidden=16, name="anchor")(baseline)
    )(anchor_state, seed=68)
    set_loss, metrics = team_set_matching_loss(
        content_prediction,
        targets,
        active,
        eligible,
        temperature=0.1,
        iterations=5,
        name="test_set",
    )
    coverage_loss, coverage_metrics = masked_agent_coverage_loss(
        content_prediction,
        targets,
        active,
        hidden,
        eligible,
        temperature=0.1,
    )
    assert content_prediction.shape == (batch, length, 2, 12)
    assert np.isfinite(np.asarray(set_loss)).all()
    assert np.isfinite(np.asarray(coverage_loss)).all()
    assert float(metrics["agent_jepa/test_set_target_std"]) > 0.1
    assert np.isfinite(float(coverage_metrics["agent_jepa/hidden_coverage_cosine"]))

    gradient = jax.grad(lambda value: scale_gradient(value, 0.1).sum())(histories)
    np.testing.assert_allclose(np.asarray(gradient), 0.1, atol=1e-6)


def test_team_set_matching_breaks_near_rank_one_slots() -> None:
    targets = jax.random.normal(jax.random.key(71), (1, 2, 3, 8))
    base = jax.random.normal(jax.random.key(72), (1, 2, 1, 8))
    noise = 1e-2 * jax.random.normal(jax.random.key(73), (1, 2, 4, 8))
    prediction = base + noise
    active = jnp.ones((1, 2, 3), bool)
    valid = jnp.ones((1, 2), bool)

    def objective(value):
        loss, _ = team_set_matching_loss(
            value,
            targets,
            active,
            valid,
            temperature=0.1,
            iterations=10,
            name="matching_test",
        )
        return loss.mean()

    initial = objective(prediction)
    first_gradient = jax.grad(objective)(prediction)
    assert float(first_gradient.std(axis=-2).mean()) > 1e-4
    for _ in range(100):
        prediction -= jax.grad(objective)(prediction)
    assert float(objective(prediction)) < 0.6 * float(initial)
    assert float(prediction.std(axis=-2).mean()) > 0.1
