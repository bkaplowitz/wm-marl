from __future__ import annotations

import elements
import embodied.jax.outs as outs
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from dreamarl.agent import Agent as LocalAgent
from dreamarl.main import _load_configs, _merge_dicts
from dreamarl.marl.axes import TeamAxis
from dreamarl.marl.core import MARLCore
from dreamarl.marl.spaces import add_agent_axis
from dreamarl.models.heads import (
    MLPHead,
    apply_action_mask,
    apply_predicted_action_mask,
    binary_vector_loss,
)
from dreamarl.models.ctde import (
    JointObservationJEPA,
    TeammateActionBelief,
    TeammateBeliefActorAdapter,
)
from dreamarl.models.normalize import Normalize
from dreamarl.training.learner import masked_mean
from dreamarl.training.optimization import OptimizationMixin


GLOBAL_KEYS = {"is_first", "is_last", "is_terminal", "consec", "stepid"}


def test_actor_optimizer_can_use_an_independent_update_timescale() -> None:
    class Factory(OptimizationMixin):
        pass

    optimizer = Factory()._make_opt(
        lr=1e-3,
        agc=0.0,
        momentum=False,
        warmup=0,
        update_every=2,
    )
    params = {"actor/kernel": jnp.ones((2,), jnp.float32)}
    grads = {"actor/kernel": jnp.ones((2,), jnp.float32)}
    state = optimizer.init(params)
    first, state = optimizer.update(grads, state, params)
    second, _ = optimizer.update(grads, state, params)
    np.testing.assert_array_equal(first["actor/kernel"], jnp.zeros((2,)))
    assert np.linalg.norm(np.asarray(second["actor/kernel"])) > 0


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


def _agent_config(num_agents: int = 1, marl_stage: str = "local"):
    configs = _load_configs()
    resolved = _merge_dicts(configs["defaults"], configs["ctde"])
    if marl_stage == "local":
        resolved = _merge_dicts(resolved, configs["local"])
    resolved = elements.Config(resolved).update(configs["debug"])
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
            controllable_alive=elements.Space(bool, ()),
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
        "controllable_alive": jnp.ones((batch, length), bool),
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

    np.testing.assert_array_equal(mixed.raw_logits, plain.raw_logits)
    assert mixed.unimix == 0.5
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
    MARLCore.__init__(agent, observations, actions, _agent_config(team_size, "ctde"))
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


def _teammate_belief_v2_agent(team_size=3, action_count=4, *, enabled=True):
    local_observations = {
        "vector": elements.Space(np.float32, (3,)),
        "reward": elements.Space(np.float32, ()),
        "agent_present": elements.Space(bool, ()),
        "agent_alive": elements.Space(bool, ()),
        "controllable_alive": elements.Space(bool, ()),
        "action_mask": elements.Space(bool, (action_count,)),
        "is_first": elements.Space(bool, ()),
        "is_last": elements.Space(bool, ()),
        "is_terminal": elements.Space(bool, ()),
    }
    observations = {
        key: (
            space
            if key in {"is_first", "is_last", "is_terminal"}
            else add_agent_axis(space, team_size)
        )
        for key, space in local_observations.items()
    }
    actions = {
        "action": add_agent_axis(
            elements.Space(np.int32, (), 0, action_count), team_size
        )
    }
    config = _agent_config(team_size, "ctde").update(
        {
            "marl.ctde.teammate_belief.enabled": enabled,
            "marl.ctde.teammate_belief.layers": 1,
            "marl.ctde.teammate_belief.units": 8,
            "marl.ctde.teammate_belief.outscale": 1.0,
            "marl.ctde.teammate_belief.adapter_layers": 1,
            "marl.ctde.teammate_belief.adapter_units": 8,
            "policy.layers": 1,
            "policy.units": 8,
            "marl.ctde.critic.width": 8,
            "marl.ctde.critic.heads": 1,
            "marl.ctde.critic.layers": 1,
            "marl.ctde.critic.value_layers": 1,
            "marl.ctde.critic.value_units": 8,
        }
    )
    agent = object.__new__(MARLCore)
    MARLCore.__init__(agent, observations, actions, config)
    return agent


def test_teammate_belief_v2_initial_policy_and_pol_params_equal_control() -> None:
    control = _teammate_belief_v2_agent(enabled=False)
    treatment = _teammate_belief_v2_agent(enabled=True)
    local_state = jax.random.normal(jax.random.key(1001), (3, 6))
    action_mask = jnp.asarray(
        [[True, True, False, True], [True, False, True, True], [True] * 4]
    )

    def initialize(agent):
        def forward():
            policy = agent.policy_distribution(local_state, 1, action_mask)
            return policy["action"].logits

        params = nj.init(forward)({}, seed=1002)
        output = nj.pure(forward)(params, seed=1003)[1]
        return params, output

    control_params, control_logits = initialize(control)
    treatment_params, treatment_logits = initialize(treatment)
    np.testing.assert_array_equal(treatment_logits, control_logits)
    for key, value in control_params.items():
        if key.startswith("pol/"):
            np.testing.assert_array_equal(treatment_params[key], value)
    residual = treatment_params["ctde_teammate_actor/residual/kernel"]
    np.testing.assert_array_equal(residual, jnp.zeros_like(residual))
    assert treatment.policy_keys == (
        "^(enc|dyn|pol|ctde_teammate_belief|ctde_teammate_actor)/"
    )


def test_teammate_belief_v2_creation_preserves_base_rng_and_sampling() -> None:
    control = _teammate_belief_v2_agent(enabled=False)
    treatment = _teammate_belief_v2_agent(enabled=True)
    local_state = jax.random.normal(jax.random.key(1004), (3, 6))
    action_mask = jnp.ones((3, 4), bool)

    def initialize(agent, seed):
        def forward():
            policy = agent.policy_distribution(local_state, 1, action_mask)
            action = policy["action"].sample(nj.seed())
            sentinel = nj.seed()
            return policy["action"].logits, action, sentinel

        return nj.pure(forward)({}, create=True, modify=True, ignore=False, seed=seed)

    control_state, control_output = initialize(control, 1005)
    treatment_state, treatment_output = initialize(treatment, 1005)
    repeated_state, repeated_output = initialize(treatment, 1005)
    different_state, _ = initialize(treatment, 1006)
    _assert_tree_equal(treatment_output, control_output)
    _assert_tree_equal(repeated_output, treatment_output)
    _assert_tree_equal(repeated_state, treatment_state)
    for key, value in control_state.items():
        np.testing.assert_array_equal(treatment_state[key], value, err_msg=key)

    predictor_key = "ctde_teammate_belief/layer0/kernel"
    adapter_key = "ctde_teammate_actor/own_projection/kernel"
    for key in (predictor_key, adapter_key):
        assert float(jnp.linalg.norm(treatment_state[key])) > 0.0
        assert not np.array_equal(
            np.asarray(treatment_state[key]), np.asarray(different_state[key])
        )


def test_teammate_belief_v2_zero_adapter_preserves_categorical_semantics() -> None:
    control = _teammate_belief_v2_agent(enabled=False)
    treatment = _teammate_belief_v2_agent(enabled=True)
    local_state = jax.random.normal(jax.random.key(1004), (3, 2, 6))
    action_mask = jnp.asarray(
        [
            [[True, True, False, True], [True, False, True, True]],
            [[True, True, True, False], [True, True, False, True]],
            [[True, False, True, True], [True, True, True, False]],
        ]
    )
    availability_logits = jnp.linspace(-2.0, 2.0, 4)[None, None]
    availability_logits = jnp.broadcast_to(availability_logits, action_mask.shape)
    exact_rows = jnp.zeros(action_mask.shape[:-1], bool).at[:, 0].set(True)

    def summarize(distribution, seed):
        action = distribution["action"]
        event = jnp.zeros(action.logits.shape[:-1], jnp.int32)
        return {
            "logits": action.logits,
            "probability": jax.nn.softmax(action.logits, axis=-1),
            "logp": action.logp(event),
            "entropy": action.entropy(),
            "pred": action.pred(),
            "sample": action.sample(jax.random.key(seed)),
        }

    def initialize(agent):
        def forward():
            exact = agent.policy_distribution(local_state, 2, action_mask)
            locally_predicted = agent.policy_distribution(local_state, 2, None)
            imagined = agent.imagination_policy_distribution(
                local_state,
                {
                    "action_mask": agent.team.unfold_sequence(action_mask),
                    "present": jnp.ones((1, 2, 3), bool),
                    "controllable_alive": jnp.ones((1, 2, 3), bool),
                },
            )
            calibrated, _ = agent._ctde_probabilistic_policy(
                local_state,
                2,
                jnp.ones(action_mask.shape[:-1], bool),
                availability_logits=availability_logits,
                exact_mask=action_mask,
                exact_rows=exact_rows,
            )
            return {
                "exact": summarize(exact, 1005),
                "locally_predicted": summarize(locally_predicted, 1006),
                "imagined": summarize(imagined, 1007),
                "calibrated": summarize(calibrated, 1008),
            }

        params = nj.init(forward)({}, seed=1009)
        outputs = nj.pure(forward)(params, seed=1010)[1]
        return params, outputs

    control_params, control_outputs = initialize(control)
    treatment_params, treatment_outputs = initialize(treatment)
    _assert_tree_equal(treatment_outputs, control_outputs)
    for key, value in control_params.items():
        if key.startswith("pol/"):
            np.testing.assert_array_equal(treatment_params[key], value)


def test_teammate_belief_v2_nonzero_residual_preserves_unimix_then_masks() -> None:
    agent = _teammate_belief_v2_agent(enabled=True)
    local_state = jax.random.normal(jax.random.key(1011), (3, 6))
    belief_context = jnp.linspace(-1.0, 1.0, 3 * 2 * 4).reshape(3, 2, 4)
    action_mask = jnp.asarray(
        [[True, True, False, True], [True, False, True, True], [True] * 4]
    )
    availability_logits = jnp.asarray(
        [[-2.0, -1.0, 1.0, 2.0], [2.0, 1.0, -1.0, -2.0], [0.0] * 4]
    )

    def forward():
        base = agent.pol(local_state, bdims=1)
        treated, residual = agent._teammate_policy_before_mask(
            local_state, 1, belief_context=belief_context
        )
        exact = apply_action_mask(treated, action_mask, "action")
        predicted = apply_predicted_action_mask(treated, availability_logits, "action")
        return {
            "raw": base["action"].raw_logits,
            "residual": residual,
            "treated": treated["action"].logits,
            "exact": exact["action"].logits,
            "predicted": predicted["action"].logits,
        }

    params = nj.init(forward)({}, seed=1012)
    params = dict(
        params,
        **{
            "ctde_teammate_actor/residual/kernel": (
                jnp.ones_like(params["ctde_teammate_actor/residual/kernel"])
                * jnp.asarray([-10.0, -3.0, 3.0, 10.0])[None]
            )
        },
    )
    output = nj.pure(forward)(params, seed=1013)[1]
    assert float(jnp.abs(output["residual"]).max()) > 0.0

    unimix = float(agent.config.policy.unimix)
    mixed_probability = (1.0 - unimix) * jax.nn.softmax(
        output["raw"] + output["residual"], axis=-1
    ) + unimix / 4
    np.testing.assert_allclose(
        jax.nn.softmax(output["treated"], axis=-1),
        mixed_probability,
        rtol=1e-6,
        atol=1e-7,
    )
    assert float(mixed_probability.min()) >= unimix / 4 - 1e-7

    exact_probability = mixed_probability * action_mask
    exact_probability /= exact_probability.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(
        jax.nn.softmax(output["exact"], axis=-1),
        exact_probability,
        rtol=1e-6,
        atol=1e-7,
    )
    availability = jax.nn.sigmoid(availability_logits)
    predicted_probability = mixed_probability * availability
    predicted_probability /= predicted_probability.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(
        jax.nn.softmax(output["predicted"], axis=-1),
        predicted_probability,
        rtol=1e-6,
        atol=1e-7,
    )


def test_teammate_belief_v2_context_is_bounded_offset_invariant_evidence() -> None:
    agent = _teammate_belief_v2_agent()
    logits = jnp.asarray([[[0.0, 1.0, -1.0, 2.0], [3.0, 3.0, 3.0, 3.0]]])
    context = agent._teammate_belief_context(logits)
    shifted = agent._teammate_belief_context(logits + 1_000.0)
    uniform = agent._teammate_belief_context(jnp.ones_like(logits) * -17.0)
    np.testing.assert_allclose(context, shifted, atol=1e-6)
    np.testing.assert_array_equal(uniform, jnp.zeros_like(uniform))
    assert float(jnp.abs(context).max()) <= 1.0


def test_teammate_actor_uniform_context_remains_zero_after_parameter_change() -> None:
    adapter = TeammateBeliefActorAdapter(
        action_count=5,
        units=8,
        layers=1,
        name="ctde_teammate_actor",
    )
    local_state = jax.random.normal(jax.random.key(1010), (3, 7))
    zero_context = jnp.zeros((3, 10), jnp.float32)

    def forward(context):
        return adapter(local_state, context, bdims=1)

    params = nj.init(forward)({}, zero_context, seed=1011)
    changed = {
        key: (
            jax.random.normal(jax.random.key(1012 + index), value.shape)
            if jnp.issubdtype(value.dtype, jnp.inexact)
            else value
        )
        for index, (key, value) in enumerate(params.items())
    }
    output = nj.pure(forward)(changed, zero_context, seed=1020)[1]
    np.testing.assert_array_equal(output, jnp.zeros_like(output))


def test_teammate_belief_v2_stops_predictor_and_adapter_inputs() -> None:
    predictor = TeammateActionBelief(
        peers=2,
        action_count=4,
        layers=1,
        units=8,
        outscale=1.0,
        name="ctde_teammate_belief",
    )
    adapter = TeammateBeliefActorAdapter(
        action_count=4,
        units=8,
        layers=1,
        name="ctde_teammate_actor",
    )
    local_state = jax.random.normal(jax.random.key(1030), (3, 7))
    context = jax.random.normal(jax.random.key(1031), (3, 8))

    def predictor_output(value):
        return predictor(value, 1)

    predictor_params = nj.init(predictor_output)({}, local_state, seed=1032)
    predictor_grad = jax.grad(
        lambda value: jnp.square(
            nj.pure(predictor_output)(predictor_params, value, seed=1033)[1]
        ).sum()
    )(local_state)
    np.testing.assert_array_equal(predictor_grad, jnp.zeros_like(predictor_grad))

    def adapter_output(state, belief):
        return adapter(state, belief, 1)

    adapter_params = nj.init(adapter_output)({}, local_state, context, seed=1034)
    adapter_params = dict(
        adapter_params,
        **{
            "ctde_teammate_actor/residual/kernel": jnp.ones_like(
                adapter_params["ctde_teammate_actor/residual/kernel"]
            )
        },
    )
    state_grad, context_grad = jax.grad(
        lambda state, belief: jnp.square(
            nj.pure(adapter_output)(adapter_params, state, belief, seed=1035)[1]
        ).sum(),
        argnums=(0, 1),
    )(local_state, context)
    np.testing.assert_array_equal(state_grad, jnp.zeros_like(state_grad))
    np.testing.assert_array_equal(context_grad, jnp.zeros_like(context_grad))


def test_teammate_belief_v2_actor_and_predictor_ownership_are_disjoint() -> None:
    agent = _teammate_belief_v2_agent()
    local_state = jax.random.normal(jax.random.key(1040), (3, 2, 6))
    action_mask = jnp.ones((3, 2, 4), bool)

    def objectives():
        policy = agent.policy_distribution(local_state, 2, action_mask)
        actor = jnp.square(policy["action"].logits).sum()
        belief = agent._teammate_belief_logits(local_state, 2)
        labels = jnp.zeros(belief.shape[:-1], jnp.int32)
        supervised = -jnp.take_along_axis(
            jax.nn.log_softmax(belief, axis=-1),
            labels[..., None],
            axis=-1,
        ).mean()
        return actor, supervised

    params = nj.init(objectives)({}, seed=1041)
    params = dict(
        params,
        **{
            "ctde_teammate_actor/residual/kernel": jnp.ones_like(
                params["ctde_teammate_actor/residual/kernel"]
            )
            * 0.1
        },
    )

    def gradient(index):
        return jax.grad(lambda state: nj.pure(objectives)(state, seed=1042)[1][index])(
            params
        )

    actor_grad = gradient(0)
    supervised_grad = gradient(1)
    predictor_keys = [key for key in params if key.startswith("ctde_teammate_belief/")]
    adapter_keys = [key for key in params if key.startswith("ctde_teammate_actor/")]
    assert predictor_keys and adapter_keys
    for key in predictor_keys:
        np.testing.assert_array_equal(actor_grad[key], jnp.zeros_like(actor_grad[key]))
    for key in adapter_keys:
        np.testing.assert_array_equal(
            supervised_grad[key], jnp.zeros_like(supervised_grad[key])
        )
    assert any(float(jnp.linalg.norm(actor_grad[key])) > 0.0 for key in adapter_keys)
    assert any(
        float(jnp.linalg.norm(supervised_grad[key])) > 0.0 for key in predictor_keys
    )


def test_teammate_belief_v2_execution_is_invariant_to_peer_state_rows() -> None:
    agent = _teammate_belief_v2_agent()
    local_state = jax.random.normal(jax.random.key(1050), (3, 6))
    action_mask = jnp.ones((3, 4), bool)

    def focal_logits(value):
        return agent.policy_distribution(value, 1, action_mask)["action"].logits[0]

    params = nj.init(focal_logits)({}, local_state, seed=1051)
    params = dict(
        params,
        **{
            "ctde_teammate_actor/residual/kernel": jnp.ones_like(
                params["ctde_teammate_actor/residual/kernel"]
            )
            * 0.1
        },
    )
    changed = local_state.at[1:].set(
        jax.random.normal(jax.random.key(1052), local_state[1:].shape) * 100.0
    )
    baseline = nj.pure(focal_logits)(params, local_state, seed=1053)[1]
    intervened = nj.pure(focal_logits)(params, changed, seed=1053)[1]
    np.testing.assert_array_equal(intervened, baseline)


@pytest.mark.parametrize(("team_size", "action_count"), ((5, 9), (8, 14)))
def test_teammate_belief_v2_policy_paths_and_smac_shapes(
    team_size: int,
    action_count: int,
) -> None:
    agent = _teammate_belief_v2_agent(team_size, action_count)
    sequence_length = 16
    local_state = jax.random.normal(
        jax.random.key(1060 + team_size), (team_size, sequence_length, 6)
    )
    action_mask = jnp.ones((team_size, sequence_length, action_count), bool)
    action_mask = action_mask.at[..., -1].set(False)
    valid = jnp.ones((team_size, sequence_length), jnp.float32)

    def forward():
        online = agent.policy_distribution(local_state, 2, action_mask)
        imagined = agent.imagination_policy_distribution(
            local_state,
            {
                "action_mask": agent.team.unfold_sequence(action_mask),
                "present": jnp.ones((1, sequence_length, team_size), bool),
                "controllable_alive": jnp.ones((1, sequence_length, team_size), bool),
            },
        )
        metrics = agent._teammate_belief_policy_metrics(local_state, action_mask, valid)
        probabilistic, _ = agent._ctde_probabilistic_policy(
            local_state,
            2,
            jnp.ones((team_size, sequence_length), bool),
            availability_logits=jnp.zeros_like(action_mask, jnp.float32),
            exact_mask=action_mask,
            exact_rows=jnp.ones((team_size, sequence_length), bool),
        )
        return (
            online["action"].logits,
            imagined["action"].logits,
            probabilistic["action"].logits,
            metrics,
        )

    params = nj.init(forward)({}, seed=1070 + team_size)
    params = dict(
        params,
        **{
            "ctde_teammate_actor/residual/kernel": (
                jnp.ones_like(params["ctde_teammate_actor/residual/kernel"])
                * jnp.arange(action_count, dtype=jnp.float32)[None]
                * 0.1
            )
        },
    )
    online, imagined, probabilistic, metrics = nj.pure(forward)(
        params, seed=1080 + team_size
    )[1]
    np.testing.assert_array_equal(online, imagined)
    np.testing.assert_array_equal(online, probabilistic)
    np.testing.assert_array_equal(
        jax.nn.softmax(online, axis=-1)[..., -1],
        jnp.zeros_like(online[..., -1]),
    )
    assert online.shape == (team_size, sequence_length, action_count)
    for value in metrics.values():
        assert np.isfinite(float(value))
    for horizon in (1, 4, 8, 15):
        for suffix in (
            "valid_fraction",
            "valid_count",
            "logit_rms",
            "context_norm",
            "residual_rms",
            "policy_kl_vs_zero",
            "policy_flip_vs_zero",
            "policy_kl_vs_peer_shuffle",
            "policy_flip_vs_peer_shuffle",
        ):
            assert f"ctde/teammate_belief_h{horizon}_{suffix}" in metrics
    assert float(metrics["ctde/teammate_belief_policy_kl_vs_zero"]) > 0.0


def test_teammate_belief_v2_online_policy_carry_is_finite_and_synced() -> None:
    team_size = 3
    agent = _teammate_belief_v2_agent(team_size, 4)
    carry = agent.init_policy(1)
    observation = {
        "vector": jax.random.normal(jax.random.key(1090), (1, team_size, 3)),
        "reward": jnp.zeros((1, team_size), jnp.float32),
        "agent_present": jnp.ones((1, team_size), bool),
        "agent_alive": jnp.ones((1, team_size), bool),
        "controllable_alive": jnp.ones((1, team_size), bool),
        "action_mask": jnp.ones((1, team_size, 4), bool),
        "is_first": jnp.ones((1,), bool),
        "is_last": jnp.zeros((1,), bool),
        "is_terminal": jnp.zeros((1,), bool),
    }

    def step():
        return agent.policy(carry, observation, mode="eval")

    params = nj.init(step)({}, seed=1091)
    _, action, output = nj.pure(step)(params, seed=1092)[1]
    assert action["action"].shape == (1, team_size)
    assert all(np.asarray(value).all() for value in output["finite"].values())
    assert any(key.startswith("ctde_teammate_belief/") for key in params)
    assert any(key.startswith("ctde_teammate_actor/") for key in params)


def test_teammate_belief_v2_replay_alignment_and_conditional_masks() -> None:
    agent = _teammate_belief_v2_agent(team_size=3, action_count=7)
    source_state = jax.random.normal(jax.random.key(1100), (1, 2, 3, 6))
    current_action = jnp.asarray([[[0, 1, 6], [3, 2, 1]]], jnp.int32)
    previous_action = jnp.full_like(current_action, 4)
    present = jnp.ones((1, 2, 3), bool)
    alive = jnp.asarray([[[True, True, False], [True, True, True]]])
    action_mask = jnp.ones((1, 2, 3, 7), bool)
    action_mask = action_mask.at[0, 0, 2].set(
        jnp.asarray([True, False, False, False, False, False, True])
    )
    current_first = jnp.asarray([[True, False]])
    next_first = jnp.asarray([[False, True]])

    def loss_and_metrics(mask, roster):
        return agent._ctde_teammate_belief_loss(
            source_state,
            current_action,
            previous_action,
            roster,
            alive,
            mask,
            current_first,
            next_first,
        )

    params = nj.init(loss_and_metrics)({}, action_mask, present, seed=1101)
    _, metrics = nj.pure(loss_and_metrics)(params, action_mask, present, seed=1102)[1]
    assert float(metrics["ctde/teammate_belief_target_count"]) == 4.0
    assert float(metrics["ctde/teammate_belief_active_peer_count"]) == 2.0
    assert float(metrics["ctde/teammate_belief_nonnoop_count"]) == 3.0
    assert float(metrics["ctde/teammate_belief_attack_count"]) == 2.0
    assert float(metrics["ctde/teammate_belief_dead_peer_fraction"]) == pytest.approx(
        0.5
    )

    illegal = action_mask.at[0, 0, 2, 6].set(False)
    _, illegal_metrics = nj.pure(loss_and_metrics)(params, illegal, present, seed=1102)[
        1
    ]
    assert float(illegal_metrics["ctde/teammate_belief_target_count"]) == 2.0
    assert float(illegal_metrics["ctde/teammate_belief_attack_count"]) == 0.0

    absent = present.at[0, 0, 2].set(False)
    _, absent_metrics = nj.pure(loss_and_metrics)(
        params, action_mask, absent, seed=1102
    )[1]
    assert float(absent_metrics["ctde/teammate_belief_target_count"]) == 2.0
    assert float(absent_metrics["ctde/teammate_belief_dead_peer_fraction"]) == 0.0


def test_teammate_belief_v2_labels_use_a_t_not_previous_action() -> None:
    agent = _teammate_belief_v2_agent(team_size=3, action_count=5)
    source_state = jnp.ones((1, 1, 3, 6), jnp.float32)
    current_action = jnp.asarray([[[0, 1, 2]]], jnp.int32)
    previous_action = jnp.full_like(current_action, 4)
    present = jnp.ones((1, 1, 3), bool)
    alive = jnp.ones((1, 1, 3), bool)
    action_mask = jnp.ones((1, 1, 3, 5), bool)
    first = jnp.zeros((1, 1), bool)

    def loss_and_metrics():
        return agent._ctde_teammate_belief_loss(
            source_state,
            current_action,
            previous_action,
            present,
            alive,
            action_mask,
            first,
            first,
        )

    params = nj.init(loss_and_metrics)({}, seed=1110)
    bias_key = "ctde_teammate_belief/logits/bias"
    kernel_key = "ctde_teammate_belief/logits/kernel"
    bias = jnp.arange(10, dtype=jnp.float32) * 0.2
    params = dict(
        params,
        **{
            bias_key: bias,
            kernel_key: jnp.zeros_like(params[kernel_key]),
        },
    )
    _, metrics = nj.pure(loss_and_metrics)(params, seed=1111)[1]
    peer_action = jnp.take(current_action, agent._teammate_peer_indices(), axis=2)
    log_probability = jax.nn.log_softmax(bias.reshape(2, 5), axis=-1)
    expected = -jnp.take_along_axis(
        jnp.broadcast_to(log_probability, (*peer_action.shape, 5)),
        peer_action[..., None],
        axis=-1,
    ).mean()
    previous_expected = -log_probability[:, 4].mean()
    actual = metrics["ctde/teammate_belief_nll"]
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
    assert not np.isclose(float(actual), float(previous_expected))


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


def test_mean_binary_mask_loss_is_invariant_to_action_count() -> None:
    short = outs.Agg(outs.Binary(jnp.zeros((2, 4))), 1, jnp.sum)
    long = outs.Agg(outs.Binary(jnp.zeros((2, 8))), 1, jnp.sum)
    short_target = jnp.zeros((2, 4), bool)
    long_target = jnp.zeros((2, 8), bool)

    short_mean = binary_vector_loss(short, short_target, "mean")
    long_mean = binary_vector_loss(long, long_target, "mean")
    short_sum = binary_vector_loss(short, short_target, "sum")
    long_sum = binary_vector_loss(long, long_target, "sum")

    np.testing.assert_allclose(short_mean, long_mean)
    np.testing.assert_allclose(long_sum, 2.0 * short_sum)


def test_balanced_binary_mask_loss_equalizes_classes_and_action_count() -> None:
    short_logits = jnp.array([[2.0, -1.0, -2.0, 1.0]], jnp.float32)
    short_target = jnp.array([[1, 1, 0, 0]], bool)
    long_logits = jnp.concatenate([short_logits, short_logits], axis=-1)
    long_target = jnp.concatenate([short_target, short_target], axis=-1)
    short = outs.Agg(outs.Binary(short_logits), 1, jnp.sum)
    long = outs.Agg(outs.Binary(long_logits), 1, jnp.sum)

    short_loss = binary_vector_loss(short, short_target, "balanced")
    long_loss = binary_vector_loss(long, long_target, "balanced")
    per_event = outs.Binary(short_logits).loss(short_target)
    expected = 0.5 * (per_event[0, :2].mean() + per_event[0, 2:].mean())

    np.testing.assert_allclose(short_loss[0], expected)
    np.testing.assert_allclose(long_loss, short_loss)


def test_balanced_binary_mask_loss_handles_single_class_rows() -> None:
    logits = jnp.zeros((2, 4), jnp.float32)
    output = outs.Agg(outs.Binary(logits), 1, jnp.sum)
    target = jnp.array([[1, 1, 1, 1], [0, 0, 0, 0]], bool)
    loss = binary_vector_loss(output, target, "balanced")

    np.testing.assert_allclose(loss, jnp.log(2.0))


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


@pytest.mark.parametrize("action_conditioning", ("add", "adaln"))
def test_ctde_joint_replay_and_recurrent_history_match(
    action_conditioning: str,
) -> None:
    model = JointObservationJEPA(
        action_count=4,
        action_low=0,
        target_dim=6,
        width=8,
        heads=2,
        agent_layers=1,
        temporal_layers=1,
        context=4,
        ffup=2,
        dropout=0.0,
        action_conditioning=action_conditioning,
        name="ctde_joint",
    )
    states = jax.random.normal(jax.random.key(691), (2, 7, 3, 5))
    actions = jax.random.randint(jax.random.key(692), (2, 7, 3), 0, 4)
    active = jnp.ones((2, 7, 3), bool)
    reset = jnp.zeros((2, 7), bool).at[:, 0].set(True).at[:, 5].set(True)

    def parallel():
        carry = model.initial(2, 3)
        carry, output, _ = model.sequence(
            carry, states, actions, active, active, reset, training=False
        )
        return carry, output

    def recurrent():
        carry = model.initial(2, 3)

        def transition(current, inputs):
            state, action, current_active, current_reset = inputs
            return model.step(
                current,
                state,
                action,
                current_active,
                current_active,
                current_reset,
                training=False,
            )

        return nj.scan(
            transition,
            carry,
            (states, actions, active, reset),
            axis=1,
        )

    params = nj.init(parallel)({}, seed=693)
    _, parallel_output = nj.pure(parallel)(params, seed=694)
    _, recurrent_output = nj.pure(recurrent)(params, seed=694)
    parallel_leaves, parallel_tree = jax.tree.flatten(parallel_output)
    recurrent_leaves, recurrent_tree = jax.tree.flatten(recurrent_output)
    assert recurrent_tree == parallel_tree
    for recurrent_value, parallel_value in zip(recurrent_leaves, parallel_leaves):
        if jnp.issubdtype(recurrent_value.dtype, jnp.inexact):
            np.testing.assert_allclose(
                np.asarray(recurrent_value, np.float32),
                np.asarray(parallel_value, np.float32),
                atol=6e-2,
                rtol=3e-2,
            )
        else:
            np.testing.assert_array_equal(recurrent_value, parallel_value)


def test_ctde_trains_joint_imagination_and_central_critic() -> None:
    observations, actions = _team_spaces(3)
    config = _agent_config(3, "ctde").update(
        {
            "opt.warmup": 0,
            "marl.ctde.opt.warmup": 0,
        }
    )
    agent = object.__new__(MARLCore)
    MARLCore.__init__(agent, observations, actions, config)
    data = _add_team_axis(_local_batch(length=5), 3)
    carry = agent.init_train(1)

    def step():
        return agent.train(carry, data)

    state = nj.init(step)({}, seed=701)
    _, (_, _, metrics) = nj.pure(step)(state, seed=702)
    assert [module.name for module in agent.ctde_modules] == [
        "ctde_joint",
        "ctde_rew",
        "ctde_con",
        "ctde_mask",
        "ctde_alive",
    ]
    assert agent.policy_keys == "^(enc|dyn|pol)/"
    for key in (
        "loss/ctde_embedding",
        "loss/ctde_interface",
        "loss/ctde_reward",
        "ctde/embedding_cosine",
        "ctde/action_mask_positive_recall",
        "ctde/action_mask_negative_specificity",
        "ctde/attack_mask_positive_recall",
        "ctde/attack_mask_target_rate",
        "ctde/attack_mask_prediction_rate",
        "opt/local_world/grad_norm",
        "opt/joint_world/grad_norm",
        "opt/actor/grad_norm",
        "opt/critic/grad_norm",
    ):
        assert np.isfinite(float(metrics[key]))
    assert not any("multistep" in key for key in metrics)


def test_ctde_soft_mask_and_liveness_update_is_finite() -> None:
    observations, actions = _team_spaces(3)
    config = _agent_config(3, "ctde").update(
        {
            "opt.warmup": 0,
            "marl.ctde.opt.warmup": 0,
            "action_mask_reduction": "mean",
            "marl.ctde.mask_calibration.enabled": True,
            "marl.ctde.mask_calibration.horizons": [1, 2],
            "marl.ctde.mask_calibration.soft_liveness": True,
        }
    )
    agent = object.__new__(MARLCore)
    MARLCore.__init__(agent, observations, actions, config)
    data = _add_team_axis(_local_batch(length=5), 3)
    carry = agent.init_train(1)

    def step():
        return agent.train(carry, data)

    state = nj.init(step)({}, seed=711)
    _, (_, _, metrics) = nj.pure(step)(state, seed=712)
    for key in (
        "loss/action_mask",
        "loss/ctde_action_mask",
        "loss/ctde_mask_calibration",
        "loss/ctde_alive_calibration",
        "ctde/mask_calibration_h2_brier",
        "ctde/alive_calibration_h2_brier",
        "opt/local_world/grad_norm",
        "opt/joint_world/grad_norm",
        "opt/actor/grad_norm",
        "opt/critic/grad_norm",
    ):
        assert np.isfinite(float(metrics[key]))


def test_ctde_two_step_self_fed_objective_is_finite() -> None:
    observations, actions = _team_spaces(3)
    config = _agent_config(3, "ctde").update(
        {
            "opt.warmup": 0,
            "marl.ctde.opt.warmup": 0,
            "marl.ctde.rollout_steps": 2,
            "marl.ctde.multistep.anchors": 2,
        }
    )
    agent = object.__new__(MARLCore)
    MARLCore.__init__(agent, observations, actions, config)
    data = _add_team_axis(_local_batch(length=5), 3)
    carry = agent.init_train(1)

    def step():
        return agent.train(carry, data)

    state = nj.init(step)({}, seed=711)
    _, (_, _, metrics) = nj.pure(step)(state, seed=712)
    for key in (
        "loss/ctde_multistep_embedding",
        "loss/ctde_multistep_interface",
        "loss/ctde_multistep_reward",
        "loss/ctde_multistep_continuation",
        "loss/ctde_multistep_action_mask",
        "loss/ctde_multistep_alive",
        "ctde/multistep_embedding_cosine",
        "ctde/multistep_posterior_kl",
    ):
        assert np.isfinite(float(metrics[key]))
    assert float(metrics["ctde/multistep_anchors"]) == 2.0
    assert "loss/ctde_multistep_posterior" not in metrics
