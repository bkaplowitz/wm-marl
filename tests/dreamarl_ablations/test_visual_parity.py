from __future__ import annotations

from pathlib import Path

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest
import ruamel.yaml as yaml

from dreamerv3.agent import Agent as ReferenceAgent
from dreamerv3 import rssm as reference
from dreamarl.ablations.algorithm import AblationAlgorithm
from dreamarl.models import visual as first_party
from dreamarl.ablations import visual as ablation_visual
from dreamarl.runtime import algorithm_root, repository_root
from dreamarl.marl.spaces import report_rows
from dreamarl.main import _load_configs, _worker_seed
from dreamarl.marl.axes import TeamAxis
from dreamarl.models.latent import CategoricalLatent
from dreamarl.ablations.rssm import GRURSSMDynamics


OBS_SPACE = {"image": elements.Space(np.uint8, (64, 64, 3))}
VJEPA_OBS_SPACE = {"image": elements.Space(np.uint8, (256, 256, 3))}
ACTION_SPACE = {"action": elements.Space(np.int32, (), 0, 4)}


def _encoder(module, name):
    return module.Encoder(
        OBS_SPACE,
        depth=2,
        mults=(1, 1),
        kernel=3,
        act="silu",
        norm="rms",
        name=name,
    )


def test_visual_encoder_matches_pinned_reference() -> None:
    official = _encoder(reference, "enc")
    port = _encoder(first_party, "enc")
    batch, length = 2, 3
    images = jax.random.randint(
        jax.random.key(1), (batch, length, 64, 64, 3), 0, 256, dtype=jnp.uint8
    )
    observations = {"image": images}
    resets = jnp.zeros((batch, length), bool).at[:, 0].set(True)

    def official_encode(obs, first):
        return official({}, obs, first, training=False)

    def port_encode(obs, first):
        return port({}, obs, first, training=False)

    params = nj.init(official_encode)({}, observations, resets, seed=10)
    reference_output = nj.pure(official_encode)(params, observations, resets, seed=11)[
        1
    ]
    port_output = nj.pure(port_encode)(params, observations, resets, seed=11)[1]
    for expected, actual in zip(
        jax.tree.leaves(reference_output), jax.tree.leaves(port_output)
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_compact_vit_preserves_the_cnn_downstream_interface() -> None:
    encoder = ablation_visual.ViTEncoder(OBS_SPACE, name="enc")
    images = jax.random.randint(
        jax.random.key(50), (1, 1, 64, 64, 3), 0, 256, dtype=jnp.uint8
    )
    resets = jnp.ones((1, 1), bool)

    def encode(obs, first):
        return encoder({}, obs, first, training=False)

    params = nj.init(encode)({}, {"image": images}, resets, seed=51)
    _, output = nj.pure(encode)(params, {"image": images}, resets, seed=52)
    tokens = output[2]
    parameter_count = sum(value.size for value in jax.tree.leaves(params))

    assert encoder.calculate_encoder_output_dim() == 4096
    assert encoder.image_grid_shape() == (8, 8, 64)
    assert tokens.shape == (1, 1, 4096)
    assert encoder.spatial_tokens(tokens).shape == (1, 1, 64, 64)
    assert 4_700_000 <= parameter_count <= 5_100_000


def test_vjepa21_encoder_uses_rope_hierarchy_and_a_16_by_16_grid() -> None:
    encoder = ablation_visual.ViTEncoder(
        VJEPA_OBS_SPACE,
        patch=16,
        model=48,
        layers=4,
        heads=3,
        token_dim=64,
        norm="layer1em6",
        act="gelu",
        position="rope3d",
        hierarchical=(0, 1, 2, 3),
        name="enc",
    )
    images = jax.random.randint(
        jax.random.key(53), (1, 1, 256, 256, 3), 0, 256, dtype=jnp.uint8
    )
    resets = jnp.ones((1, 1), bool)
    visible = jnp.ones((1, 1, 16, 16), bool).at[..., 4:9, 5:10].set(False)

    def encode(obs, first, context):
        output = encoder({}, obs, first, training=False)[2]
        visible_tokens = encoder.visible_spatial_tokens(obs, first, context)
        return output, visible_tokens

    params = nj.init(encode)({}, {"image": images}, resets, visible, seed=54)
    _, output = nj.pure(encode)(params, {"image": images}, resets, visible, seed=55)
    full_tokens, visible_tokens = output

    assert encoder.image_grid_shape() == (16, 16, 64)
    assert encoder.predictor_token_dim == 4 * 48
    assert encoder.calculate_encoder_output_dim() == 256 * 64
    assert full_tokens.shape == (1, 1, 256 * 64)
    assert visible_tokens.shape == (1, 1, 256, 4 * 48)
    hidden = np.asarray(visible_tokens).reshape((1, 1, 16, 16, 4 * 48))
    assert (hidden[..., 4:9, 5:10, :] == 0).all()


def test_vjepa_predictor_uses_positional_target_queries() -> None:
    predictor = ablation_visual.SpatialTokenPredictor(
        grid=(16, 16),
        input_dim=8,
        model=24,
        layers=1,
        heads=4,
        name="predictor",
    )
    context = jax.random.normal(jax.random.key(56), (1, 2, 256, 8))
    target = jnp.zeros((1, 2, 16, 16), bool).at[..., 3:8, 4:9].set(True)
    visible = ~target

    def predict(tokens, context_mask, target_mask):
        return predictor(tokens, context_mask, target_mask)

    params = nj.init(predict)({}, context, visible, target, seed=57)
    _, (prediction, context_prediction) = nj.pure(predict)(
        params, context, visible, target, seed=58
    )
    assert prediction.shape == context.shape
    assert context_prediction.shape == context.shape
    output = np.asarray(prediction).reshape((1, 2, 16, 16, 8))
    assert np.isfinite(output).all()
    assert (output[..., :3, :, :] == 0).all()
    assert np.linalg.norm(output[..., 3:8, 4:9, :]) > 0


def test_leworldmodel_vit_uses_unmasked_cls_representation() -> None:
    space = {"image": elements.Space(np.uint8, (224, 224, 3))}
    encoder = ablation_visual.ViTEncoder(
        space,
        patch=14,
        model=48,
        layers=2,
        heads=3,
        token_dim=48,
        norm="layer1em6",
        act="gelu",
        pool="cls",
        position="learned",
        name="enc",
    )
    images = jax.random.randint(
        jax.random.key(59), (1, 2, 224, 224, 3), 0, 256, dtype=jnp.uint8
    )
    resets = jnp.zeros((1, 2), bool)

    def encode(obs, first):
        return encoder({}, obs, first, training=False)[2]

    params = nj.init(encode)({}, {"image": images}, resets, seed=60)
    _, output = nj.pure(encode)(params, {"image": images}, resets, seed=61)
    assert output.shape == (1, 2, 48)
    assert encoder.calculate_encoder_output_dim() == 48
    assert "enc/cls_token" in params
    assert "enc/position" in params
    assert not any("mask_token" in key for key in params)


def test_visual_decoder_matches_pinned_reference() -> None:
    kwargs = {
        "depth": 2,
        "mults": (1, 1),
        "kernel": 3,
        "bspace": 2,
        "act": "silu",
        "norm": "rms",
    }
    official = reference.Decoder(OBS_SPACE, name="dec", **kwargs)
    port = ablation_visual.Decoder(OBS_SPACE, name="dec", **kwargs)
    batch, length = 2, 3
    features = {
        "deter": jax.random.normal(jax.random.key(40), (batch, length, 16)),
        "stoch": jax.random.normal(jax.random.key(41), (batch, length, 2, 4)),
    }
    resets = jnp.zeros((batch, length), bool).at[:, 0].set(True)

    def official_decode(feat, first):
        return official({}, feat, first, training=False)[2]["image"].pred()

    def port_decode(feat, first):
        return port({}, feat, first, training=False)[2]["image"].pred()

    params = nj.init(official_decode)({}, features, resets, seed=42)
    expected = nj.pure(official_decode)(params, features, resets, seed=43)[1]
    actual = nj.pure(port_decode)(params, features, resets, seed=43)[1]
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_categorical_prior_matches_pinned_reference() -> None:
    kwargs = {
        "deter": 16,
        "hidden": 8,
        "stoch": 2,
        "classes": 4,
        "imglayers": 2,
        "act": "silu",
        "norm": "rms",
    }
    official = reference.RSSM(
        ACTION_SPACE,
        blocks=2,
        name="dyn",
        **kwargs,
    )
    port = CategoricalLatent(
        ACTION_SPACE,
        enc_output=12,
        name="dyn",
        **kwargs,
    )
    feature = jax.random.normal(jax.random.key(2), (3, 16), dtype=jnp.bfloat16)

    def official_outputs(value):
        return official._prior(value)

    def port_outputs(value):
        return port._prior(value)

    params = nj.init(official_outputs)({}, feature, seed=20)
    expected = nj.pure(official_outputs)(params, feature, seed=21)[1]
    actual = nj.pure(port_outputs)(params, feature, seed=21)[1]
    for left, right in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
        np.testing.assert_array_equal(np.asarray(right), np.asarray(left))

    def scalar(function, variables):
        outputs = nj.pure(function)(variables, feature, seed=22)[1]
        return sum(value.mean() for value in outputs)

    expected_grad = jax.grad(lambda variables: scalar(official_outputs, variables))(
        params
    )
    actual_grad = jax.grad(lambda variables: scalar(port_outputs, variables))(params)
    for left, right in zip(
        jax.tree.leaves(expected_grad), jax.tree.leaves(actual_grad)
    ):
        np.testing.assert_array_equal(np.asarray(right), np.asarray(left))


def test_first_party_gru_rssm_matches_pinned_reference() -> None:
    kwargs = {
        "deter": 16,
        "hidden": 8,
        "stoch": 2,
        "classes": 4,
        "imglayers": 2,
        "obslayers": 1,
        "dynlayers": 1,
        "blocks": 2,
        "act": "silu",
        "norm": "rms",
    }
    official = reference.RSSM(
        ACTION_SPACE,
        name="dyn",
        **kwargs,
    )
    port = GRURSSMDynamics(
        ACTION_SPACE,
        enc_output=12,
        name="dyn",
        **kwargs,
    )
    batch, length = 2, 5
    tokens = jax.random.normal(jax.random.key(30), (batch, length, 12))
    actions = {"action": jnp.arange(batch * length).reshape(batch, length) % 4}
    resets = jnp.zeros((batch, length), bool).at[:, 0].set(True)

    def official_observe(current_tokens):
        carry, _, feat = official.observe(
            official.initial(batch),
            current_tokens,
            actions,
            resets,
            training=False,
        )
        return carry, feat

    def port_observe(current_tokens):
        carry, _, feat, _ = port.observe(
            port.initial(batch),
            current_tokens,
            actions,
            resets,
            training=False,
        )
        return carry, feat

    params = nj.init(official_observe)({}, tokens, seed=31)
    expected = nj.pure(official_observe)(params, tokens, seed=32)[1]
    actual = nj.pure(port_observe)(params, tokens, seed=32)[1]
    for left, right in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
        np.testing.assert_array_equal(np.asarray(right), np.asarray(left))


def test_agent_axis_is_lossless_and_does_not_mix_agents() -> None:
    batch, length, agents, width = 2, 4, 3, 5
    value = jnp.arange(batch * length * agents * width).reshape(
        batch, length, agents, width
    )
    axis = TeamAxis(agents)
    folded = axis.fold_sequence(value)
    restored = axis.unfold_sequence(folded)
    assert folded.shape == (batch * agents, length, width)
    np.testing.assert_array_equal(np.asarray(restored), np.asarray(value))


def test_report_rows_preserve_complete_teams() -> None:
    assert report_rows(32, 2) == 6
    assert report_rows(48, 3) == 6
    assert report_rows(80, 5) == 5
    assert report_rows(112, 7) == 7


def test_report_rows_reject_incomplete_teams() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        report_rows(16, 5)


def test_worker_seed_is_stable_and_worker_specific() -> None:
    assert _worker_seed(7, 11) == _worker_seed(7, 11)
    assert _worker_seed(7, 11) != _worker_seed(7, 12)
    assert _worker_seed(7, 11) != _worker_seed(8, 11)


def _resolved_config(path: Path, *names: str) -> elements.Config:
    configs = yaml.YAML(typ="safe").load(path.read_text(encoding="utf-8"))
    config = elements.Config(configs["defaults"])
    for name in names:
        config = config.update(configs[name])
    return config.update({"agent.imag_length": 3})


def _resolved_ablation_config(*names: str) -> elements.Config:
    configs = _load_configs(algorithm_root() / "ablations" / "configs.yaml")
    config = elements.Config(configs["defaults"])
    for name in names:
        config = config.update(configs[name])
    return config.update({"agent.imag_length": 3})


def _agent_config(config: elements.Config) -> elements.Config:
    return elements.Config(
        **config.agent,
        logdir="/tmp/dreamarl-full-parity",
        seed=0,
        jax=config.jax,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=0,
        replicas=1,
    )


def test_singleton_reconstruction_loss_and_gradients_match_dreamerv3() -> None:
    reference_config = _resolved_config(
        repository_root() / "external" / "dreamerv3" / "dreamerv3" / "configs.yaml",
        "debug",
    )
    first_party_config = _resolved_ablation_config(
        "dreamerv3_control",
        "debug",
    )

    local_obs_space = {
        "image": elements.Space(np.uint8, (64, 64, 3), 0, 256),
        "reward": elements.Space(np.float32, ()),
        "is_first": elements.Space(bool, ()),
        "is_last": elements.Space(bool, ()),
        "is_terminal": elements.Space(bool, ()),
    }
    local_act_space = {
        "action": elements.Space(np.float32, (2,), -1.0, 1.0),
    }
    joint_obs_space = {
        key: (
            space
            if key in {"is_first", "is_last", "is_terminal"}
            else elements.Space(
                space.dtype,
                (1, *space.shape),
                np.expand_dims(space.low, 0),
                np.expand_dims(space.high, 0),
            )
        )
        for key, space in local_obs_space.items()
    }
    joint_act_space = {
        key: elements.Space(
            space.dtype,
            (1, *space.shape),
            np.expand_dims(space.low, 0),
            np.expand_dims(space.high, 0),
        )
        for key, space in local_act_space.items()
    }

    official = object.__new__(ReferenceAgent)
    ReferenceAgent.__init__(
        official,
        local_obs_space,
        local_act_space,
        _agent_config(reference_config),
    )
    port = object.__new__(AblationAlgorithm)
    AblationAlgorithm.__init__(
        port,
        joint_obs_space,
        joint_act_space,
        _agent_config(first_party_config),
    )

    batch, length = 1, 4
    observations = {
        "image": jax.random.randint(
            jax.random.key(101),
            (batch, length, 64, 64, 3),
            0,
            256,
            dtype=jnp.uint8,
        ),
        "reward": jax.random.uniform(jax.random.key(102), (batch, length)),
        "is_first": jnp.zeros((batch, length), bool).at[:, 0].set(True),
        "is_last": jnp.zeros((batch, length), bool),
        "is_terminal": jnp.zeros((batch, length), bool),
    }
    previous_actions = {
        "action": jax.random.uniform(
            jax.random.key(103), (batch, length, 2), minval=-1.0, maxval=1.0
        )
    }
    official_carry = official.init_policy(batch)[:3]
    port_carry = port._init_local(batch)[:3]

    def official_loss():
        return official.loss(
            official_carry, observations, previous_actions, training=True
        )[0]

    def port_loss():
        return port.loss(port_carry, observations, previous_actions, training=True)[0]

    variables = nj.init(official_loss)({}, seed=104)
    port_variables = nj.init(port_loss)({}, seed=104)
    assert variables.keys() == port_variables.keys()
    for expected, actual in zip(
        jax.tree.leaves(variables), jax.tree.leaves(port_variables)
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def evaluate(function, state):
        return nj.pure(function)(state, seed=105)[1]

    expected_loss = evaluate(official_loss, variables)
    actual_loss = evaluate(port_loss, variables)
    np.testing.assert_array_equal(np.asarray(actual_loss), np.asarray(expected_loss))

    expected_gradients = jax.grad(
        lambda state: evaluate(official_loss, state), allow_int=True
    )(variables)
    actual_gradients = jax.grad(
        lambda state: evaluate(port_loss, state), allow_int=True
    )(variables)
    for expected, actual in zip(
        jax.tree.leaves(expected_gradients), jax.tree.leaves(actual_gradients)
    ):
        if getattr(expected.dtype, "kind", None) == "V":
            continue
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
