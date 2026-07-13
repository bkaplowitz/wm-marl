from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.core import unfreeze

import world_marl.dreamer_v3_baseline as dreamer_v3
from world_marl.dreamer_v3_baseline.config import (
    DecoderConfig,
    DreamerProfile,
    EncoderConfig,
    HeadConfig,
    ObservationMode,
    PolicyConfig,
    RSSMConfig,
)
from world_marl.dreamer_v3_baseline.network_oracle import NETWORKS_SOURCE_SPEC
from world_marl.dreamer_v3_baseline.networks import (
    BlockGRU,
    BlockLinear,
    Conv2D,
    DictDecoder,
    DictEncoder,
    Initializer,
    Linear,
    MLP,
    MLPHead,
    RMSNorm,
    TensorSpace,
)
from world_marl.dreamer_v3_baseline.oracle import (
    OracleHarness,
    OracleManifest,
    ParameterTranslator,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "dreamer_v3"
OFFICIAL_CHECKOUT = Path(
    os.environ.get(
        "DREAMERV3_ORACLE_CHECKOUT",
        "/private/tmp/danijar-dreamerv3-20260713",
    )
)
SOURCE_HASHES = {
    "dreamerv3/rssm.py": (
        "d6d50166914e94fb8bd17a5d5dbda9d42cdd37b85819bb1e9fff3a64d4ad2eb6"
    ),
    "embodied/jax/heads.py": (
        "437641cde21e7f9e3f69b88ad8f6b7e7c22e54eec8c5b19eef6127afde1a9b3f"
    ),
    "embodied/jax/nets.py": (
        "9a1c0c71ad7d3596572a44416e78434f777d8f4dbcbe8ca0dd6b86bb8246392c"
    ),
}


def _assert_f32(actual, expected) -> None:
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def _flat_params(variables) -> dict[str, np.ndarray]:
    flat: dict[str, np.ndarray] = {}

    def visit(prefix: tuple[str, ...], value) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit((*prefix, key), child)
        else:
            flat["/".join(prefix)] = np.asarray(value)

    visit((), unfreeze(variables["params"]))
    return flat


def _nested_params(flat: dict[str, np.ndarray]) -> dict:
    root: dict = {}
    for path, value in flat.items():
        cursor = root
        parts = path.split("/")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = jnp.asarray(value)
    return {"params": root}


def _translate_case(
    arrays: dict[str, np.ndarray],
    prefix: str,
    destination_variables,
) -> dict:
    marker = f"{prefix}.param."
    source = {
        name.removeprefix(marker).replace("__", "/"): value
        for name, value in arrays.items()
        if name.startswith(marker)
    }
    destinations = _flat_params(destination_variables)
    translator = ParameterTranslator()
    assert source
    assert destinations
    for path in sorted(source):
        translator.register(path, path)
    translated = translator.translate(source, destinations)
    translator.assert_fully_consumed()
    return _nested_params(translated)


@pytest.fixture(
    params=(DreamerProfile.PAPER, DreamerProfile.UPSTREAM_CURRENT),
    ids=lambda profile: profile.value,
)
def official_case(request) -> tuple[OracleManifest, dict[str, np.ndarray]]:
    profile = request.param
    stem = f"{profile.value}-vision-networks"
    fixture_path = FIXTURE_DIR / f"{stem}.npz"
    manifest_path = FIXTURE_DIR / f"{stem}.manifest.json"
    manifest = OracleManifest.load(manifest_path, fixture_path=fixture_path)
    with np.load(fixture_path, allow_pickle=False) as fixture:
        arrays = {name: fixture[name] for name in fixture.files}
    return manifest, arrays


def test_network_oracles_pin_exact_three_file_authority(official_case) -> None:
    manifest, _ = official_case

    assert manifest.source_spec == NETWORKS_SOURCE_SPEC.name
    assert dict(manifest.official_file_hashes) == SOURCE_HASHES
    assert manifest.observation_mode is ObservationMode.VISION
    request = json.loads(manifest.generator_request)
    assert request["official_commit"] == manifest.official_commit
    assert request["profile"] == manifest.profile.value
    assert request["source_spec"] == NETWORKS_SOURCE_SPEC.name


def test_network_oracle_replays_exact_worker_command_and_stdin(official_case) -> None:
    if not (OFFICIAL_CHECKOUT / ".git").exists():
        pytest.skip("explicit DreamerV3 oracle checkout is unavailable")
    manifest, arrays = official_case

    replayed = subprocess.run(
        manifest.generator_command,
        cwd=OFFICIAL_CHECKOUT,
        input=manifest.generator_request,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(replayed.stdout)

    assert int(payload["worker_pid"]) != os.getpid()
    assert tuple(sorted(payload["arrays"])) == tuple(arrays)
    for name, spec in payload["arrays"].items():
        np.testing.assert_array_equal(
            arrays[name],
            np.asarray(spec["values"], dtype=spec["dtype"]),
        )


def test_network_oracle_regeneration_is_byte_deterministic(tmp_path: Path) -> None:
    if not (OFFICIAL_CHECKOUT / ".git").exists():
        pytest.skip("explicit DreamerV3 oracle checkout is unavailable")
    first = OracleHarness(OFFICIAL_CHECKOUT, tmp_path / "first")
    second = OracleHarness(OFFICIAL_CHECKOUT, tmp_path / "second")

    first_fixture, first_manifest = first.run_networks_case(
        DreamerProfile.PAPER,
        ObservationMode.VISION,
    )
    second_fixture, second_manifest = second.run_networks_case(
        DreamerProfile.PAPER,
        ObservationMode.VISION,
    )

    assert first_fixture.read_bytes() == second_fixture.read_bytes()
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    assert first.last_worker_pid is not None
    assert second.last_worker_pid is not None


@pytest.mark.parametrize(
    "case",
    ("zeros", "uniform_in", "normal_out", "trunc_normal_avg", "normed_none"),
)
def test_initializer_names_fans_scales_and_dtypes_match_official(
    official_case,
    case: str,
) -> None:
    _, arrays = official_case
    name = arrays[f"initializer.{case}.name"].tobytes().decode()
    scale = arrays[f"initializer.{case}.scale"].item()
    seed = jnp.asarray(arrays[f"initializer.{case}.seed"])
    shape = tuple(arrays[f"initializer.{case}.shape"].tolist())
    fshape = tuple(arrays[f"initializer.{case}.fshape"].tolist()) or None

    actual = Initializer(name, scale=scale)(
        seed,
        shape,
        jnp.float32,
        fshape=fshape,
    )

    assert actual.dtype == jnp.float32
    _assert_f32(actual, arrays[f"initializer.{case}.output"])


def test_initializer_rejects_unknown_distribution_and_invalid_shape() -> None:
    with pytest.raises(ValueError, match="initializer"):
        Initializer("orthogonal_in")
    with pytest.raises(ValueError, match="positive"):
        Initializer("zeros")(jax.random.PRNGKey(0), (0, 2))


def test_rmsnorm_has_no_mean_subtraction_and_translated_forward_parity(
    official_case,
) -> None:
    _, arrays = official_case
    inputs = jnp.asarray(arrays["rms.input"])
    module = RMSNorm()
    initialized = module.init(jax.random.PRNGKey(0), inputs)
    params = _translate_case(arrays, "rms", initialized)

    output = module.apply(params, inputs)

    assert set(_flat_params(params)) == {"scale"}
    assert not np.allclose(np.asarray(inputs.mean(-1)), 0.0)
    _assert_f32(output, arrays["rms.output"])


def test_linear_parameter_names_shapes_and_forward_match_official(
    official_case,
) -> None:
    _, arrays = official_case
    inputs = jnp.asarray(arrays["linear.input"])
    module = Linear(5, initializer="trunc_normal_in")
    initialized = module.init(jax.random.PRNGKey(0), inputs)
    params = _translate_case(arrays, "linear", initialized)

    flat = _flat_params(params)
    assert {name: value.shape for name, value in flat.items()} == {
        "bias": (5,),
        "kernel": (3, 5),
    }
    _assert_f32(module.apply(params, inputs), arrays["linear.output"])


def test_linear_explicit_norm_activation_and_zero_output_initializer() -> None:
    inputs = jnp.asarray([[1.0, -2.0, 3.0]], jnp.float32)
    module = Linear(
        4,
        initializer="trunc_normal_in",
        output_scale=0.0,
        normalization="rms",
        activation="silu",
    )
    variables = module.init(jax.random.PRNGKey(1), inputs)

    flat = _flat_params(variables)
    np.testing.assert_array_equal(flat["kernel"], 0.0)
    assert set(flat) == {"bias", "kernel", "norm/scale"}
    np.testing.assert_array_equal(module.apply(variables, inputs), 0.0)


def test_blocklinear_uses_independent_block_matrices_and_exact_einsum(
    official_case,
) -> None:
    _, arrays = official_case
    inputs = jnp.asarray(arrays["blocklinear.input"])
    module = BlockLinear(8, blocks=4)
    initialized = module.init(jax.random.PRNGKey(0), inputs)
    params = _translate_case(arrays, "blocklinear", initialized)

    flat = _flat_params(params)
    assert flat["kernel"].shape == (4, 2, 2)
    _assert_f32(module.apply(params, inputs), arrays["blocklinear.output"])

    isolated = np.zeros_like(flat["kernel"])
    isolated[2] = 1.0
    changed = dict(flat)
    changed["kernel"] = isolated
    changed["bias"] = np.zeros((8,), np.float32)
    output = module.apply(_nested_params(changed), inputs)
    np.testing.assert_array_equal(output[..., :4], 0.0)
    np.testing.assert_array_equal(output[..., 6:], 0.0)
    assert np.any(np.asarray(output[..., 4:6]) != 0.0)


@pytest.mark.parametrize("prefix", ("conv", "transposed_conv"))
def test_conv2d_regular_and_manual_transposed_paths_match_official(
    official_case,
    prefix: str,
) -> None:
    _, arrays = official_case
    inputs = jnp.asarray(arrays[f"{prefix}.input"])
    transposed = prefix == "transposed_conv"
    module = Conv2D(3, kernel=3, stride=2, transposed=transposed)
    initialized = module.init(jax.random.PRNGKey(0), inputs)
    params = _translate_case(arrays, prefix, initialized)

    flat = _flat_params(params)
    assert flat["kernel"].shape == (3, 3, 2, 3)
    _assert_f32(module.apply(params, inputs), arrays[f"{prefix}.output"])


def test_mlp_has_exact_hidden_layer_count_names_and_forward_parity(
    official_case,
) -> None:
    _, arrays = official_case
    inputs = jnp.asarray(arrays["mlp.input"])
    module = MLP(2, 6, activation="silu", normalization="rms")
    initialized = module.init(jax.random.PRNGKey(0), inputs)
    params = _translate_case(arrays, "mlp", initialized)

    flat = _flat_params(params)
    assert set(flat) == {
        "linear0/bias",
        "linear0/kernel",
        "linear1/bias",
        "linear1/kernel",
        "norm0/scale",
        "norm1/scale",
    }
    _assert_f32(module.apply(params, inputs), arrays["mlp.output"])


def _small_rssm_config() -> RSSMConfig:
    return RSSMConfig(
        deter=16,
        hidden=8,
        stoch=2,
        classes=3,
        blocks=8,
        free_nats=1.0,
        unimix=0.01,
        activation="silu",
        normalization="rms",
        image_layers=2,
        observation_layers=1,
        dynamics_layers=1,
        absolute=False,
        initializer="trunc_normal_in",
        output_scale=1.0,
        _legacy=True,
    )


def test_blockgru_matches_exact_official_projection_group_and_gate_order(
    official_case,
) -> None:
    _, arrays = official_case
    deter = jnp.asarray(arrays["blockgru.deter"])
    stoch = jnp.asarray(arrays["blockgru.stoch"])
    action = jnp.asarray(arrays["blockgru.action"])
    reset = jnp.asarray(arrays["blockgru.is_first"])
    module = BlockGRU(_small_rssm_config(), action_dim=3)
    initialized = module.init(jax.random.PRNGKey(0), deter, stoch, action, reset)
    params = _translate_case(arrays, "blockgru", initialized)

    flat = _flat_params(params)
    assert flat["dynhid0/kernel"].shape == (8, 26, 2)
    assert flat["dyngru/kernel"].shape == (8, 2, 6)
    assert not any(
        value.ndim == 2 and value.shape == (16, 48) for value in flat.values()
    )
    _assert_f32(
        module.apply(params, deter, stoch, action, reset),
        arrays["blockgru.output"],
    )


def test_blockgru_stops_action_scale_gradient_and_zeros_reset_inputs(
    official_case,
) -> None:
    _, arrays = official_case
    deter = jnp.asarray(arrays["blockgru.deter"])
    stoch = jnp.asarray(arrays["blockgru.stoch"])
    action = jnp.asarray(arrays["blockgru.action"])
    reset = jnp.asarray(arrays["blockgru.is_first"])
    module = BlockGRU(_small_rssm_config(), action_dim=3)
    initialized = module.init(jax.random.PRNGKey(0), deter, stoch, action, reset)
    params = _translate_case(arrays, "blockgru", initialized)

    gradient = jax.grad(
        lambda value: module.apply(params, deter, stoch, value, reset).sum()
    )(action)
    _assert_f32(gradient, arrays["blockgru.grad_action"])

    changed_deter = deter.at[0].set(999.0)
    changed_stoch = stoch.at[0].set(-999.0)
    changed_action = action.at[0].set(999.0)
    original = module.apply(params, deter, stoch, action, reset)
    changed = module.apply(
        params,
        changed_deter,
        changed_stoch,
        changed_action,
        reset,
    )
    _assert_f32(changed[0], original[0])


def _encoder_spaces() -> dict[str, TensorSpace]:
    return {
        "a_image": TensorSpace((32, 32, 1), "uint8"),
        "m_vector": TensorSpace((2,), "float32"),
        "z_image": TensorSpace((32, 32, 2), "uint8"),
        "z_vector": TensorSpace((1,), "float32"),
    }


def _encoder_config(profile: DreamerProfile) -> EncoderConfig:
    return EncoderConfig(
        depth=2,
        multipliers=(1, 2, 2),
        layers=2,
        units=6,
        activation="silu",
        normalization="rms",
        initializer="trunc_normal_in",
        symlog=True,
        outer=False,
        kernel=3,
        strided=profile is DreamerProfile.PAPER,
    )


def test_dictencoder_sorted_key_partition_symlog_and_mixed_forward_parity(
    official_case,
) -> None:
    manifest, arrays = official_case
    spaces = _encoder_spaces()
    obs = {key: jnp.asarray(arrays[f"encoder.input.{key}"]) for key in reversed(spaces)}
    module = DictEncoder(spaces, _encoder_config(manifest.profile))
    initialized = module.init(jax.random.PRNGKey(0), obs)
    params = _translate_case(arrays, "encoder", initialized)

    output = module.apply(params, obs)

    assert output.shape == tuple(arrays["encoder.output"].shape)
    _assert_f32(output, arrays["encoder.output"])
    reordered = {key: obs[key] for key in sorted(obs)}
    _assert_f32(module.apply(params, reordered), output)


def test_dictencoder_uses_distinct_paper_stride_and_current_maxpool_branches(
    official_case,
) -> None:
    manifest, arrays = official_case
    obs = {
        key: jnp.asarray(arrays[f"encoder.input.{key}"]) for key in _encoder_spaces()
    }
    image_only = {key: value for key, value in obs.items() if "image" in key}
    image_module = DictEncoder(
        {key: _encoder_spaces()[key] for key in image_only},
        _encoder_config(manifest.profile),
    )
    image_initialized = image_module.init(jax.random.PRNGKey(0), image_only)
    image_params = _translate_case(arrays, "encoder_image", image_initialized)
    output = image_module.apply(image_params, image_only)

    _assert_f32(output, arrays["encoder_image.output"])
    assert output.shape[-1] == arrays["encoder_image.output"].shape[-1]


def test_dictencoder_vector_only_path_matches_official_symlog_mlp(
    official_case,
) -> None:
    manifest, arrays = official_case
    spaces = {key: space for key, space in _encoder_spaces().items() if "vector" in key}
    obs = {key: jnp.asarray(arrays[f"encoder.input.{key}"]) for key in spaces}
    module = DictEncoder(spaces, _encoder_config(manifest.profile))
    initialized = module.init(jax.random.PRNGKey(0), obs)
    params = _translate_case(arrays, "encoder_vector", initialized)

    _assert_f32(module.apply(params, obs), arrays["encoder_vector.output"])


def test_dictencoder_rejects_metadata_and_wrong_image_dtype() -> None:
    config = _encoder_config(DreamerProfile.PAPER)
    metadata = DictEncoder({"is_first": TensorSpace((), "bool")}, config)
    with pytest.raises(ValueError, match="metadata"):
        metadata.init(jax.random.PRNGKey(0), {"is_first": jnp.ones((1,), bool)})

    module = DictEncoder(
        {"image": TensorSpace((32, 32, 3), "uint8")},
        config,
    )
    with pytest.raises((TypeError, ValueError), match="uint8"):
        module.init(
            jax.random.PRNGKey(0),
            {"image": jnp.zeros((1, 32, 32, 3), jnp.float32)},
        )


def _decoder_spaces() -> dict[str, TensorSpace]:
    return {
        "a_image": TensorSpace((32, 32, 1), "uint8"),
        "m_cont": TensorSpace((2,), "float32"),
        "z_disc": TensorSpace((1,), "int32", classes=3),
        "z_image": TensorSpace((32, 32, 2), "uint8"),
    }


def _decoder_config(profile: DreamerProfile) -> DecoderConfig:
    return DecoderConfig(
        depth=2,
        multipliers=(1, 2, 2),
        layers=2,
        units=6,
        activation="silu",
        normalization="rms",
        output_scale=1.0,
        initializer="trunc_normal_in",
        outer=False,
        kernel=3,
        bias_space=2,
        strided=profile is DreamerProfile.PAPER,
        image_output="mse",
    )


def test_dictdecoder_output_families_targets_and_no_batch_reduction(
    official_case,
) -> None:
    manifest, arrays = official_case
    feat = {
        "deter": jnp.asarray(arrays["decoder.deter"]),
        "stoch": jnp.asarray(arrays["decoder.stoch"]),
    }
    spaces = _decoder_spaces()
    module = DictDecoder(spaces, _decoder_config(manifest.profile))
    initialized = module.init(jax.random.PRNGKey(0), feat)
    params = _translate_case(arrays, "decoder", initialized)

    outputs = module.apply(params, feat)

    assert set(outputs) == set(spaces)
    for key in ("a_image", "z_image", "m_cont", "z_disc"):
        _assert_f32(outputs[key].pred(), arrays[f"decoder.pred.{key}"])
    assert outputs["a_image"].pred().shape[:2] == feat["deter"].shape[:2]
    assert outputs["m_cont"].loss(jnp.ones((2, 2, 2), jnp.float32)).shape == (2, 2)
    image_target = jnp.asarray(arrays["decoder.target.a_image"])
    assert image_target.min() >= 0.0 and image_target.max() <= 1.0
    _assert_f32(
        outputs["a_image"].loss(image_target),
        arrays["decoder.loss.a_image"],
    )


def test_dictdecoder_profile_image_branch_matches_exact_resolution(
    official_case,
) -> None:
    manifest, arrays = official_case
    feat = {
        "deter": jnp.asarray(arrays["decoder.deter"]),
        "stoch": jnp.asarray(arrays["decoder.stoch"]),
    }
    spaces = {"image": TensorSpace((32, 32, 3), "uint8")}
    module = DictDecoder(spaces, _decoder_config(manifest.profile))
    initialized = module.init(jax.random.PRNGKey(0), feat)
    params = _translate_case(arrays, "decoder_image", initialized)

    output = module.apply(params, feat)["image"].pred()

    assert output.shape == (2, 2, 32, 32, 3)
    _assert_f32(output, arrays["decoder_image.output"])


def test_dictdecoder_vector_only_continuous_and_discrete_paths_match_official(
    official_case,
) -> None:
    manifest, arrays = official_case
    feat = {
        "deter": jnp.asarray(arrays["decoder.deter"]),
        "stoch": jnp.asarray(arrays["decoder.stoch"]),
    }
    spaces = {key: space for key, space in _decoder_spaces().items() if not space.image}
    module = DictDecoder(spaces, _decoder_config(manifest.profile))
    initialized = module.init(jax.random.PRNGKey(0), feat)
    params = _translate_case(arrays, "decoder_vector", initialized)

    outputs = module.apply(params, feat)

    _assert_f32(outputs["m_cont"].pred(), arrays["decoder_vector.pred.m_cont"])
    _assert_f32(outputs["z_disc"].pred(), arrays["decoder_vector.pred.z_disc"])


@pytest.mark.parametrize(
    ("name", "space", "config"),
    [
        (
            "reward",
            TensorSpace((), "float32"),
            HeadConfig(layers=1, units=6, output="symexp_twohot", output_scale=0.0),
        ),
        (
            "continue",
            TensorSpace((), "bool", classes=2),
            HeadConfig(layers=1, units=6, output="binary", bins=None),
        ),
        (
            "policy",
            TensorSpace((3,), "float32"),
            PolicyConfig(layers=2, units=6),
        ),
        (
            "categorical",
            TensorSpace((1,), "int32", classes=4),
            HeadConfig(layers=1, units=6, output="categorical", bins=None),
        ),
    ],
)
def test_mlphead_exact_hidden_output_layers_and_output_families(
    official_case,
    name: str,
    space: TensorSpace,
    config: HeadConfig | PolicyConfig,
) -> None:
    _, arrays = official_case
    inputs = jnp.asarray(arrays[f"head.{name}.input"])
    module = MLPHead(space, config)
    initialized = module.init(jax.random.PRNGKey(0), inputs)
    params = _translate_case(arrays, f"head.{name}", initialized)

    output = module.apply(params, inputs)

    _assert_f32(output.pred(), arrays[f"head.{name}.output"])
    flat = _flat_params(params)
    assert (
        sum(path.startswith("mlp/linear") and path.endswith("kernel") for path in flat)
        == config.layers
    )
    assert any(path.startswith("head/") and path.endswith("kernel") for path in flat)
    if name == "reward":
        np.testing.assert_array_equal(flat["head/logits/kernel"], 0.0)
    if name == "policy":
        assert np.all(np.asarray(output.output.stddev) >= config.min_std)
        assert np.all(np.asarray(output.output.stddev) <= config.max_std)


def test_task_three_interfaces_are_exported_from_package_boundary() -> None:
    expected = {
        "BlockGRU",
        "BlockLinear",
        "Conv2D",
        "DictDecoder",
        "DictEncoder",
        "Initializer",
        "Linear",
        "MLP",
        "MLPHead",
        "NETWORKS_SOURCE_SPEC",
        "RMSNorm",
        "TensorSpace",
    }

    assert expected <= set(dreamer_v3.__all__)
    assert all(getattr(dreamer_v3, name) is not None for name in expected)
