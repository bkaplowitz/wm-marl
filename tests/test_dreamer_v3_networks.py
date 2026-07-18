from __future__ import annotations

import json
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
    OracleManifest,
    ParameterTranslator,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "dreamer_v3"
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


def _assert_source_equal(actual, expected) -> None:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    uses_bfloat16 = "bfloat16" in (actual.dtype.name, expected.dtype.name)
    tolerance = 2e-2 if uses_bfloat16 else 1e-5
    if actual.dtype.name == "bfloat16":
        actual = actual.astype(np.float32)
    else:
        assert actual.dtype.name == expected.dtype.name
    if expected.dtype.name == "bfloat16":
        expected = expected.astype(np.float32)
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def _compute_dtype(name: str):
    return {"bfloat16": jnp.bfloat16, "float32": jnp.float32}[name]


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
    params=(
        (DreamerProfile.PAPER, "bfloat16"),
        (DreamerProfile.PAPER, "float32"),
        (DreamerProfile.UPSTREAM_CURRENT, "bfloat16"),
        (DreamerProfile.UPSTREAM_CURRENT, "float32"),
    ),
    ids=lambda item: f"{item[0].value}-{item[1]}",
)
def official_case(request) -> tuple[OracleManifest, dict[str, np.ndarray]]:
    profile, compute_dtype = request.param
    suffix = "" if compute_dtype == "bfloat16" else "-float32"
    stem = f"{profile.value}-vision-networks{suffix}"
    fixture_path = FIXTURE_DIR / f"{stem}.npz"
    manifest_path = FIXTURE_DIR / f"{stem}.manifest.json"
    manifest = OracleManifest.load(manifest_path, fixture_path=fixture_path)
    with np.load(fixture_path, allow_pickle=False) as fixture:
        arrays = {name: fixture[name] for name in fixture.files}
    return manifest, arrays


def test_network_manifests_record_the_dtype_of_the_frozen_fixture(
    official_case,
) -> None:
    manifest, arrays = official_case

    request = json.loads(manifest.generator_request)
    assert request["dtype"] == manifest.dtype
    assert arrays["execution.compute_dtype"].tobytes().decode() == manifest.dtype
    assert all(
        value.dtype.name == "float32"
        for name, value in arrays.items()
        if ".param." in name
    )


def test_network_oracles_pin_exact_three_file_authority(official_case) -> None:
    manifest, _ = official_case

    assert manifest.source_spec == NETWORKS_SOURCE_SPEC.name
    assert dict(manifest.official_file_hashes) == SOURCE_HASHES
    assert manifest.observation_mode is ObservationMode.VISION
    request = json.loads(manifest.generator_request)
    assert request["source_revision"] == manifest.official_commit
    assert request["profile"] == manifest.profile.value
    assert request["source_spec"] == NETWORKS_SOURCE_SPEC.name


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
    _assert_source_equal(actual, arrays[f"initializer.{case}.output"])


def test_initializer_rejects_unknown_distribution_and_invalid_shape() -> None:
    with pytest.raises(ValueError, match="initializer"):
        Initializer("orthogonal_in")
    with pytest.raises(ValueError, match="positive"):
        Initializer("zeros")(jax.random.PRNGKey(0), (0, 2))


def test_rmsnorm_has_no_mean_subtraction_and_translated_forward_parity(
    official_case,
) -> None:
    manifest, arrays = official_case
    inputs = jnp.asarray(arrays["rms.input"]).astype(_compute_dtype(manifest.dtype))
    module = RMSNorm(compute_dtype=_compute_dtype(manifest.dtype))
    initialized = module.init(jax.random.PRNGKey(0), inputs)
    params = _translate_case(arrays, "rms", initialized)

    output = module.apply(params, inputs)

    assert set(_flat_params(params)) == {"scale"}
    assert not np.allclose(np.asarray(inputs.mean(-1)), 0.0)
    _assert_source_equal(output, arrays["rms.output"])


def test_linear_parameter_names_shapes_and_forward_match_official(
    official_case,
) -> None:
    manifest, arrays = official_case
    inputs = jnp.asarray(arrays["linear.input"]).astype(_compute_dtype(manifest.dtype))
    module = Linear(
        5,
        initializer="trunc_normal_in",
        compute_dtype=_compute_dtype(manifest.dtype),
    )
    initialized = module.init(jax.random.PRNGKey(0), inputs)
    params = _translate_case(arrays, "linear", initialized)

    flat = _flat_params(params)
    assert {name: value.shape for name, value in flat.items()} == {
        "bias": (5,),
        "kernel": (3, 5),
    }
    _assert_source_equal(module.apply(params, inputs), arrays["linear.output"])


def test_linear_explicit_norm_activation_and_zero_output_initializer() -> None:
    inputs = jnp.asarray([[1.0, -2.0, 3.0]], jnp.float32)
    module = Linear(
        4,
        initializer="trunc_normal_in",
        output_scale=0.0,
        normalization="rms",
        activation="silu",
        compute_dtype=jnp.float32,
    )
    variables = module.init(jax.random.PRNGKey(1), inputs)

    flat = _flat_params(variables)
    np.testing.assert_array_equal(flat["kernel"], 0.0)
    assert set(flat) == {"bias", "kernel", "norm/scale"}
    np.testing.assert_array_equal(module.apply(variables, inputs), 0.0)


def test_blocklinear_uses_independent_block_matrices_and_exact_einsum(
    official_case,
) -> None:
    manifest, arrays = official_case
    inputs = jnp.asarray(arrays["blocklinear.input"]).astype(
        _compute_dtype(manifest.dtype)
    )
    module = BlockLinear(
        8,
        blocks=4,
        compute_dtype=_compute_dtype(manifest.dtype),
    )
    initialized = module.init(jax.random.PRNGKey(0), inputs)
    params = _translate_case(arrays, "blocklinear", initialized)

    flat = _flat_params(params)
    assert flat["kernel"].shape == (4, 2, 2)
    _assert_source_equal(module.apply(params, inputs), arrays["blocklinear.output"])

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
    manifest, arrays = official_case
    inputs = jnp.asarray(arrays[f"{prefix}.input"]).astype(
        _compute_dtype(manifest.dtype)
    )
    transposed = prefix == "transposed_conv"
    module = Conv2D(
        3,
        kernel=3,
        stride=2,
        transposed=transposed,
        compute_dtype=_compute_dtype(manifest.dtype),
    )
    initialized = module.init(jax.random.PRNGKey(0), inputs)
    params = _translate_case(arrays, prefix, initialized)

    flat = _flat_params(params)
    assert flat["kernel"].shape == (3, 3, 2, 3)
    _assert_source_equal(module.apply(params, inputs), arrays[f"{prefix}.output"])


def test_mlp_has_exact_hidden_layer_count_names_and_forward_parity(
    official_case,
) -> None:
    manifest, arrays = official_case
    inputs = jnp.asarray(arrays["mlp.input"])
    module = MLP(
        2,
        6,
        activation="silu",
        normalization="rms",
        compute_dtype=_compute_dtype(manifest.dtype),
    )
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
    _assert_source_equal(module.apply(params, inputs), arrays["mlp.output"])


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
    manifest, arrays = official_case
    deter = jnp.asarray(arrays["blockgru.deter"])
    stoch = jnp.asarray(arrays["blockgru.stoch"])
    action = jnp.asarray(arrays["blockgru.action"])
    reset = jnp.asarray(arrays["blockgru.is_first"])
    module = BlockGRU(
        _small_rssm_config(),
        action_dim=3,
        compute_dtype=_compute_dtype(manifest.dtype),
    )
    initialized = module.init(jax.random.PRNGKey(0), deter, stoch, action, reset)
    params = _translate_case(arrays, "blockgru", initialized)

    flat = _flat_params(params)
    assert flat["dynhid0/kernel"].shape == (8, 26, 2)
    assert flat["dyngru/kernel"].shape == (8, 2, 6)
    assert not any(
        value.ndim == 2 and value.shape == (16, 48) for value in flat.values()
    )
    _assert_source_equal(
        module.apply(params, deter, stoch, action, reset),
        arrays["blockgru.output"],
    )


def test_blockgru_stops_action_scale_gradient_and_zeros_reset_inputs(
    official_case,
) -> None:
    manifest, arrays = official_case
    deter = jnp.asarray(arrays["blockgru.deter"])
    stoch = jnp.asarray(arrays["blockgru.stoch"])
    action = jnp.asarray(arrays["blockgru.action"])
    reset = jnp.asarray(arrays["blockgru.is_first"])
    module = BlockGRU(
        _small_rssm_config(),
        action_dim=3,
        compute_dtype=_compute_dtype(manifest.dtype),
    )
    initialized = module.init(jax.random.PRNGKey(0), deter, stoch, action, reset)
    params = _translate_case(arrays, "blockgru", initialized)

    gradient = jax.grad(
        lambda value: module.apply(params, deter, stoch, value, reset).sum()
    )(action)
    _assert_source_equal(gradient, arrays["blockgru.grad_action"])

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
    _assert_source_equal(changed[0], original[0])


def _encoder_spaces() -> dict[str, TensorSpace]:
    return {
        "a_image": TensorSpace((32, 32, 1), "float32"),
        "m_vector": TensorSpace((2,), "float16"),
        "z_image": TensorSpace((32, 32, 2), "float32"),
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
    module = DictEncoder(
        spaces,
        _encoder_config(manifest.profile),
        compute_dtype=_compute_dtype(manifest.dtype),
    )
    initialized = module.init(jax.random.PRNGKey(0), obs)
    params = _translate_case(arrays, "encoder", initialized)

    output = module.apply(params, obs)

    assert output.shape == tuple(arrays["encoder.output"].shape)
    _assert_source_equal(output, arrays["encoder.output"])
    reordered = {key: obs[key] for key in sorted(obs)}
    _assert_source_equal(module.apply(params, reordered), output)


def test_dictencoder_accepts_full_official_agent_observation_superset(
    official_case,
) -> None:
    manifest, arrays = official_case
    model_keys = tuple(_encoder_spaces())
    metadata_keys = ("is_first", "is_last", "is_terminal", "reward")
    observations = {
        key: jnp.asarray(arrays[f"encoder.input.{key}"])
        for key in (*model_keys, *metadata_keys)
    }
    module = DictEncoder(
        _encoder_spaces(),
        _encoder_config(manifest.profile),
        compute_dtype=_compute_dtype(manifest.dtype),
    )
    initialized = module.init(jax.random.PRNGKey(0), observations)
    params = _translate_case(arrays, "encoder", initialized)

    output = module.apply(params, observations)

    _assert_source_equal(output, arrays["encoder.output"])
    _assert_source_equal(output, arrays["encoder.subset_output"])
    for key in model_keys:
        assert arrays[f"encoder.missing.{key}.rejected"].item() == 1


def test_dictencoder_superset_still_rejects_missing_and_invalid_required_values(
    official_case,
) -> None:
    manifest, arrays = official_case
    spaces = _encoder_spaces()
    observations = {key: jnp.asarray(arrays[f"encoder.input.{key}"]) for key in spaces}
    observations["is_first"] = jnp.asarray(arrays["encoder.input.is_first"])
    module = DictEncoder(
        spaces,
        _encoder_config(manifest.profile),
        compute_dtype=_compute_dtype(manifest.dtype),
    )

    for missing_key in spaces:
        missing = {
            key: value for key, value in observations.items() if key != missing_key
        }
        with pytest.raises(ValueError, match="required observation keys"):
            module.init(jax.random.PRNGKey(0), missing)

    wrong_vector = dict(observations)
    wrong_vector["m_vector"] = wrong_vector["m_vector"][..., :1]
    with pytest.raises(ValueError, match="vector shape mismatch"):
        module.init(jax.random.PRNGKey(0), wrong_vector)

    wrong_image = dict(observations)
    wrong_image["a_image"] = wrong_image["a_image"].astype(jnp.float32)
    with pytest.raises(TypeError, match="uint8"):
        module.init(jax.random.PRNGKey(0), wrong_image)


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
        compute_dtype=_compute_dtype(manifest.dtype),
    )
    image_initialized = image_module.init(jax.random.PRNGKey(0), image_only)
    image_params = _translate_case(arrays, "encoder_image", image_initialized)
    output = image_module.apply(image_params, image_only)

    _assert_source_equal(output, arrays["encoder_image.output"])
    assert output.shape[-1] == arrays["encoder_image.output"].shape[-1]


def test_dictencoder_vector_only_path_matches_official_symlog_mlp(
    official_case,
) -> None:
    manifest, arrays = official_case
    spaces = {key: space for key, space in _encoder_spaces().items() if "vector" in key}
    obs = {key: jnp.asarray(arrays[f"encoder.input.{key}"]) for key in spaces}
    module = DictEncoder(
        spaces,
        _encoder_config(manifest.profile),
        compute_dtype=_compute_dtype(manifest.dtype),
    )
    initialized = module.init(jax.random.PRNGKey(0), obs)
    params = _translate_case(arrays, "encoder_vector", initialized)

    _assert_source_equal(module.apply(params, obs), arrays["encoder_vector.output"])


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


def test_networks_default_to_bfloat16_compute_with_float32_parameters() -> None:
    mlp_input = jnp.asarray([[1.0, -2.0, 3.0]], jnp.float32)
    canonical = MLP(1, 4)
    canonical_variables = canonical.init(jax.random.PRNGKey(10), mlp_input)
    canonical_output = canonical.apply(canonical_variables, mlp_input)

    explicit = MLP(1, 4, compute_dtype=jnp.float32)
    explicit_variables = explicit.init(jax.random.PRNGKey(10), mlp_input)
    explicit_output = explicit.apply(explicit_variables, mlp_input)

    assert canonical.compute_dtype == jnp.bfloat16
    assert canonical_output.dtype == jnp.bfloat16
    assert explicit_output.dtype == jnp.float32
    assert all(
        value.dtype.name == "float32"
        for value in _flat_params(canonical_variables).values()
    )

    primitive_inputs = {
        RMSNorm(): jnp.ones((1, 4), jnp.bfloat16),
        Linear(4): jnp.ones((1, 3), jnp.bfloat16),
        BlockLinear(4, 2): jnp.ones((1, 4), jnp.bfloat16),
        Conv2D(2, 3): jnp.ones((1, 4, 4, 1), jnp.bfloat16),
    }
    for index, (module, value) in enumerate(primitive_inputs.items()):
        variables = module.init(jax.random.PRNGKey(20 + index), value)
        assert module.compute_dtype == jnp.bfloat16
        assert module.apply(variables, value).dtype == jnp.bfloat16
        assert all(
            item.dtype.name == "float32" for item in _flat_params(variables).values()
        )


def test_dictencoder_and_blockgru_default_composites_emit_bfloat16() -> None:
    encoder = DictEncoder(
        {"vector": TensorSpace((2,), "float32")},
        EncoderConfig(layers=1, units=4),
    )
    observations = {"vector": jnp.asarray([[1.0, -2.0]], jnp.float32)}
    encoder_variables = encoder.init(jax.random.PRNGKey(30), observations)
    assert encoder.apply(encoder_variables, observations).dtype == jnp.bfloat16

    config = _small_rssm_config()
    blockgru = BlockGRU(config, action_dim=3)
    deter = jnp.ones((1, config.deter), jnp.float32)
    stoch = jnp.ones((1, config.stoch, config.classes), jnp.float32)
    action = jnp.ones((1, 3), jnp.float32)
    variables = blockgru.init(jax.random.PRNGKey(31), deter, stoch, action)
    assert blockgru.apply(variables, deter, stoch, action).dtype == jnp.bfloat16


def test_dictionary_partition_is_rank_based_and_casts_vector_runtime_dtypes() -> None:
    encoder_config = EncoderConfig(layers=1, units=4, multipliers=(1,))
    decoder_config = DecoderConfig(
        layers=1,
        units=4,
        depth=2,
        multipliers=(1,),
        kernel=3,
        bias_space=2,
    )
    rank_three_float = TensorSpace((8, 8, 1), "float32")

    image_encoder = DictEncoder(
        {"image": rank_three_float},
        encoder_config,
        compute_dtype=jnp.float32,
    )
    image_observations = {"image": jnp.zeros((1, 8, 8, 1), jnp.uint8)}
    image_variables = image_encoder.init(jax.random.PRNGKey(40), image_observations)
    assert image_encoder.apply(image_variables, image_observations).ndim == 2

    vector_encoder = DictEncoder(
        {"vector": TensorSpace((2,), "float32")},
        encoder_config,
        compute_dtype=jnp.float32,
    )
    vector_observations = {"vector": jnp.zeros((1, 2), jnp.float16)}
    vector_variables = vector_encoder.init(jax.random.PRNGKey(42), vector_observations)
    assert (
        vector_encoder.apply(vector_variables, vector_observations).dtype == jnp.float32
    )

    discrete_encoder = DictEncoder(
        {"vector": TensorSpace((2,), "int32", classes=3)},
        encoder_config,
        compute_dtype=jnp.float32,
    )
    discrete_observations = {"vector": jnp.zeros((1, 2), jnp.int16)}
    discrete_variables = discrete_encoder.init(
        jax.random.PRNGKey(43), discrete_observations
    )
    assert (
        discrete_encoder.apply(discrete_variables, discrete_observations).dtype
        == jnp.float32
    )

    decoder = DictDecoder(
        {"image": rank_three_float},
        decoder_config,
        compute_dtype=jnp.float32,
    )
    features = {
        "deter": jnp.zeros((1, 4), jnp.float32),
        "stoch": jnp.zeros((1, 1, 2), jnp.float32),
    }
    decoder_variables = decoder.init(jax.random.PRNGKey(41), features)
    assert decoder.apply(decoder_variables, features)["image"].pred().shape == (
        1,
        8,
        8,
        1,
    )


def _decoder_spaces() -> dict[str, TensorSpace]:
    return {
        "a_image": TensorSpace((32, 32, 1), "float32"),
        "m_cont": TensorSpace((2,), "float32"),
        "z_disc": TensorSpace((1,), "int32", classes=3),
        "z_image": TensorSpace((32, 32, 2), "float32"),
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
    module = DictDecoder(
        spaces,
        _decoder_config(manifest.profile),
        compute_dtype=_compute_dtype(manifest.dtype),
    )
    initialized = module.init(jax.random.PRNGKey(0), feat)
    params = _translate_case(arrays, "decoder", initialized)

    outputs = module.apply(params, feat)

    assert set(outputs) == set(spaces)
    for key in ("a_image", "z_image", "m_cont", "z_disc"):
        _assert_source_equal(outputs[key].pred(), arrays[f"decoder.pred.{key}"])
    assert outputs["a_image"].pred().shape[:2] == feat["deter"].shape[:2]
    assert outputs["m_cont"].loss(jnp.ones((2, 2, 2), jnp.float32)).shape == (2, 2)
    image_target = jnp.asarray(arrays["decoder.target.a_image"])
    assert image_target.min() >= 0.0 and image_target.max() <= 1.0
    _assert_source_equal(
        outputs["a_image"].loss(image_target),
        arrays["decoder.loss.a_image"],
    )


def test_dictdecoder_accepts_full_official_rssm_feature_superset(
    official_case,
) -> None:
    manifest, arrays = official_case
    features = {
        key: jnp.asarray(arrays[f"decoder.{key}"])
        for key in ("deter", "stoch", "logit")
    }
    module = DictDecoder(
        _decoder_spaces(),
        _decoder_config(manifest.profile),
        compute_dtype=_compute_dtype(manifest.dtype),
    )
    initialized = module.init(jax.random.PRNGKey(0), features)
    params = _translate_case(arrays, "decoder", initialized)

    outputs = module.apply(params, features)

    for key, output in outputs.items():
        target = jnp.asarray(arrays[f"decoder.target.{key}"])
        _assert_source_equal(output.pred(), arrays[f"decoder.pred.{key}"])
        _assert_source_equal(output.pred(), arrays[f"decoder.subset.pred.{key}"])
        _assert_source_equal(output.loss(target), arrays[f"decoder.loss.{key}"])
        _assert_source_equal(output.loss(target), arrays[f"decoder.subset.loss.{key}"])
    for key in ("deter", "stoch"):
        assert arrays[f"decoder.missing.{key}.rejected"].item() == 1


def test_dictdecoder_superset_still_rejects_missing_and_invalid_required_values(
    official_case,
) -> None:
    manifest, arrays = official_case
    features = {
        key: jnp.asarray(arrays[f"decoder.{key}"])
        for key in ("deter", "stoch", "logit")
    }
    module = DictDecoder(
        _decoder_spaces(),
        _decoder_config(manifest.profile),
        compute_dtype=_compute_dtype(manifest.dtype),
    )

    for missing_key in ("deter", "stoch"):
        missing = {key: value for key, value in features.items() if key != missing_key}
        with pytest.raises(ValueError, match="required features"):
            module.init(jax.random.PRNGKey(0), missing)

    invalid = dict(features)
    invalid["deter"] = invalid["deter"].astype(jnp.int32)
    with pytest.raises(TypeError, match="floating dtypes"):
        module.init(jax.random.PRNGKey(0), invalid)


def test_dictdecoder_profile_image_branch_matches_exact_resolution(
    official_case,
) -> None:
    manifest, arrays = official_case
    feat = {
        "deter": jnp.asarray(arrays["decoder.deter"]),
        "stoch": jnp.asarray(arrays["decoder.stoch"]),
    }
    spaces = {"image": TensorSpace((32, 32, 3), "float32")}
    module = DictDecoder(
        spaces,
        _decoder_config(manifest.profile),
        compute_dtype=_compute_dtype(manifest.dtype),
    )
    initialized = module.init(jax.random.PRNGKey(0), feat)
    params = _translate_case(arrays, "decoder_image", initialized)

    output = module.apply(params, feat)["image"].pred()

    assert output.shape == (2, 2, 32, 32, 3)
    _assert_source_equal(output, arrays["decoder_image.output"])


def test_dictdecoder_vector_only_continuous_and_discrete_paths_match_official(
    official_case,
) -> None:
    manifest, arrays = official_case
    feat = {
        "deter": jnp.asarray(arrays["decoder.deter"]),
        "stoch": jnp.asarray(arrays["decoder.stoch"]),
    }
    spaces = {key: space for key, space in _decoder_spaces().items() if not space.image}
    module = DictDecoder(
        spaces,
        _decoder_config(manifest.profile),
        compute_dtype=_compute_dtype(manifest.dtype),
    )
    initialized = module.init(jax.random.PRNGKey(0), feat)
    params = _translate_case(arrays, "decoder_vector", initialized)

    outputs = module.apply(params, feat)

    _assert_source_equal(outputs["m_cont"].pred(), arrays["decoder_vector.pred.m_cont"])
    _assert_source_equal(outputs["z_disc"].pred(), arrays["decoder_vector.pred.z_disc"])


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
    manifest, arrays = official_case
    inputs = jnp.asarray(arrays[f"head.{name}.input"])
    module = MLPHead(
        space,
        config,
        compute_dtype=_compute_dtype(manifest.dtype),
    )
    initialized = module.init(jax.random.PRNGKey(0), inputs)
    params = _translate_case(arrays, f"head.{name}", initialized)

    output = module.apply(params, inputs)

    _assert_source_equal(output.pred(), arrays[f"head.{name}.output"])
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


@pytest.mark.parametrize(
    ("family", "space"),
    (
        ("binary", TensorSpace((), "bool", classes=2)),
        ("categorical", TensorSpace((2,), "int32", classes=3)),
        ("onehot", TensorSpace((2,), "int32", classes=3)),
        ("mse", TensorSpace((2,), "float32")),
        ("symlog_mse", TensorSpace((), "float32")),
        ("symexp_twohot", TensorSpace((2,), "float32")),
        ("bounded_normal", TensorSpace((2,), "float32")),
    ),
)
def test_mlphead_supports_exact_valid_family_space_pairs(
    family: str,
    space: TensorSpace,
) -> None:
    if family in {"onehot", "bounded_normal"}:
        config: HeadConfig | PolicyConfig = PolicyConfig(
            layers=1,
            units=4,
            discrete=family,
            continuous=family,
        )
    else:
        config = HeadConfig(
            layers=1,
            units=4,
            output=family,
            bins=7 if family == "symexp_twohot" else None,
        )
    module = MLPHead(space, config, compute_dtype=jnp.float32)
    inputs = jnp.ones((3, 5), jnp.float32)
    variables = module.init(jax.random.PRNGKey(50), inputs)

    output = module.apply(variables, inputs)

    expected_shape = (3, *space.shape)
    if family == "onehot":
        expected_shape = (*expected_shape, 3)
        assert _flat_params(variables)["head/logits/kernel"].shape == (4, 6)
    assert output.pred().shape == expected_shape


@pytest.mark.parametrize(
    ("family", "space"),
    (
        ("binary", TensorSpace((), "float32")),
        ("categorical", TensorSpace((2,), "float32")),
        ("onehot", TensorSpace((2,), "float32")),
        ("mse", TensorSpace((2,), "int32", classes=3)),
        ("symlog_mse", TensorSpace((2,), "int32", classes=3)),
        ("symexp_twohot", TensorSpace((2,), "int32", classes=3)),
        ("bounded_normal", TensorSpace((2,), "int32", classes=3)),
    ),
)
def test_mlphead_rejects_every_invalid_family_space_pair(
    family: str,
    space: TensorSpace,
) -> None:
    if family in {"onehot", "bounded_normal"}:
        config: HeadConfig | PolicyConfig = PolicyConfig(
            layers=0,
            units=4,
            discrete=family,
            continuous=family,
        )
    else:
        config = HeadConfig(
            layers=0,
            units=4,
            output=family,
            bins=7 if family == "symexp_twohot" else None,
        )
    module = MLPHead(space, config, compute_dtype=jnp.float32)

    with pytest.raises(ValueError, match="requires"):
        module.init(jax.random.PRNGKey(51), jnp.ones((2, 4), jnp.float32))


def test_mlphead_rejects_nonuniform_discrete_classes() -> None:
    with pytest.raises(ValueError, match="uniform"):
        space = TensorSpace((2,), "int32", classes=(2, 3))
        MLPHead(
            space,
            HeadConfig(layers=0, units=4, output="categorical", bins=None),
            compute_dtype=jnp.float32,
        ).init(jax.random.PRNGKey(52), jnp.ones((2, 4), jnp.float32))


def test_categorical_and_bounded_normal_expose_official_entropy_bounds() -> None:
    inputs = jnp.ones((2, 4), jnp.float32)
    categorical = MLPHead(
        TensorSpace((), "int32", classes=4),
        HeadConfig(layers=0, units=4, output="categorical", bins=None),
        compute_dtype=jnp.float32,
    )
    bounded = MLPHead(
        TensorSpace((2,), "float32"),
        PolicyConfig(layers=0, units=4),
        compute_dtype=jnp.float32,
    )
    categorical_output = categorical.apply(
        categorical.init(jax.random.PRNGKey(53), inputs), inputs
    )
    bounded_output = bounded.apply(bounded.init(jax.random.PRNGKey(54), inputs), inputs)

    assert categorical_output.minent == 0
    np.testing.assert_allclose(categorical_output.maxent, np.log(4), rtol=0, atol=0)
    assert hasattr(bounded_output.output, "minent")
    assert hasattr(bounded_output.output, "maxent")
    assert np.all(bounded_output.output.minent <= bounded_output.output.maxent)


def test_dictdecoder_preserves_declared_nonlexicographic_image_channel_order() -> None:
    spaces = {
        "z_image": TensorSpace((32, 32, 2), "uint8"),
        "a_image": TensorSpace((32, 32, 1), "uint8"),
    }
    features = {
        "deter": jnp.zeros((1, 16), jnp.float32),
        "stoch": jnp.zeros((1, 2, 3), jnp.float32),
    }
    module = DictDecoder(
        spaces,
        _decoder_config(DreamerProfile.PAPER),
        compute_dtype=jnp.float32,
    )
    variables = module.init(jax.random.PRNGKey(55), features)
    flat = {
        name: np.zeros_like(value) for name, value in _flat_params(variables).items()
    }
    flat["imgout/bias"] = np.asarray([-2.0, 0.0, 2.0], np.float32)

    outputs = module.apply(_nested_params(flat), features)

    z_expected = jax.nn.sigmoid(jnp.asarray([-2.0, 0.0], jnp.float32))
    a_expected = jax.nn.sigmoid(jnp.asarray([2.0], jnp.float32))
    np.testing.assert_allclose(outputs["z_image"].pred()[0, 0, 0], z_expected)
    np.testing.assert_allclose(outputs["a_image"].pred()[0, 0, 0], a_expected)


@pytest.mark.parametrize(
    ("case", "space", "config"),
    (
        (
            "binary_scalar",
            TensorSpace((), "bool", classes=2),
            HeadConfig(layers=1, units=6, output="binary", bins=None),
        ),
        (
            "binary_vector",
            TensorSpace((2,), "bool", classes=2),
            HeadConfig(layers=1, units=6, output="binary", bins=None),
        ),
        (
            "categorical_scalar",
            TensorSpace((), "int32", classes=4),
            HeadConfig(layers=1, units=6, output="categorical", bins=None),
        ),
        (
            "categorical_vector",
            TensorSpace((2,), "int32", classes=3),
            HeadConfig(layers=1, units=6, output="categorical", bins=None),
        ),
        (
            "onehot_scalar",
            TensorSpace((), "int32", classes=3),
            PolicyConfig(layers=1, units=6, discrete="onehot"),
        ),
        (
            "onehot_vector",
            TensorSpace((2,), "int32", classes=3),
            PolicyConfig(layers=1, units=6, discrete="onehot"),
        ),
        (
            "mse_scalar",
            TensorSpace((), "float32"),
            HeadConfig(layers=1, units=6, output="mse", bins=None),
        ),
        (
            "mse_vector",
            TensorSpace((2,), "float32"),
            HeadConfig(layers=1, units=6, output="mse", bins=None),
        ),
        (
            "symlog_mse_scalar",
            TensorSpace((), "float32"),
            HeadConfig(layers=1, units=6, output="symlog_mse", bins=None),
        ),
        (
            "symlog_mse_vector",
            TensorSpace((2,), "float32"),
            HeadConfig(layers=1, units=6, output="symlog_mse", bins=None),
        ),
        (
            "symexp_twohot_scalar",
            TensorSpace((), "float32"),
            HeadConfig(layers=1, units=6, output="symexp_twohot", bins=7),
        ),
        (
            "symexp_twohot_vector",
            TensorSpace((2,), "float32"),
            HeadConfig(layers=1, units=6, output="symexp_twohot", bins=7),
        ),
        (
            "bounded_normal_scalar",
            TensorSpace((), "float32"),
            PolicyConfig(layers=1, units=6),
        ),
        (
            "bounded_normal_vector",
            TensorSpace((2,), "float32"),
            PolicyConfig(layers=1, units=6),
        ),
    ),
)
def test_all_head_families_scalar_and_vector_match_source_executed_cases(
    official_case,
    case: str,
    space: TensorSpace,
    config: HeadConfig | PolicyConfig,
) -> None:
    manifest, arrays = official_case
    prefix = f"head_family.{case}"
    inputs = jnp.asarray(arrays[f"{prefix}.input"])
    module = MLPHead(
        space,
        config,
        compute_dtype=_compute_dtype(manifest.dtype),
    )
    initialized = module.init(jax.random.PRNGKey(0), inputs)
    params = _translate_case(arrays, prefix, initialized)

    output = module.apply(params, inputs)

    _assert_source_equal(output.pred(), arrays[f"{prefix}.output"])
    raw_output = output.output if hasattr(output, "output") else output
    if f"{prefix}.minent" in arrays:
        _assert_source_equal(raw_output.minent, arrays[f"{prefix}.minent"])
        _assert_source_equal(raw_output.maxent, arrays[f"{prefix}.maxent"])


def test_invalid_head_pairs_and_nonuniform_classes_are_source_executed(
    official_case,
) -> None:
    _, arrays = official_case
    invalid_cases = (
        "binary",
        "bounded_normal",
        "categorical",
        "mse",
        "onehot",
        "symlog_mse",
        "symexp_twohot",
        "categorical_nonuniform",
        "onehot_nonuniform",
    )
    assert all(
        arrays[f"head_invalid.{case}.rejected"].item() == 1 for case in invalid_cases
    )


def test_rank_partition_and_nonlexicographic_decoder_are_source_executed(
    official_case,
) -> None:
    manifest, arrays = official_case
    assert arrays["partition.encoder_rank3_float_rejected"].item() == 1
    assert arrays["partition.decoder_rank3_float_is_image"].item() == 1

    spaces = {
        "z_image": TensorSpace((32, 32, 2), "uint8"),
        "a_image": TensorSpace((32, 32, 1), "uint8"),
    }
    features = {
        "deter": jnp.asarray(arrays["decoder_order.deter"]),
        "stoch": jnp.asarray(arrays["decoder_order.stoch"]),
    }
    module = DictDecoder(
        spaces,
        _decoder_config(manifest.profile),
        compute_dtype=_compute_dtype(manifest.dtype),
    )
    initialized = module.init(jax.random.PRNGKey(0), features)
    params = _translate_case(arrays, "decoder_order", initialized)
    outputs = module.apply(params, features)

    for key in spaces:
        _assert_source_equal(outputs[key].pred(), arrays[f"decoder_order.pred.{key}"])
        target = jnp.asarray(arrays[f"decoder_order.target.{key}"])
        _assert_source_equal(
            outputs[key].loss(target),
            arrays[f"decoder_order.loss.{key}"],
        )


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
