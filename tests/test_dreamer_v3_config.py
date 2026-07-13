from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

import world_marl.dreamer_v3_baseline as dreamer_v3
from world_marl.dreamer_v3_baseline.config import (
    DreamerProfile,
    DreamerV3Config,
    ModelSize,
    NetworkSize,
    ObservationMode,
    OptimizerConfig,
    RSSMConfig,
    resolve_dreamer_config,
)


CANONICAL_DIGESTS = {
    (DreamerProfile.PAPER, ObservationMode.VISION): (
        "0f7b619eeb24d87d2d20b3331cd0a4229d4e322f007af52a85f1eb9028af4ca3"
    ),
    (DreamerProfile.PAPER, ObservationMode.PROPRIO): (
        "6e1b19b9d058ab579980dad3ed8d8caa16e805f0633bb623611d12216c5987c2"
    ),
    (DreamerProfile.UPSTREAM_CURRENT, ObservationMode.VISION): (
        "8973a4665e54be56322a158ddceea785b19850806bfcdb65743a310fbed4b94c"
    ),
    (DreamerProfile.UPSTREAM_CURRENT, ObservationMode.PROPRIO): (
        "ce029c8275701e2a692133b1d4bd1cdabe16144fc5864309d59fe9fdbae69908"
    ),
}


CANONICAL_MUTATIONS = (
    (
        "profile",
        lambda config: replace(
            config,
            profile=(
                DreamerProfile.UPSTREAM_CURRENT
                if config.profile is DreamerProfile.PAPER
                else DreamerProfile.PAPER
            ),
        ),
    ),
    (
        "observation_mode",
        lambda config: replace(
            config,
            observation_mode=(
                ObservationMode.PROPRIO
                if config.observation_mode is ObservationMode.VISION
                else ObservationMode.VISION
            ),
        ),
    ),
    (
        "model_size",
        lambda config: replace(
            config,
            model_size=(
                ModelSize.M1 if config.model_size is ModelSize.M200 else ModelSize.M200
            ),
        ),
    ),
    (
        "network",
        lambda config: replace(config, network=ModelSize.M100.resolve()),
    ),
    (
        "rssm",
        lambda config: replace(
            config,
            rssm=replace(config.rssm, free_nats=2.0),
        ),
    ),
    (
        "encoder",
        lambda config: replace(
            config,
            encoder=replace(config.encoder, layers=2),
        ),
    ),
    (
        "decoder",
        lambda config: replace(
            config,
            decoder=replace(config.decoder, image_output="normal"),
        ),
    ),
    (
        "reward_head",
        lambda config: replace(
            config,
            reward_head=replace(config.reward_head, output_scale=0.1),
        ),
    ),
    (
        "continue_head",
        lambda config: replace(
            config,
            continue_head=replace(config.continue_head, activation="relu"),
        ),
    ),
    (
        "policy",
        lambda config: replace(
            config,
            policy=replace(config.policy, min_std=0.2),
        ),
    ),
    (
        "value_head",
        lambda config: replace(
            config,
            value_head=replace(config.value_head, bins=127),
        ),
    ),
    (
        "optimizer",
        lambda config: replace(
            config,
            optimizer=replace(config.optimizer, learning_rate=1e-4),
        ),
    ),
    (
        "replay",
        lambda config: replace(
            config,
            replay=replace(config.replay, online=False),
        ),
    ),
    (
        "run",
        lambda config: replace(
            config,
            run=replace(config.run, envs=8),
        ),
    ),
    (
        "loss_scales",
        lambda config: replace(
            config,
            loss_scales=replace(config.loss_scales, rec=2.0),
        ),
    ),
    (
        "imagination",
        lambda config: replace(
            config,
            imagination=replace(config.imagination, length=16),
        ),
    ),
    (
        "slow_value",
        lambda config: replace(
            config,
            slow_value=replace(config.slow_value, every=2),
        ),
    ),
    (
        "return_normalizer",
        lambda config: replace(
            config,
            return_normalizer=replace(config.return_normalizer, rate=0.02),
        ),
    ),
    (
        "value_normalizer",
        lambda config: replace(
            config,
            value_normalizer=replace(config.value_normalizer, rate=0.02),
        ),
    ),
    (
        "advantage_normalizer",
        lambda config: replace(
            config,
            advantage_normalizer=replace(config.advantage_normalizer, rate=0.02),
        ),
    ),
)


@pytest.mark.parametrize(
    ("model_size", "expected"),
    [
        (ModelSize.M1, NetworkSize(model_dim=64, deter=512, depth=4, classes=4)),
        (
            ModelSize.M12,
            NetworkSize(model_dim=256, deter=2048, depth=16, classes=16),
        ),
        (
            ModelSize.M25,
            NetworkSize(model_dim=384, deter=3072, depth=24, classes=24),
        ),
        (
            ModelSize.M50,
            NetworkSize(model_dim=512, deter=4096, depth=32, classes=32),
        ),
        (
            ModelSize.M100,
            NetworkSize(model_dim=768, deter=6144, depth=48, classes=48),
        ),
        (
            ModelSize.M200,
            NetworkSize(model_dim=1024, deter=8192, depth=64, classes=64),
        ),
        (
            ModelSize.M400,
            NetworkSize(model_dim=1536, deter=12288, depth=96, classes=96),
        ),
    ],
)
def test_model_size_table_matches_official_profiles(
    model_size: ModelSize,
    expected: NetworkSize,
) -> None:
    assert model_size.resolve() == expected
    assert expected.deter == 8 * expected.model_dim
    assert expected.depth == expected.model_dim // 16
    assert expected.classes == expected.model_dim // 16


@pytest.mark.parametrize(
    (
        "profile",
        "mode",
        "model_size",
        "steps",
        "ratio",
        "beta2",
        "strided",
    ),
    [
        (
            DreamerProfile.PAPER,
            ObservationMode.VISION,
            ModelSize.M200,
            1_000_000,
            256.0,
            0.99,
            True,
        ),
        (
            DreamerProfile.PAPER,
            ObservationMode.PROPRIO,
            ModelSize.M1,
            1_000_000,
            1024.0,
            0.99,
            True,
        ),
        (
            DreamerProfile.UPSTREAM_CURRENT,
            ObservationMode.VISION,
            ModelSize.M200,
            1_100_000,
            256.0,
            0.999,
            False,
        ),
        (
            DreamerProfile.UPSTREAM_CURRENT,
            ObservationMode.PROPRIO,
            ModelSize.M1,
            1_100_000,
            1024.0,
            0.999,
            False,
        ),
    ],
)
def test_exact_profile_and_dmc_mode_snapshots(
    profile: DreamerProfile,
    mode: ObservationMode,
    model_size: ModelSize,
    steps: int,
    ratio: float,
    beta2: float,
    strided: bool,
) -> None:
    config = resolve_dreamer_config(profile=profile, observation_mode=mode)
    network = model_size.resolve()

    assert config.profile is profile
    assert config.observation_mode is mode
    assert config.model_size is model_size
    assert config.network == network
    assert config.rssm == RSSMConfig(
        deter=network.deter,
        hidden=network.model_dim,
        stoch=32,
        classes=network.classes,
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
    )
    assert config.encoder.depth == network.depth
    assert config.encoder.multipliers == (2, 3, 4, 4)
    assert config.encoder.layers == 3
    assert config.encoder.units == network.model_dim
    assert config.encoder.activation == "silu"
    assert config.encoder.normalization == "rms"
    assert config.encoder.initializer == "trunc_normal_in"
    assert config.encoder.symlog is True
    assert config.encoder.outer is False
    assert config.encoder.kernel == 5
    assert config.encoder.strided is strided
    assert config.decoder.depth == network.depth
    assert config.decoder.multipliers == (2, 3, 4, 4)
    assert config.decoder.layers == 3
    assert config.decoder.units == network.model_dim
    assert config.decoder.activation == "silu"
    assert config.decoder.normalization == "rms"
    assert config.decoder.output_scale == 1.0
    assert config.decoder.initializer == "trunc_normal_in"
    assert config.decoder.outer is False
    assert config.decoder.kernel == 5
    assert config.decoder.bias_space == 8
    assert config.decoder.strided is strided
    assert config.reward_head.layers == 1
    assert config.reward_head.units == network.model_dim
    assert config.reward_head.output == "symexp_twohot"
    assert config.reward_head.output_scale == 0.0
    assert config.reward_head.bins == 255
    assert config.continue_head.layers == 1
    assert config.continue_head.units == network.model_dim
    assert config.continue_head.output == "binary"
    assert config.continue_head.output_scale == 1.0
    assert config.continue_head.bins is None
    assert config.policy.layers == 3
    assert config.policy.units == network.model_dim
    assert config.policy.min_std == 0.1
    assert config.policy.max_std == 1.0
    assert config.policy.output_scale == 0.01
    assert config.policy.unimix == 0.01
    assert config.policy.discrete == "categorical"
    assert config.policy.continuous == "bounded_normal"
    assert config.value_head.layers == 3
    assert config.value_head.units == network.model_dim
    assert config.value_head.output == "symexp_twohot"
    assert config.value_head.output_scale == 0.0
    assert config.value_head.bins == 255
    assert config.optimizer == OptimizerConfig(beta2=beta2)
    assert config.replay.capacity == 5_000_000
    assert config.replay.chunk_size == 1024
    assert config.replay.online is True
    assert config.replay.uniform_fraction == 1.0
    assert config.replay.priority_fraction == 0.0
    assert config.replay.recency_fraction == 0.0
    assert config.replay.context == 1
    assert config.replay.sequence_length == 64
    assert config.run.steps == steps
    assert config.run.replay_ratio == ratio
    assert config.run.envs == 16
    assert config.run.eval_envs == 4
    assert config.run.eval_episodes == 1
    assert config.run.log_every == 120
    assert config.run.report_every == 300
    assert config.run.save_every == 900
    assert config.run.batch_size == 16
    assert config.run.batch_length == 64
    assert config.run.report_length == 32
    assert config.run.consecutive_train == 1
    assert config.run.consecutive_report == 1
    assert config.run.replay_context == 1
    assert config.run.action_repeat == 1
    assert config.run.image_size == (64, 64)
    assert config.run.camera == -1
    assert config.run.compute_dtype == "bfloat16"
    assert config.run.gradient_updates_per_transition == pytest.approx(
        ratio / (16 * 64)
    )
    assert config.loss_scales.as_tuple() == (
        ("rec", 1.0),
        ("rew", 1.0),
        ("con", 1.0),
        ("dyn", 1.0),
        ("rep", 0.1),
        ("policy", 1.0),
        ("value", 1.0),
        ("repval", 0.3),
    )
    assert config.imagination.length == 15
    assert config.imagination.horizon == 333
    assert config.imagination.continuation_discount is True
    assert config.imagination.lambda_ == 0.95
    assert config.imagination.actor_entropy == 3e-4
    assert config.imagination.ac_grads is False
    assert config.imagination.reward_grad is True
    assert config.imagination.repval_loss is True
    assert config.imagination.repval_grad is True
    assert config.slow_value.rate == 0.02
    assert config.slow_value.every == 1
    assert config.return_normalizer.rate == 0.01
    assert config.return_normalizer.limit == 1.0
    assert config.return_normalizer.low_percentile == 5.0
    assert config.return_normalizer.high_percentile == 95.0
    assert config.return_normalizer.debias is False
    config.validate()


def test_default_profile_is_independent_paper_vision_configuration() -> None:
    default = resolve_dreamer_config()
    paper = resolve_dreamer_config(DreamerProfile.PAPER, ObservationMode.VISION)
    current = resolve_dreamer_config(
        DreamerProfile.UPSTREAM_CURRENT,
        ObservationMode.VISION,
    )

    assert default == paper
    assert paper is not current
    assert paper.rssm is not current.rssm
    assert paper.encoder is not current.encoder
    assert paper.decoder is not current.decoder
    assert paper.optimizer is not current.optimizer
    assert paper.run is not current.run
    assert paper.encoder.strided is True
    assert current.encoder.strided is False


def test_direct_legacy_config_preserves_existing_small_baseline_dimensions() -> None:
    config = DreamerV3Config(action_dim=4, observation_shape=(8, 8, 3))

    assert config.rssm.deterministic_size == 128
    assert config.rssm.stochastic_size == 16
    assert config.rssm.discrete_classes == 16
    assert config.rssm.hidden_size == 256
    assert config.encoder.embedding_dim == 64
    assert config.encoder.hidden_dims == (128, 128)
    assert config.reward_head.hidden_dims == (128, 128)
    assert config.continue_head.hidden_dims == (128, 128)
    assert config.dynamics_kl_scale == 0.5
    assert config.representation_kl_scale == 0.1


@pytest.mark.parametrize(
    ("profile", "mode"),
    CANONICAL_DIGESTS,
)
def test_canonical_hashes_are_pinned_to_all_authority_snapshots(
    profile: DreamerProfile,
    mode: ObservationMode,
) -> None:
    config = resolve_dreamer_config(profile, mode)

    assert config.canonical_hash() == CANONICAL_DIGESTS[(profile, mode)]
    assert (
        resolve_dreamer_config(profile, mode).canonical_hash()
        == (CANONICAL_DIGESTS[(profile, mode)])
    )


def test_canonical_configuration_is_immutable() -> None:
    config = resolve_dreamer_config()

    with pytest.raises(FrozenInstanceError):
        config.profile = DreamerProfile.UPSTREAM_CURRENT  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        config.encoder.depth = 1  # type: ignore[misc]


@pytest.mark.parametrize(("profile", "mode"), CANONICAL_DIGESTS)
@pytest.mark.parametrize(
    ("component", "mutate"),
    CANONICAL_MUTATIONS,
    ids=[name for name, _ in CANONICAL_MUTATIONS],
)
def test_profile_labels_reject_mutation_across_the_full_canonical_tree(
    profile: DreamerProfile,
    mode: ObservationMode,
    component: str,
    mutate,
) -> None:
    config = resolve_dreamer_config(profile, mode)

    with pytest.raises(ValueError, match="canonical|match|requires"):
        mutate(config)


def test_rssm_config_supports_dataclasses_replace_before_profile_rejection() -> None:
    config = resolve_dreamer_config()

    mutated_rssm = replace(config.rssm, free_nats=2.0)

    assert mutated_rssm.free_nats == 2.0
    with pytest.raises(ValueError, match="canonical"):
        replace(config, rssm=mutated_rssm)


@pytest.mark.parametrize(("profile", "mode"), CANONICAL_DIGESTS)
@pytest.mark.parametrize(
    ("mutation_name", "mutate"),
    [
        (
            "float_to_int",
            lambda config: replace(
                config,
                run=replace(
                    config.run,
                    replay_ratio=int(config.run.replay_ratio),
                ),
            ),
        ),
        (
            "positive_to_negative_zero",
            lambda config: replace(
                config,
                replay=replace(config.replay, priority_fraction=-0.0),
            ),
        ),
    ],
)
def test_canonical_profile_validation_is_lossless_for_numeric_representation(
    profile: DreamerProfile,
    mode: ObservationMode,
    mutation_name: str,
    mutate,
) -> None:
    config = resolve_dreamer_config(profile, mode)

    assert config.canonical_hash() == CANONICAL_DIGESTS[(profile, mode)]
    with pytest.raises(ValueError, match="canonical authority hash"):
        mutate(config)


def test_legacy_rssm_supports_lossless_dataclasses_replace() -> None:
    legacy = DreamerV3Config(
        action_dim=4,
        observation_shape=(8, 8, 3),
    ).rssm

    unchanged = replace(legacy)
    updated = replace(legacy, free_nats=2.0)

    assert unchanged == legacy
    assert unchanged.stoch == 16
    assert updated.stoch == 16
    assert updated.free_nats == 2.0
    updated.validate(canonical=False)


def test_task_one_interfaces_are_exported_from_the_package_boundary() -> None:
    expected = {
        "DecoderConfig",
        "DreamerProfile",
        "DreamerV3Config",
        "EncoderConfig",
        "HeadConfig",
        "ImaginationConfig",
        "LossScaleConfig",
        "ModelSize",
        "NetworkSize",
        "NormalizerConfig",
        "ObservationMode",
        "OptimizerConfig",
        "OracleHarness",
        "OracleManifest",
        "OracleSourceSpec",
        "ParameterMapping",
        "ParameterTranslator",
        "PolicyConfig",
        "RSSMConfig",
        "ReplayConfig",
        "RunConfig",
        "SlowValueConfig",
        "TensorSpec",
        "resolve_dreamer_config",
    }

    assert expected <= set(dreamer_v3.__all__)
    assert all(getattr(dreamer_v3, name) is not None for name in expected)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RSSMConfig(stoch=31),
        lambda: RSSMConfig(blocks=4),
        lambda: RSSMConfig(deter=8190),
        lambda: RSSMConfig(classes=0),
        lambda: RSSMConfig(unimix=1.0),
        lambda: RSSMConfig(free_nats=-1.0),
    ],
)
def test_invalid_rssm_combinations_fail_before_initialization(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def test_noncanonical_dmc_model_sizes_and_profile_patches_are_rejected() -> None:
    with pytest.raises(ValueError, match="vision.*200m"):
        resolve_dreamer_config(
            DreamerProfile.PAPER,
            ObservationMode.VISION,
            ModelSize.M1,
        )
    with pytest.raises(ValueError, match="proprio.*1m"):
        resolve_dreamer_config(
            DreamerProfile.PAPER,
            ObservationMode.PROPRIO,
            ModelSize.M200,
        )

    paper = resolve_dreamer_config()
    with pytest.raises(ValueError, match="beta2"):
        replace(paper, optimizer=replace(paper.optimizer, beta2=0.999))
