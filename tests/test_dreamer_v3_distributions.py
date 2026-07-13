from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import world_marl.dreamer_v3_baseline as dreamer_v3
import world_marl.dreamer_v3_baseline.oracle as dreamer_oracle
from world_marl.dreamer_v3_baseline.config import (
    DreamerProfile,
    ObservationMode,
)
from world_marl.dreamer_v3_baseline.distributions import (
    AggregateOutput,
    BinaryOutput,
    CategoricalOutput,
    MSEOutput,
    NormalOutput,
    OneHotOutput,
    TwoHotOutput,
    symexp,
    symlog,
)
from world_marl.dreamer_v3_baseline.oracle import (
    DISTRIBUTIONS_SOURCE_SPEC,
    OracleHarness,
    OracleManifest,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "dreamer_v3"
OFFICIAL_CHECKOUT = Path(
    os.environ.get(
        "DREAMERV3_ORACLE_CHECKOUT",
        "/private/tmp/danijar-dreamerv3-20260713",
    )
)
SOURCE_HASHES = {
    "embodied/jax/heads.py": (
        "437641cde21e7f9e3f69b88ad8f6b7e7c22e54eec8c5b19eef6127afde1a9b3f"
    ),
    "embodied/jax/nets.py": (
        "9a1c0c71ad7d3596572a44416e78434f777d8f4dbcbe8ca0dd6b86bb8246392c"
    ),
    "embodied/jax/outs.py": (
        "7e80691f175c71be614f089023cce3a809e0d026c6d5ce89bf566d5f11eb3ed0"
    ),
}


def _assert_scalar(actual, expected) -> None:
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)


def _assert_f32(actual, expected) -> None:
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.fixture(
    params=(DreamerProfile.PAPER, DreamerProfile.UPSTREAM_CURRENT),
    ids=lambda profile: profile.value,
)
def official_case(request) -> tuple[OracleManifest, dict[str, np.ndarray]]:
    profile = request.param
    stem = f"{profile.value}-proprio-distributions"
    fixture_path = FIXTURE_DIR / f"{stem}.npz"
    manifest_path = FIXTURE_DIR / f"{stem}.manifest.json"
    manifest = OracleManifest.load(manifest_path, fixture_path=fixture_path)
    with np.load(fixture_path, allow_pickle=False) as fixture:
        arrays = {name: fixture[name] for name in fixture.files}
    return manifest, arrays


def test_distribution_oracles_pin_both_exact_authority_revisions(
    official_case,
) -> None:
    manifest, _ = official_case

    assert manifest.source_spec == DISTRIBUTIONS_SOURCE_SPEC.name
    assert dict(manifest.official_file_hashes) == SOURCE_HASHES
    assert manifest.observation_mode is ObservationMode.PROPRIO
    assert manifest.generator_request is not None
    request = json.loads(manifest.generator_request)
    assert request["official_commit"] == manifest.official_commit
    assert request["profile"] == manifest.profile.value
    assert request["source_spec"] == DISTRIBUTIONS_SOURCE_SPEC.name


def test_distribution_oracle_command_and_stdin_replay_exact_official_arrays(
    official_case,
) -> None:
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


def test_distribution_oracles_persist_supplied_categorical_noise_cases(
    official_case,
) -> None:
    _, arrays = official_case
    required = {
        "categorical.supplied_noise",
        "categorical.supplied_sample",
        "onehot.effective_logits",
        "onehot.supplied_noise",
        "onehot.supplied_sample",
        "onehot.supplied_sample_grad",
    }

    assert required <= set(arrays)
    assert arrays["categorical.supplied_noise"].dtype == np.float32
    assert arrays["categorical.supplied_noise"].shape == (3, 2, 4)
    assert arrays["categorical.supplied_sample"].dtype == np.int32
    assert arrays["categorical.supplied_sample"].shape == (3, 2)
    assert arrays["onehot.supplied_noise"].dtype == np.float32
    assert arrays["onehot.supplied_noise"].shape == (2, 2, 4)
    assert arrays["onehot.supplied_sample"].dtype == np.float32
    assert arrays["onehot.supplied_sample"].shape == (2, 2, 4)
    assert arrays["onehot.supplied_sample_grad"].shape == (2, 4)


def test_distribution_oracle_writer_is_byte_deterministic(
    tmp_path: Path,
) -> None:
    if not (OFFICIAL_CHECKOUT / ".git").exists():
        pytest.skip("explicit DreamerV3 oracle checkout is unavailable")
    first = OracleHarness(OFFICIAL_CHECKOUT, tmp_path / "first")
    second = OracleHarness(OFFICIAL_CHECKOUT, tmp_path / "second")

    first_fixture, first_manifest = first.run_distributions_case(
        DreamerProfile.PAPER,
        ObservationMode.PROPRIO,
    )
    second_fixture, second_manifest = second.run_distributions_case(
        DreamerProfile.PAPER,
        ObservationMode.PROPRIO,
    )

    assert first_fixture.read_bytes() == second_fixture.read_bytes()
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    assert first.last_worker_pid is not None
    assert second.last_worker_pid is not None


def test_symlog_and_symexp_match_official_extreme_scalar_oracle(
    official_case,
) -> None:
    _, arrays = official_case
    values = jnp.asarray(arrays["scalar.input"])

    _assert_scalar(symlog(values), arrays["scalar.symlog"])
    _assert_scalar(symexp(symlog(values)), arrays["scalar.roundtrip"])


def test_mse_is_unreduced_square_without_half_and_stops_target_gradient(
    official_case,
) -> None:
    _, arrays = official_case
    mean = jnp.asarray(arrays["mse.mean"])
    target = jnp.asarray(arrays["mse.target"])
    output = MSEOutput(mean)

    _assert_f32(output.pred(), arrays["mse.pred"])
    _assert_f32(output.loss(target), arrays["mse.loss"])
    assert output.loss(target).shape == mean.shape
    _assert_f32(
        jax.grad(lambda value: MSEOutput(value).loss(target).sum())(mean),
        arrays["mse.grad_mean"],
    )
    _assert_f32(
        jax.grad(lambda value: output.loss(value).sum())(target),
        arrays["mse.grad_target"],
    )
    np.testing.assert_array_equal(arrays["mse.grad_target"], 0.0)


def test_mse_undefined_probability_protocol_methods_fail_explicitly(
    official_case,
) -> None:
    _, arrays = official_case
    output = MSEOutput(jnp.asarray(arrays["mse.mean"]))
    target = jnp.asarray(arrays["mse.target"])

    with pytest.raises(NotImplementedError):
        output.sample(jnp.asarray(arrays["mse.seed"]))
    with pytest.raises(NotImplementedError):
        output.logp(target)
    with pytest.raises(NotImplementedError):
        output.prob(target)
    with pytest.raises(NotImplementedError):
        output.entropy()
    with pytest.raises(NotImplementedError):
        output.kl(output)


def test_normal_public_protocol_matches_official_values_and_seeded_sample(
    official_case,
) -> None:
    _, arrays = official_case
    output = NormalOutput(arrays["normal.mean"], arrays["normal.stddev"])
    other = NormalOutput(
        arrays["normal.other_mean"],
        arrays["normal.other_stddev"],
    )
    event = jnp.asarray(arrays["normal.event"])
    seed = jnp.asarray(arrays["normal.seed"])

    _assert_f32(output.pred(), arrays["normal.pred"])
    _assert_f32(output.sample(seed, shape=(2,)), arrays["normal.sample"])
    _assert_f32(output.logp(event), arrays["normal.logp"])
    _assert_f32(output.prob(event), arrays["normal.prob"])
    _assert_f32(output.entropy(), arrays["normal.entropy"])
    _assert_f32(output.kl(other), arrays["normal.kl"])
    _assert_f32(output.loss(event), arrays["normal.loss"])
    _assert_f32(
        jax.grad(lambda value: output.loss(value).sum())(event),
        arrays["normal.grad_target"],
    )
    np.testing.assert_array_equal(arrays["normal.grad_target"], 0.0)


def test_bounded_normal_matches_official_head_but_samples_plain_normal(
    official_case,
) -> None:
    _, arrays = official_case
    raw_mean = jnp.asarray(arrays["bounded.raw_mean"])
    raw_stddev = jnp.asarray(arrays["bounded.raw_stddev"])
    event = jnp.asarray(arrays["bounded.event"])
    seed = jnp.asarray(arrays["bounded.seed"])
    output = NormalOutput.bounded(raw_mean, raw_stddev, 0.1, 1.0)

    _assert_f32(output.mean, arrays["bounded.mean"])
    _assert_f32(output.stddev, arrays["bounded.stddev"])
    _assert_f32(output.pred(), arrays["bounded.pred"])
    _assert_f32(output.sample(seed), arrays["bounded.sample"])
    _assert_f32(output.logp(event), arrays["bounded.logp"])
    _assert_f32(output.prob(event), arrays["bounded.prob"])
    _assert_f32(output.entropy(), arrays["bounded.entropy"])
    _assert_f32(output.loss(event), arrays["bounded.loss"])
    _assert_f32(
        jax.grad(
            lambda value: (
                NormalOutput.bounded(value, raw_stddev, 0.1, 1.0).loss(event).sum()
            )
        )(raw_mean),
        arrays["bounded.grad_raw_mean"],
    )
    _assert_f32(
        jax.grad(
            lambda value: (
                NormalOutput.bounded(raw_mean, value, 0.1, 1.0).loss(event).sum()
            )
        )(raw_stddev),
        arrays["bounded.grad_raw_stddev"],
    )
    assert np.any(np.abs(np.asarray(output.sample(seed))) > 1.0)


def test_binary_uses_stable_extreme_log_probabilities_and_seeded_samples(
    official_case,
) -> None:
    _, arrays = official_case
    output = BinaryOutput(arrays["binary.logit"])
    event = jnp.asarray(arrays["binary.event"])
    seed = jnp.asarray(arrays["binary.seed"])

    np.testing.assert_array_equal(output.pred(), arrays["binary.pred"])
    np.testing.assert_array_equal(
        output.sample(seed, shape=(3,)), arrays["binary.sample"]
    )
    _assert_f32(output.logp(event), arrays["binary.logp"])
    _assert_f32(output.prob(event), arrays["binary.prob"])
    _assert_f32(output.loss(event), arrays["binary.loss"])
    assert np.all(np.isfinite(np.asarray(output.logp(event))))
    with pytest.raises(NotImplementedError):
        output.entropy()
    with pytest.raises(NotImplementedError):
        output.kl(output)


def test_categorical_unimix_and_every_public_method_match_official(
    official_case,
) -> None:
    _, arrays = official_case
    output = CategoricalOutput(arrays["categorical.logits"], unimix=0.01)
    other = CategoricalOutput(arrays["categorical.other_logits"], unimix=0.01)
    event = jnp.asarray(arrays["categorical.event"])
    seed = jnp.asarray(arrays["categorical.seed"])

    _assert_f32(output.logits, arrays["categorical.effective_logits"])
    _assert_f32(jax.nn.softmax(output.logits), arrays["categorical.probs"])
    np.testing.assert_array_equal(output.pred(), arrays["categorical.pred"])
    np.testing.assert_array_equal(
        output.sample(seed, shape=(3,)), arrays["categorical.sample"]
    )
    _assert_f32(output.logp(event), arrays["categorical.logp"])
    _assert_f32(output.prob(event), arrays["categorical.prob"])
    _assert_f32(output.entropy(), arrays["categorical.entropy"])
    _assert_f32(output.kl(other), arrays["categorical.kl"])
    _assert_f32(output.loss(event), arrays["categorical.loss"])
    np.testing.assert_allclose(
        np.asarray(jax.nn.softmax(output.logits)).sum(-1),
        1.0,
        rtol=0.0,
        atol=1e-7,
    )


def test_categorical_sample_matches_official_with_supplied_gumbel_noise(
    official_case,
) -> None:
    _, arrays = official_case
    output = CategoricalOutput(arrays["categorical.logits"], unimix=0.01)

    with dreamer_oracle._supplied_categorical_noise_scope(
        jax.random,
        expected_logits=arrays["categorical.effective_logits"],
        expected_output_shape=(3, 2),
        noise=arrays["categorical.supplied_noise"],
    ):
        sample = output.sample(jnp.asarray(arrays["categorical.seed"]), shape=(3,))

    np.testing.assert_array_equal(sample, arrays["categorical.supplied_sample"])


def test_supplied_categorical_noise_rejects_host_float64_before_jax_coercion() -> None:
    logits = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    noise = np.zeros((3, 3), dtype=np.float64)
    entered = False

    with pytest.raises(ValueError, match=r"noise dtype.*float64.*float32"):
        with dreamer_oracle._supplied_categorical_noise_scope(
            jax.random,
            expected_logits=logits,
            expected_output_shape=(3,),
            noise=noise,
        ):
            entered = True

    assert not entered


def test_supplied_categorical_noise_supports_logits_without_batch_dimensions() -> None:
    logits = jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float32)
    noise = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
        ],
        dtype=np.float32,
    )

    with dreamer_oracle._supplied_categorical_noise_scope(
        jax.random,
        expected_logits=logits,
        expected_output_shape=(3,),
        noise=noise,
    ):
        sample = jax.random.categorical(
            jax.random.PRNGKey(0),
            logits,
            axis=-1,
            shape=(3,),
        )

    np.testing.assert_array_equal(sample, np.asarray([2, 0, 1], dtype=np.int32))


def test_onehot_is_hard_forward_straight_through_backward_and_matches_oracle(
    official_case,
) -> None:
    _, arrays = official_case
    logits = jnp.asarray(arrays["onehot.logits"])
    event = jnp.asarray(arrays["onehot.event"])
    seed = jnp.asarray(arrays["onehot.seed"])
    weights = jnp.asarray(arrays["onehot.sample_weights"])
    output = OneHotOutput(logits, unimix=0.01)
    other = OneHotOutput(arrays["onehot.other_logits"], unimix=0.01)

    _assert_f32(output.pred(), arrays["onehot.pred"])
    _assert_f32(output.sample(seed), arrays["onehot.sample"])
    _assert_f32(output.logp(event), arrays["onehot.logp"])
    _assert_f32(output.prob(event), arrays["onehot.prob"])
    _assert_f32(output.entropy(), arrays["onehot.entropy"])
    _assert_f32(output.kl(other), arrays["onehot.kl"])
    _assert_f32(output.loss(event), arrays["onehot.loss"])
    np.testing.assert_array_equal(np.asarray(output.sample(seed)).sum(-1), 1.0)
    np.testing.assert_array_equal(
        np.logical_or(
            np.asarray(output.sample(seed)) == 0.0,
            np.asarray(output.sample(seed)) == 1.0,
        ),
        True,
    )
    _assert_f32(
        jax.grad(
            lambda value: (
                OneHotOutput(value, unimix=0.01).sample(seed) * weights
            ).sum()
        )(logits),
        arrays["onehot.sample_grad"],
    )
    _assert_f32(
        jax.grad(lambda value: output.loss(value).sum())(event),
        arrays["onehot.grad_target"],
    )
    np.testing.assert_array_equal(arrays["onehot.grad_target"], 0.0)


def test_onehot_sample_and_straight_through_gradient_match_official_with_supplied_noise(
    official_case,
) -> None:
    _, arrays = official_case
    logits = jnp.asarray(arrays["onehot.logits"])
    seed = jnp.asarray(arrays["onehot.seed"])
    weights = jnp.asarray(arrays["onehot.sample_weights"])
    output = OneHotOutput(logits, unimix=0.01)
    scope = dreamer_oracle._supplied_categorical_noise_scope

    with scope(
        jax.random,
        expected_logits=arrays["onehot.effective_logits"],
        expected_output_shape=(2, 2),
        noise=arrays["onehot.supplied_noise"],
    ):
        sample = output.sample(seed, shape=(2,))

    np.testing.assert_array_equal(sample, arrays["onehot.supplied_sample"])
    np.testing.assert_array_equal(np.asarray(sample).sum(-1), 1.0)
    np.testing.assert_array_equal(
        np.logical_or(np.asarray(sample) == 0.0, np.asarray(sample) == 1.0),
        True,
    )

    def objective(value):
        with scope(
            jax.random,
            expected_logits=arrays["onehot.effective_logits"],
            expected_output_shape=(2, 2),
            noise=arrays["onehot.supplied_noise"],
        ):
            supplied_sample = OneHotOutput(value, unimix=0.01).sample(
                seed,
                shape=(2,),
            )
        return (supplied_sample * weights).sum()

    _assert_f32(
        jax.grad(objective)(logits),
        arrays["onehot.supplied_sample_grad"],
    )


def test_twohot_odd_bins_clamping_interpolation_order_and_gradients_match_oracle(
    official_case,
) -> None:
    _, arrays = official_case
    logits = jnp.asarray(arrays["twohot_odd.logits"])
    target = jnp.asarray(arrays["twohot_odd.target"])
    output = TwoHotOutput(logits, bins=255)

    _assert_f32(output.bins, arrays["twohot_odd.bins"])
    _assert_f32(output.pred(), arrays["twohot_odd.pred"])
    _assert_f32(output.loss(target), arrays["twohot_odd.loss"])
    _assert_f32(
        jax.grad(lambda value: TwoHotOutput(value, bins=255).loss(target).sum())(
            logits
        ),
        arrays["twohot_odd.grad_logits"],
    )
    _assert_f32(
        jax.grad(lambda value: output.loss(value).sum())(target),
        arrays["twohot_odd.grad_target"],
    )
    assert target[0] < output.bins[0]
    assert target[-1] > output.bins[-1]
    assert np.asarray(output.pred())[0] == 0.0
    np.testing.assert_array_equal(arrays["twohot_odd.grad_target"], 0.0)


def test_twohot_even_bins_use_official_symmetric_branch(
    official_case,
) -> None:
    _, arrays = official_case
    output = TwoHotOutput(arrays["twohot_even.logits"], bins=8)
    target = jnp.asarray(arrays["twohot_even.target"])

    _assert_f32(output.bins, arrays["twohot_even.bins"])
    _assert_f32(output.pred(), arrays["twohot_even.pred"])
    _assert_f32(output.loss(target), arrays["twohot_even.loss"])
    assert np.asarray(output.pred())[0] == 0.0


def test_twohot_undefined_sampling_probability_methods_fail_explicitly(
    official_case,
) -> None:
    _, arrays = official_case
    output = TwoHotOutput(arrays["twohot_even.logits"], bins=8)
    event = jnp.asarray(arrays["twohot_even.target"])

    with pytest.raises(NotImplementedError):
        output.sample(jnp.asarray(arrays["twohot_even.seed"]))
    with pytest.raises(NotImplementedError):
        output.logp(event)
    with pytest.raises(NotImplementedError):
        output.prob(event)
    with pytest.raises(NotImplementedError):
        output.entropy()
    with pytest.raises(NotImplementedError):
        output.kl(output)


def test_aggregate_reduces_exact_event_axes_with_official_prob_sum_behavior(
    official_case,
) -> None:
    _, arrays = official_case
    output = AggregateOutput(
        NormalOutput(arrays["aggregate.mean"], arrays["aggregate.stddev"]),
        dims=2,
        aggregate=jnp.mean,
    )
    other = AggregateOutput(
        NormalOutput(
            arrays["aggregate.other_mean"],
            arrays["aggregate.other_stddev"],
        ),
        dims=2,
        aggregate=jnp.mean,
    )
    event = jnp.asarray(arrays["aggregate.event"])
    seed = jnp.asarray(arrays["aggregate.seed"])

    _assert_f32(output.pred(), arrays["aggregate.pred"])
    _assert_f32(output.sample(seed), arrays["aggregate.sample"])
    _assert_f32(output.loss(event), arrays["aggregate.loss"])
    _assert_f32(output.logp(event), arrays["aggregate.logp"])
    _assert_f32(output.prob(event), arrays["aggregate.prob"])
    _assert_f32(output.entropy(), arrays["aggregate.entropy"])
    _assert_f32(output.kl(other), arrays["aggregate.kl"])
    assert not np.allclose(np.asarray(output.prob(event)), np.exp(output.logp(event)))


def test_aggregate_mse_sums_only_configured_trailing_event_axes_and_stops_target(
    official_case,
) -> None:
    _, arrays = official_case
    mean = jnp.asarray(arrays["aggregate_mse.mean"])
    target = jnp.asarray(arrays["aggregate_mse.target"])
    output = AggregateOutput(MSEOutput(mean), dims=2)

    _assert_f32(output.loss(target), arrays["aggregate_mse.loss"])
    assert output.loss(target).shape == mean.shape[:-2]
    _assert_f32(
        jax.grad(lambda value: output.loss(value).sum())(target),
        arrays["aggregate_mse.grad_target"],
    )
    np.testing.assert_array_equal(arrays["aggregate_mse.grad_target"], 0.0)


def test_task_two_interfaces_are_exported_from_package_boundary() -> None:
    expected = {
        "AggregateOutput",
        "BinaryOutput",
        "CategoricalOutput",
        "DISTRIBUTIONS_SOURCE_SPEC",
        "MSEOutput",
        "NormalOutput",
        "OneHotOutput",
        "TwoHotOutput",
        "symexp",
        "symlog",
    }

    assert expected <= set(dreamer_v3.__all__)
    assert all(getattr(dreamer_v3, name) is not None for name in expected)
