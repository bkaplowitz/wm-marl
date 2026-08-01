from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from world_marl.dreamarl.representation_diagnostics import (
    Control,
    Intervention,
    adapter_residual,
    apply_control,
    build_source,
    categorical_kl,
    init_adapter,
    intervention_prediction,
    leave_one_out_mean,
)


def _tensors(batch=3, time=4, agents=3):
    stoch = jnp.arange(batch * time * agents * 6, dtype=jnp.float32).reshape(
        batch, time, agents, 2, 3
    )
    actions = jnp.arange(batch * time * agents * 2, dtype=jnp.float32).reshape(
        batch, time, agents, 2
    )
    pair = jnp.concatenate([stoch.reshape(batch, time, agents, -1), actions], -1)
    return {
        "stoch": stoch,
        "pair": pair,
        "pred_token": jnp.ones((batch, time, agents, 5)),
        "target_token": jnp.ones((batch, time, agents, 5)),
        "prior_logit": jnp.ones((batch, time, agents, 2, 3)),
        "post_logit": jnp.ones((batch, time, agents, 2, 3)),
        "reset": jnp.zeros((batch, time, agents), bool),
    }


def test_leave_one_out_mean_excludes_focal_agent():
    values = jnp.array([[[[1.0], [3.0], [8.0]]]])
    actual = leave_one_out_mean(values)
    np.testing.assert_allclose(actual[..., 0], [[[5.5, 4.5, 2.0]]])


def test_leave_one_out_singleton_is_zero():
    values = jnp.ones((2, 3, 1, 4))
    np.testing.assert_array_equal(leave_one_out_mean(values), 0)


def test_leave_one_out_is_permutation_equivariant():
    values = jax.random.normal(jax.random.key(3), (2, 4, 5, 7))
    permutation = jnp.array([2, 4, 1, 0, 3])
    expected = leave_one_out_mean(values)[:, :, permutation]
    actual = leave_one_out_mean(values[:, :, permutation])
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_paired_source_preserves_state_action_correspondence():
    tensors = _tensors()
    latent_width = np.prod(tensors["stoch"].shape[-2:])
    tensors["pair"] = tensors["pair"].at[..., :latent_width].add(1000)
    source = build_source(tensors, Intervention.PAIRED_PREDICTOR)
    np.testing.assert_array_equal(
        source[..., :latent_width],
        tensors["pair"][..., :latent_width],
    )
    np.testing.assert_array_equal(
        source[..., latent_width:], tensors["pair"][..., latent_width:]
    )


def test_predictor_sources_have_parameter_matched_widths():
    tensors = _tensors()
    widths = {
        build_source(tensors, intervention).shape[-1]
        for intervention in (
            Intervention.ACTIONS_PREDICTOR,
            Intervention.LATENTS_PREDICTOR,
            Intervention.PAIRED_PREDICTOR,
            Intervention.PAIRED_SHUFFLED_PREDICTOR,
        )
    }
    assert widths == {tensors["pair"].shape[-1]}


def test_zero_initialized_adapter_is_exact_baseline():
    tensors = _tensors()
    source = build_source(tensors, Intervention.PAIRED_PREDICTOR)
    params = init_adapter(jax.random.key(0), source.shape[-1], 5, 8)
    prediction, prior = intervention_prediction(
        tensors, Intervention.PAIRED_PREDICTOR, params
    )
    np.testing.assert_array_equal(prediction, tensors["pred_token"])
    np.testing.assert_array_equal(prior, tensors["prior_logit"])


def test_recurrent_adapter_is_causal():
    source = jax.random.normal(jax.random.key(1), (2, 5, 3, 4))
    params = init_adapter(jax.random.key(2), 4, 6, 8, recurrent=True)
    params = {**params, "output_kernel": jnp.ones((8, 6))}
    changed = source.at[:, 3:].add(50)
    original_output = adapter_residual(params, source, recurrent=True)
    changed_output = adapter_residual(params, changed, recurrent=True)
    np.testing.assert_allclose(
        original_output[:, :3], changed_output[:, :3], atol=1e-6
    )


def test_adapter_is_permutation_equivariant():
    source = jax.random.normal(jax.random.key(5), (2, 4, 5, 7))
    params = init_adapter(jax.random.key(6), 7, 3, 8)
    params = {**params, "output_kernel": jnp.ones((8, 3))}
    permutation = jnp.array([3, 1, 4, 0, 2])
    expected = adapter_residual(params, source)[:, :, permutation]
    actual = adapter_residual(params, source[:, :, permutation])
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_controls_keep_shape_and_null_information():
    values = jnp.arange(3 * 2 * 4 * 5).reshape(3, 2, 4, 5)
    key = jax.random.key(4)
    for control in Control:
        actual = apply_control(values, control, key)
        assert actual.shape == values.shape
    np.testing.assert_array_equal(
        apply_control(values, Control.NULL, key), jnp.zeros_like(values)
    )


def test_raw_kl_is_zero_for_identical_logits():
    logits = jax.random.normal(jax.random.key(9), (2, 3, 4, 5))
    np.testing.assert_allclose(categorical_kl(logits, logits), 0, atol=1e-6)
