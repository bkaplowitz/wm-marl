from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from flow_matching.models import TokenizedDiscreteDenoiser
from flow_matching.paths import (
    factorized_jump_rates,
    sample_discrete_conditional_path,
)
from flow_matching.simulate import sample_marginal_discrete_flow_model
from flow_matching.train import (
    conditioned_discrete_train_step,
    create_discrete_conditioned_train_state,
)


def test_corruption_keeps_clean_tokens_at_t_one():
    # kappa_1 = 1 => Bernoulli(1) is always True => x_t == z for every factor.
    z = jnp.asarray([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=jnp.int32)
    t = jnp.ones((2, 1))
    xt = sample_discrete_conditional_path(jax.random.PRNGKey(0), z, t, 4)
    np.testing.assert_array_equal(np.asarray(xt), np.asarray(z))


def test_corruption_is_uniform_source_at_t_zero():
    # kappa_0 = 0 => every factor is replaced by a uniform noise token, so the
    # output is independent of z and (marginally) uniform over the categories.
    num_categories = 5
    z = jnp.zeros((4096, 3), dtype=jnp.int32)
    t = jnp.zeros((4096, 1))
    xt = np.asarray(
        sample_discrete_conditional_path(jax.random.PRNGKey(1), z, t, num_categories)
    )
    assert xt.min() >= 0 and xt.max() < num_categories
    counts = np.bincount(xt.reshape(-1), minlength=num_categories)
    expected = xt.size / num_categories
    assert np.all(np.abs(counts - expected) < 0.15 * expected)


def test_mixture_path_rates_is_posterior_over_one_minus_t():
    posterior = jnp.asarray([[[0.1, 0.2, 0.7]]])
    t = jnp.asarray(0.5)
    rates = factorized_jump_rates(posterior, t)
    np.testing.assert_allclose(
        np.asarray(rates), np.asarray(posterior) / 0.5, rtol=1e-6
    )


def _toy_discrete_setup(key, *, num_factors=3, num_categories=4, batch=8):
    # A deterministic cond -> z map: cond_vars one-hot-encode the target tokens,
    # so a small denoiser can memorize it and the sampler should recover z.
    z = jax.random.randint(key, (batch, num_factors), 0, num_categories)
    cond_vars = jax.nn.one_hot(z, num_categories).reshape(batch, -1)
    model = TokenizedDiscreteDenoiser(num_categories=num_categories)
    state = create_discrete_conditioned_train_state(
        jax.random.PRNGKey(0),
        model,
        1e-2,
        num_factors=num_factors,
        cond_dim=num_factors * num_categories,
    )
    return z, cond_vars, state


def test_discrete_denoiser_learns_toy_map_and_sampler_recovers_it():
    z, cond_vars, state = _toy_discrete_setup(jax.random.PRNGKey(3))
    num_factors, num_categories = z.shape[1], 4

    rng = jax.random.PRNGKey(7)
    losses = []
    for _ in range(400):
        rng, step_key = jax.random.split(rng)
        state, loss = conditioned_discrete_train_step(
            state, step_key, z, cond_vars, num_categories
        )
        losses.append(float(loss))

    assert losses[-1] < losses[0]  # the fit actually moved the loss

    tokens = sample_marginal_discrete_flow_model(
        state.apply_fn,
        state.params,
        jax.random.PRNGKey(11),
        cond_vars,
        num_factors=num_factors,
        num_categories=num_categories,
        steps=16,
    )
    assert tokens.shape == z.shape
    assert tokens.min() >= 0 and tokens.max() < num_categories
    accuracy = float(jnp.mean((tokens == z).astype(jnp.float32)))
    assert accuracy > 0.9, accuracy


def _explicit_discrete_sample(
    apply_fn, params, key, cond_vars, *, num_factors, num_categories, steps
):
    # Plain-Python reference for the lax.scan CTMC sampler. Mirrors the exact key
    # threading (split init key first, then split a sample key each step) so any
    # divergence in the scan body shows up as a token mismatch.
    batch = cond_vars.shape[0]
    h = 1.0 / steps
    key, init_key = jax.random.split(key)
    xt = jax.random.randint(init_key, (batch, num_factors), 0, num_categories)
    step_key = key
    ts = jnp.arange(steps) / steps
    for i in range(steps):
        t = ts[i]
        step_key, sample_key = jax.random.split(step_key)
        tt = jnp.full((batch, 1), t)
        logits = apply_fn({"params": params}, xt, tt, cond_vars)
        posterior = jax.nn.softmax(logits, axis=-1)
        rates = factorized_jump_rates(posterior, t)
        current = jax.nn.one_hot(xt, num_categories)
        off_diag = h * rates * (1.0 - current)
        self_prob = 1.0 - jnp.sum(off_diag, axis=-1, keepdims=True)
        probs = off_diag + current * self_prob
        xt = jax.random.categorical(sample_key, jnp.log(probs), axis=-1)
    return xt


def test_discrete_sampler_scan_matches_python_loop():
    num_factors, num_categories, batch, steps = 3, 4, 6, 8
    _, cond_vars, state = _toy_discrete_setup(
        jax.random.PRNGKey(5), num_factors=num_factors, num_categories=num_categories
    )
    cond_vars = jax.random.normal(
        jax.random.PRNGKey(9), (batch, num_factors * num_categories)
    )
    key = jax.random.PRNGKey(2)

    scan_tokens = sample_marginal_discrete_flow_model(
        state.apply_fn,
        state.params,
        key,
        cond_vars,
        num_factors=num_factors,
        num_categories=num_categories,
        steps=steps,
    )
    loop_tokens = _explicit_discrete_sample(
        state.apply_fn,
        state.params,
        key,
        cond_vars,
        num_factors=num_factors,
        num_categories=num_categories,
        steps=steps,
    )

    np.testing.assert_array_equal(np.asarray(scan_tokens), np.asarray(loop_tokens))
