from __future__ import annotations

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from dreamarl.world_model import world_model_backend
from dreamarl.world_model.transformer import (
    CausalTransformer,
    ParallelTransformerDynamics,
)


ACTION_SPACE = {"action": elements.Space(np.int32, (), 0, 4)}


def _assert_close(actual, expected, *, atol=2e-5):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.dtype.kind in "biu" or expected.dtype.kind in "biu":
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(
            actual.astype(np.float32),
            expected.astype(np.float32),
            atol=atol,
            rtol=atol,
        )


def _transformer() -> CausalTransformer:
    return CausalTransformer(
        5,
        units=16,
        output=12,
        layers=2,
        heads=4,
        context=8,
        ffup=2,
        name="temporal",
    )


def _parallel_and_recurrent(pairs, resets):
    model = _transformer()
    cache = model.initial(pairs.shape[0])
    final, parallel, snapshots = model.sequence(cache, pairs, resets)
    recurrent = []
    recurrent_caches = []
    for index in range(pairs.shape[1]):
        cache, state = model.step(cache, pairs[:, index], resets[:, index])
        recurrent.append(state)
        recurrent_caches.append(cache)
    recurrent = jnp.stack(recurrent, axis=1)
    recurrent_caches = jax.tree.map(
        lambda *values: jnp.stack(values, axis=1), *recurrent_caches
    )
    return final, parallel, snapshots, recurrent, recurrent_caches


def _initialized_transformer_result(pairs, resets):
    params = nj.init(_parallel_and_recurrent)({}, pairs, resets, seed=1)
    return nj.pure(_parallel_and_recurrent)(params, pairs, resets, seed=2)[1]


def test_parallel_sequence_matches_cached_recurrent_execution() -> None:
    pairs = jax.random.normal(jax.random.key(3), (2, 6, 5))
    resets = jnp.array(
        [
            [True, False, False, True, False, False],
            [True, False, False, False, False, False],
        ],
        bool,
    )
    final, parallel, snapshots, recurrent, recurrent_caches = (
        _initialized_transformer_result(pairs, resets)
    )
    _assert_close(parallel, recurrent)
    for key in ("keys", "values", "valid", "position"):
        _assert_close(snapshots[key], recurrent_caches[key])
        _assert_close(final[key], recurrent_caches[key][:, -1])


def test_parallel_sequence_is_causal_and_reset_isolates_history() -> None:
    pairs = jax.random.normal(jax.random.key(4), (2, 6, 5))
    resets = jnp.array([[True, False, False, True, False, False]] * 2, bool)
    params = nj.init(
        lambda x: _transformer().sequence(_transformer().initial(x.shape[0]), x, resets)
    )({}, pairs, seed=5)

    def states(inputs):
        return _transformer().sequence(
            _transformer().initial(inputs.shape[0]), inputs, resets
        )[1]

    baseline = nj.pure(states)(params, pairs, seed=6)[1]
    changed_future = pairs.at[:, 4:].add(100)
    future_result = nj.pure(states)(params, changed_future, seed=6)[1]
    _assert_close(baseline[:, :4], future_result[:, :4])

    changed_prefix = pairs.at[:, :3].add(100)
    prefix_result = nj.pure(states)(params, changed_prefix, seed=6)[1]
    _assert_close(baseline[:, 3:], prefix_result[:, 3:])


def _dynamics(*, posterior_context: str = "observation") -> ParallelTransformerDynamics:
    return ParallelTransformerDynamics(
        ACTION_SPACE,
        enc_output=12,
        deter=16,
        hidden=8,
        stoch=2,
        classes=4,
        blocks=2,
        imglayers=2,
        obslayers=1,
        dynlayers=1,
        model=16,
        layers=2,
        heads=4,
        context=8,
        ffup=2,
        posterior_context=posterior_context,
        act="silu",
        norm="rms",
        name="dyn",
    )


def test_history_conditioned_posterior_uses_only_causal_history() -> None:
    model = _dynamics(posterior_context="history")
    tokens = jax.random.normal(jax.random.key(70), (2, 6, 12))
    actions = {"action": jnp.arange(12).reshape(2, 6) % 4}
    resets = jnp.array([[True, False, False, False, False, False]] * 2, bool)

    def observe(current_tokens):
        return model.observe(
            model.initial(2),
            current_tokens,
            actions,
            resets,
            training=False,
        )[2]["logit"]

    params = nj.init(observe)({}, tokens, seed=71)
    baseline = nj.pure(observe)(params, tokens, seed=72)[1]

    changed_past = tokens.at[:, 0].add(100)
    past_result = nj.pure(observe)(params, changed_past, seed=72)[1]
    assert not np.allclose(
        np.asarray(baseline[:, 1:], dtype=np.float32),
        np.asarray(past_result[:, 1:], dtype=np.float32),
    )

    changed_future = tokens.at[:, 4:].add(100)
    future_result = nj.pure(observe)(params, changed_future, seed=72)[1]
    _assert_close(baseline[:, :4], future_result[:, :4])


def test_parallel_dynamics_loss_and_gradients_are_finite() -> None:
    model = _dynamics()
    tokens = jax.random.normal(jax.random.key(10), (2, 6, 12))
    actions = {"action": jnp.arange(12).reshape(2, 6) % 4}
    resets = jnp.zeros((2, 6), bool).at[:, 0].set(True)

    def loss():
        output = model.loss(
            model.initial(2),
            tokens,
            actions,
            resets,
            training=True,
            slow_tokens=tokens,
        )
        return sum(value.mean() for value in output[2].values())

    params = nj.init(loss)({}, seed=11)

    def scalar_loss(variables):
        return nj.pure(loss)(variables, seed=12)[1]

    value = scalar_loss(params)
    gradients = jax.grad(scalar_loss)(params)
    assert np.isfinite(np.asarray(value)).all()
    assert all(
        np.isfinite(np.asarray(item)).all() for item in jax.tree.leaves(gradients)
    )


def test_world_model_backends_are_explicit_and_first_party() -> None:
    candidate = world_model_backend()
    assert candidate.name == "parallel_transformer"
    assert (
        candidate.dynamics_model("parallel_transformer") is ParallelTransformerDynamics
    )
