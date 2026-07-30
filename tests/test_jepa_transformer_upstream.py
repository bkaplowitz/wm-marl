"""Runtime tests for the Dreamer-CDP M3 overlay.

These tests are skipped in the main repository environment because the pinned
Dreamer-CDP dependencies intentionally live in a separate virtualenv. The GPU
runtime gate sets ``JEPA_TRANSFORMER_RUNTIME`` and executes this file with that
virtualenv.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest


runtime = os.environ.get("JEPA_TRANSFORMER_RUNTIME")
if not runtime:
    pytest.skip("requires generated Dreamer-CDP runtime", allow_module_level=True)
sys.path.insert(0, str(Path(runtime).resolve()))

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nj = pytest.importorskip("ninjax")

from dreamerv3.m3_rssm import CausalKVTransformer  # noqa: E402


def _assert_equal(actual, expected, *, atol=2e-5):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.dtype.kind in "biu" or expected.dtype.kind in "biu":
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(
            actual.astype(np.float32), expected.astype(np.float32), atol=atol
        )


def _small_transformer() -> CausalKVTransformer:
    return CausalKVTransformer(
        5,
        units=16,
        output=12,
        layers=2,
        heads=4,
        context=6,
        ffup=2,
        name="temporal",
    )


def _rollout(net, pairs, resets):
    cache = net.initial(pairs.shape[0])
    outputs = []
    caches = []
    for index in range(pairs.shape[1]):
        cache, deter = net.step(cache, pairs[:, index], resets[:, index])
        outputs.append(deter)
        caches.append(cache)
    return jax.tree.map(lambda *xs: jnp.stack(xs, 1), *caches), jnp.stack(
        outputs, 1
    )


def _initialized_rollout(pairs, resets):
    net = _small_transformer()
    params = nj.init(lambda x, r: _rollout(net, x, r))(
        {}, pairs, resets, seed=0
    )
    _, result = nj.pure(lambda x, r: _rollout(net, x, r))(
        params, pairs, resets, seed=1
    )
    return result


def test_reset_erases_all_prior_temporal_history():
    rng = np.random.default_rng(4)
    pairs = rng.normal(size=(2, 7, 5)).astype(np.float32)
    pairs[1, 3:] = pairs[0, 3:]
    resets = np.zeros((2, 7), bool)
    resets[:, 0] = True
    resets[:, 3] = True

    caches, outputs = _initialized_rollout(pairs, resets)

    _assert_equal(outputs[0, 3:], outputs[1, 3:])
    for key in ("keys", "values", "valid", "position"):
        _assert_equal(caches[key][0, 3:], caches[key][1, 3:])


def test_replaying_a_prefix_reconstructs_the_exact_cache():
    rng = np.random.default_rng(9)
    pairs = rng.normal(size=(1, 8, 5)).astype(np.float32)
    resets = np.zeros((1, 8), bool)
    resets[:, 0] = True
    resets[:, 5] = True
    net = _small_transformer()

    def compare(full_pairs, full_resets):
        full_cache = net.initial(1)
        snapshots = []
        for index in range(full_pairs.shape[1]):
            full_cache, _ = net.step(
                full_cache, full_pairs[:, index], full_resets[:, index]
            )
            snapshots.append(full_cache)
        replay_cache = net.initial(1)
        for index in range(full_pairs.shape[1] - 1):
            replay_cache, _ = net.step(
                replay_cache, full_pairs[:, index], full_resets[:, index]
            )
        expected = jax.tree.map(lambda *xs: jnp.stack(xs, 1), *snapshots)
        return expected, replay_cache

    params = nj.init(compare)({}, pairs, resets, seed=0)
    _, (expected, replayed) = nj.pure(compare)(params, pairs, resets, seed=1)
    for key in ("keys", "values", "valid", "position"):
        _assert_equal(replayed[key], expected[key][:, -2])


def test_future_pairs_cannot_change_an_existing_temporal_state():
    rng = np.random.default_rng(12)
    pairs = rng.normal(size=(1, 6, 5)).astype(np.float32)
    changed = pairs.copy()
    changed[:, 4:] += 100
    resets = np.zeros((1, 6), bool)
    resets[:, 0] = True

    _, outputs = _initialized_rollout(pairs, resets)
    _, changed_outputs = _initialized_rollout(changed, resets)
    _assert_equal(outputs[:, :4], changed_outputs[:, :4])
