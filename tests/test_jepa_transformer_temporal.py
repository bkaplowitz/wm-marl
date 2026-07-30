from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from world_marl.jepa_transformer.temporal import (
    CausalTemporalTransformer,
    TemporalConfig,
    episode_positions,
    segment_causal_mask,
)


def _model() -> CausalTemporalTransformer:
    return CausalTemporalTransformer(
        TemporalConfig(
            pair_dim=5,
            model_dim=16,
            state_dim=12,
            num_layers=2,
            num_heads=4,
            mlp_ratio=2,
            context_length=8,
        )
    )


def test_positions_and_attention_restart_at_episode_boundaries():
    is_first = jnp.array([[True, False, False, True, False]], bool)
    np.testing.assert_array_equal(episode_positions(is_first), [[0, 1, 2, 0, 1]])
    mask = segment_causal_mask(is_first)[0]
    assert mask[2, 0]
    assert not mask[3, 2]
    assert mask[4, 3]
    assert not mask[1, 2]


def test_state_t_does_not_depend_on_current_or_future_pair():
    model = _model()
    pairs = jax.random.normal(jax.random.key(1), (2, 6, 5))
    is_first = jnp.array([[True, False, False, False, False, False]] * 2, dtype=bool)
    variables = model.init(jax.random.key(2), pairs, is_first)
    baseline = model.apply(variables, pairs, is_first)
    changed = pairs.at[:, 3:].add(100.0)
    counterfactual = model.apply(variables, changed, is_first)
    np.testing.assert_allclose(baseline[:, :4], counterfactual[:, :4], atol=1e-5)


def test_reset_erases_all_preceding_history():
    model = _model()
    suffix = jax.random.normal(jax.random.key(3), (1, 3, 5))
    prefix_a = jnp.zeros((1, 3, 5))
    prefix_b = jnp.full((1, 3, 5), 50.0)
    is_first = jnp.array([[True, False, False, True, False, False]], bool)
    pairs_a = jnp.concatenate([prefix_a, suffix], axis=1)
    pairs_b = jnp.concatenate([prefix_b, suffix], axis=1)
    variables = model.init(jax.random.key(4), pairs_a, is_first)
    states_a = model.apply(variables, pairs_a, is_first)
    states_b = model.apply(variables, pairs_b, is_first)
    np.testing.assert_allclose(states_a[:, 3:], states_b[:, 3:], atol=1e-5)


def test_recurrent_cache_matches_parallel_causal_execution():
    model = _model()
    pairs = jax.random.normal(jax.random.key(5), (2, 6, 5))
    is_first = jnp.array(
        [
            [True, False, False, True, False, False],
            [True, False, False, False, False, False],
        ],
        bool,
    )
    variables = model.init(jax.random.key(6), pairs, is_first)
    parallel = model.apply(variables, pairs, is_first)
    cache = model.initial(pairs.shape[0])
    recurrent = []
    previous = jnp.zeros_like(pairs[:, 0])
    for index in range(pairs.shape[1]):
        cache, state = model.apply(
            variables,
            cache,
            previous,
            is_first[:, index],
            method=model.step,
        )
        recurrent.append(state)
        previous = pairs[:, index]
    recurrent = jnp.stack(recurrent, axis=1)
    np.testing.assert_allclose(parallel, recurrent, atol=2e-5, rtol=2e-5)


def test_parallel_path_rejects_sequences_longer_than_cache_contract():
    model = _model()
    pairs = jnp.zeros((1, 9, 5))
    first = jnp.zeros((1, 9), bool).at[:, 0].set(True)
    with pytest.raises(ValueError, match="context_length"):
        model.init(jax.random.key(7), pairs, first)
