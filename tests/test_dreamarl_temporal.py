from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from world_marl.dreamarl.config import DreaMARLConfig, DynamicsConfig
from world_marl.dreamarl.temporal import CausalKVTransformer


def _module() -> CausalKVTransformer:
    return CausalKVTransformer(
        pair_dim=7,
        model_dim=16,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2,
        context_length=8,
    )


def test_config_rejects_attention_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="divisible"):
        DynamicsConfig(model_dim=15, num_heads=4)
    with pytest.raises(ValueError, match="positive"):
        DreaMARLConfig(max_agents=0, action_dim=5)


def test_recurrent_cache_matches_parallel_sequence() -> None:
    module = _module()
    key = jax.random.PRNGKey(0)
    pairs = jax.random.normal(key, (7, 3, 7))
    is_first = jnp.zeros((7, 3), bool).at[0].set(True)
    is_first = is_first.at[4, 1].set(True)
    params = module.init(key, pairs, is_first)["params"]
    parallel = module.apply({"params": params}, pairs, is_first)

    cache = module.initial(3)
    recurrent = []
    for index in range(pairs.shape[0]):
        cache, hidden = module.apply(
            {"params": params},
            cache,
            pairs[index],
            is_first[index],
            method=module.step,
        )
        recurrent.append(hidden)
    recurrent = jnp.stack(recurrent)
    np.testing.assert_allclose(recurrent, parallel, rtol=2e-5, atol=2e-5)


def test_future_pairs_cannot_change_past_hidden_states() -> None:
    module = _module()
    key = jax.random.PRNGKey(1)
    pairs = jax.random.normal(key, (8, 2, 7))
    is_first = jnp.zeros((8, 2), bool).at[0].set(True)
    params = module.init(key, pairs, is_first)["params"]
    baseline = module.apply({"params": params}, pairs, is_first)
    changed = pairs.at[5:].add(100.0)
    perturbed = module.apply({"params": params}, changed, is_first)
    np.testing.assert_allclose(perturbed[:5], baseline[:5], atol=1e-6)


def test_reset_isolates_previous_episode() -> None:
    module = _module()
    key = jax.random.PRNGKey(2)
    pairs = jax.random.normal(key, (8, 2, 7))
    is_first = jnp.zeros((8, 2), bool).at[0].set(True).at[4].set(True)
    params = module.init(key, pairs, is_first)["params"]
    baseline = module.apply({"params": params}, pairs, is_first)
    changed = pairs.at[:4].add(100.0)
    perturbed = module.apply({"params": params}, changed, is_first)
    np.testing.assert_allclose(perturbed[4:], baseline[4:], atol=1e-6)
