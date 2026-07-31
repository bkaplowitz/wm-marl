from __future__ import annotations

import jax
import numpy as np

from world_marl.dreamarl.contracts import sequence_batch_to_numpy
from world_marl.dreamarl.environments import CoinGameAdapter
from world_marl.dreamarl.learner import DreaMARLLearner
from world_marl.dreamarl.runtime import DreaMARLRuntime

from test_dreamarl_world_model import _config


def test_random_collection_preserves_lifecycle_and_real_successors() -> None:
    config = _config()
    adapter = CoinGameAdapter(max_episode_steps=4)
    runtime = DreaMARLRuntime(adapter, config)
    driver = runtime.initialize_driver(3, jax.random.PRNGKey(60))
    output = runtime.collect_random(driver, 9)
    host = sequence_batch_to_numpy(
        output.transitions, adapter.spec.agent_ids
    )
    assert host.observations.shape == (9, 3, 2, 36)
    assert np.all(host.is_first[1:] == host.is_last[:-1])
    assert not np.any(host.is_terminal)
    assert int(np.sum(host.is_last)) == 6
    np.testing.assert_allclose(
        host.next_observations[:-1][~host.is_last[:-1]],
        host.observations[1:][~host.is_last[:-1]],
    )


def test_policy_collection_is_compiled_recurrent_and_seed_deterministic() -> None:
    config = _config()
    adapter = CoinGameAdapter(max_episode_steps=4)
    runtime = DreaMARLRuntime(adapter, config)
    driver = runtime.initialize_driver(3, jax.random.PRNGKey(61))
    random_output = runtime.collect_random(driver, 8)
    learner = DreaMARLLearner(config)
    learner_state = learner.initialize(
        random_output.transitions, jax.random.PRNGKey(62)
    )
    context = runtime.initialize_policy_context(
        random_output.driver, learner_state.world_model.params
    )
    arguments = (
        random_output.driver,
        context,
        learner_state.world_model.params,
        learner_state.actor.params,
        8,
        True,
    )
    first = runtime.collect_policy(*arguments)
    second = runtime.collect_policy(*arguments)
    for left, right in zip(
        jax.tree.leaves(first), jax.tree.leaves(second), strict=True
    ):
        np.testing.assert_array_equal(left, right)
    host = sequence_batch_to_numpy(
        first.transitions, adapter.spec.agent_ids
    )
    assert host.actions.shape == (8, 3, 2)
    assert np.all(host.actions >= 0)
    assert np.all(host.actions < config.action_dim)
