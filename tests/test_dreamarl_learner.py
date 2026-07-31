from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from world_marl.dreamarl.contracts import stack_sequence_batches
from world_marl.dreamarl.learner import DreaMARLLearner

from test_dreamarl_world_model import _batch, _config


def _all_finite(tree) -> bool:
    return all(bool(jnp.all(jnp.isfinite(value))) for value in jax.tree.leaves(tree))


def test_learner_updates_all_components_and_targets() -> None:
    learner = DreaMARLLearner(_config())
    batch = _batch()
    state = learner.initialize(batch, jax.random.PRNGKey(50))
    counts = learner.parameter_counts(state)
    assert counts["total"] == (
        counts["world_model"] + counts["actor"] + counts["critic"]
    )
    assert all(value > 0 for value in counts.values())

    previous_encoder = state.target_encoder
    world_output = learner.world_model_step(state, batch)
    assert int(world_output.state.world_updates) == 1
    assert _all_finite(world_output.state.world_model.params)
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(
            jax.tree.leaves(previous_encoder),
            jax.tree.leaves(world_output.state.target_encoder),
            strict=True,
        )
    )

    control_output = learner.actor_critic_step(world_output.state, batch)
    assert int(control_output.state.actor_updates) == 1
    assert int(control_output.state.critic_updates) == 1
    assert _all_finite(control_output.state.actor.params)
    assert _all_finite(control_output.state.critic.params)
    assert all(
        bool(jnp.isfinite(value)) for value in control_output.metrics.values()
    )


def test_initialized_and_updated_learner_is_seed_deterministic() -> None:
    learner = DreaMARLLearner(_config())
    batch = _batch()
    first = learner.initialize(batch, jax.random.PRNGKey(51))
    second = learner.initialize(batch, jax.random.PRNGKey(51))
    first = learner.world_model_step(first, batch).state
    second = learner.world_model_step(second, batch).state
    first = learner.actor_critic_step(first, batch).state
    second = learner.actor_critic_step(second, batch).state
    for left, right in zip(
        jax.tree.leaves(first), jax.tree.leaves(second), strict=True
    ):
        np.testing.assert_array_equal(left, right)


def test_prefetched_train_scan_applies_one_atomic_pair_per_batch() -> None:
    learner = DreaMARLLearner(_config())
    batch = _batch()
    state = learner.initialize(batch, jax.random.PRNGKey(52))
    batches = stack_sequence_batches([batch, batch, batch])
    output = learner.train_steps(state, batches)
    assert int(output.state.world_updates) == 3
    assert int(output.state.actor_updates) == 3
    assert int(output.state.critic_updates) == 3
    assert output.metrics["world_model/loss"].shape == (3,)
    assert output.metrics["actor/loss"].shape == (3,)
