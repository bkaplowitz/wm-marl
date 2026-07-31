from __future__ import annotations

from dataclasses import replace

import jax

from world_marl.dreamarl.learner import DreaMARLLearner
from world_marl.dreamarl.losses import world_model_loss

from test_dreamarl_world_model import _batch, _config


def test_world_model_can_overfit_a_fixed_tiny_sequence() -> None:
    config = _config()
    config = replace(
        config,
        optimizer=replace(
            config.optimizer, world_model_learning_rate=2e-3
        ),
        world_model_loss=replace(config.world_model_loss, free_nats=0.0),
    )
    batch = _batch()
    learner = DreaMARLLearner(config)
    state = learner.initialize(batch, jax.random.PRNGKey(80))
    evaluation_key = jax.random.PRNGKey(81)

    def loss(current) -> float:
        return float(
            world_model_loss(
                learner.world_model,
                current.world_model.params,
                current.target_encoder,
                batch,
                evaluation_key,
                config,
            ).loss
        )

    initial = loss(state)
    for _ in range(100):
        state = learner.world_model_step(state, batch).state
    final = loss(state)
    assert final < 0.15 * initial
