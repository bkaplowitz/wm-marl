from __future__ import annotations

import jax
import numpy as np

from world_marl.dreamarl.checkpointing import (
    load_dreamarl_checkpoint,
    save_dreamarl_checkpoint,
)
from world_marl.dreamarl.config import ReplayConfig
from world_marl.dreamarl.contracts import (
    sequence_batch_to_jax,
    sequence_batch_to_numpy,
)
from world_marl.dreamarl.environments import CoinGameAdapter
from world_marl.dreamarl.learner import DreaMARLLearner
from world_marl.dreamarl.replay import JointSequenceReplay
from world_marl.dreamarl.runtime import DreaMARLRuntime

from test_dreamarl_world_model import _config


def test_checkpoint_roundtrip_restores_every_training_state(tmp_path) -> None:
    config = _config()
    adapter = CoinGameAdapter(max_episode_steps=4)
    runtime = DreaMARLRuntime(adapter, config)
    driver = runtime.initialize_driver(3, jax.random.PRNGKey(70))
    collected = runtime.collect_random(driver, 8)
    replay = JointSequenceReplay(
        ReplayConfig(capacity=48, sequence_length=4, batch_size=2), seed=71
    )
    replay.append(
        sequence_batch_to_numpy(
            collected.transitions, adapter.spec.agent_ids
        )
    )
    sample = sequence_batch_to_jax(replay.sample_template())
    learner = DreaMARLLearner(config)
    learner_state = learner.initialize(sample, jax.random.PRNGKey(72))
    learner_state = learner.world_model_step(learner_state, sample).state
    context = runtime.initialize_policy_context(
        collected.driver, learner_state.world_model.params
    )
    save_dreamarl_checkpoint(
        tmp_path,
        learner_state=learner_state,
        driver=collected.driver,
        policy_context=context,
        replay=replay,
        metadata={"environment_steps": 24},
    )
    expected_sample = replay.sample()

    restored_replay = JointSequenceReplay(replay.config, seed=0)
    restored_replay.load(tmp_path)
    restored_template = learner.initialize(sample, jax.random.PRNGKey(0))
    driver_template = runtime.initialize_driver(3, jax.random.PRNGKey(0))
    context_template = runtime.initialize_policy_context(
        driver_template, restored_template.world_model.params
    )
    restored, restored_driver, restored_context, metadata = (
        load_dreamarl_checkpoint(
            tmp_path,
            learner_template=restored_template,
            driver_template=driver_template,
            policy_context_template=context_template,
            replay=restored_replay,
        )
    )
    assert metadata["environment_steps"] == 24
    for expected_tree, actual_tree in (
        (learner_state, restored),
        (collected.driver, restored_driver),
        (context, restored_context),
    ):
        for expected, actual in zip(
            jax.tree.leaves(expected_tree),
            jax.tree.leaves(actual_tree),
            strict=True,
        ):
            np.testing.assert_array_equal(expected, actual)
    actual_sample = restored_replay.sample()
    for field in actual_sample.__dataclass_fields__:
        if field != "agent_ids":
            np.testing.assert_array_equal(
                getattr(expected_sample, field), getattr(actual_sample, field)
            )
