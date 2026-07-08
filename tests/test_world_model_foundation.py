from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from world_marl.world_model_foundation.metrics import METRIC_KEYS
from world_marl.world_model_foundation.preprocess import normalize_observations
from world_marl.world_model_foundation.replay import (
    WorldModelSequenceBatch,
    synthetic_observation_batch,
)
from world_marl.world_model_foundation.sources import world_model_sources


ROOT = Path(__file__).resolve().parents[1]


def test_sequence_batch_validates_time_major_contract() -> None:
    observations = np.zeros((5, 3, 8, 8, 3), dtype=np.float32)
    actions = np.zeros((5, 3), dtype=np.int32)
    rewards = np.zeros((5, 3), dtype=np.float32)
    continues = np.ones((5, 3), dtype=np.float32)
    is_first = np.zeros((5, 3), dtype=bool)
    is_terminal = np.zeros((5, 3), dtype=bool)

    batch = WorldModelSequenceBatch(
        observations=observations,
        actions=actions,
        rewards=rewards,
        continues=continues,
        is_first=is_first,
        is_terminal=is_terminal,
        metadata={"env": "synthetic:image-grid"},
    )

    assert batch.time_steps == 5
    assert batch.batch_size == 3
    assert batch.observation_shape == (8, 8, 3)
    assert batch.action_shape == ()
    assert batch.metadata["env"] == "synthetic:image-grid"


def test_sequence_batch_rejects_shape_mismatch() -> None:
    observations = np.zeros((5, 3, 8, 8, 3), dtype=np.float32)
    bad_rewards = np.zeros((4, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="rewards must start with"):
        WorldModelSequenceBatch(
            observations=observations,
            actions=np.zeros((5, 3), dtype=np.int32),
            rewards=bad_rewards,
            continues=np.ones((5, 3), dtype=np.float32),
            is_first=np.zeros((5, 3), dtype=bool),
            is_terminal=np.zeros((5, 3), dtype=bool),
        )


def test_synthetic_observation_batch_is_deterministic_and_time_consistent() -> None:
    first = synthetic_observation_batch(
        time_steps=6, batch_size=2, observation_shape=(6, 6, 3)
    )
    second = synthetic_observation_batch(
        time_steps=6, batch_size=2, observation_shape=(6, 6, 3)
    )

    np.testing.assert_array_equal(first.observations, second.observations)
    assert first.observations.shape == (6, 2, 6, 6, 3)
    assert first.actions.shape == (6, 2)
    assert bool(np.all(first.is_first[0]))
    assert not bool(np.any(first.is_first[1:]))
    assert bool(np.all(first.continues == 1.0 - first.is_terminal.astype(np.float32)))


def test_normalize_observations_accepts_uint8_and_float_inputs() -> None:
    uint8_images = np.array([0, 127, 255], dtype=np.uint8).reshape((1, 1, 1, 3))
    normalized = normalize_observations(uint8_images)

    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized.reshape(-1), [0.0, 127.0 / 255.0, 1.0])

    float_images = np.array([-1.0, 0.5, 2.0], dtype=np.float32).reshape((1, 1, 1, 3))
    clipped = normalize_observations(float_images)
    np.testing.assert_allclose(clipped.reshape(-1), [0.0, 0.5, 1.0])


def test_metric_keys_and_sources_cover_world_model_foundation() -> None:
    for name in (
        "observation_prediction_loss",
        "reconstruction_loss",
        "reward_loss",
        "continue_loss",
        "token_prediction_loss",
        "real_env_return",
    ):
        assert name in METRIC_KEYS

    sources = world_model_sources()
    assert sources["dreamer_v3"]["paper_url"] == "https://arxiv.org/abs/2301.04104"
    assert sources["genie"]["paper_url"] == "https://arxiv.org/abs/2402.15391"
    assert (
        sources["genie_3"]["announcement_url"]
        == "https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/"
    )
    assert sources["jasmine"]["repo_url"] == "https://github.com/p-doom/jasmine"
    assert sources["jafar"]["repo_url"] == "https://github.com/FLAIROx/jafar"


def test_architecture_docs_lock_source_papers_and_boundaries() -> None:
    dreamer_doc = (
        ROOT / "src/world_marl/dreamer_v3_baseline/ARCHITECTURE.md"
    ).read_text()
    genie_doc = (ROOT / "src/world_marl/genie_like_jax/ARCHITECTURE.md").read_text()

    assert "https://arxiv.org/abs/2301.04104" in dreamer_doc
    assert "taken directly from the DreamerV3 paper" in dreamer_doc
    assert "Genie" not in dreamer_doc
    assert "LeJEPA" not in dreamer_doc

    assert "https://arxiv.org/abs/2402.15391" in genie_doc
    assert "taken directly from the public Genie paper" in genie_doc
    assert "LAM produces discrete latent action codes" in genie_doc
    assert "VQ-VAE video tokenizer is primary" in genie_doc
    assert "dynamics predicts next-frame tokens" in genie_doc
    assert "Direct next-observation generation is a modern variant" in genie_doc
    assert (
        "Genie 3 is a capability target, not a complete public architecture"
        in genie_doc
    )
    assert "https://github.com/p-doom/jasmine" in genie_doc
    assert "LeWM/LeJEPA innovations are ablations only" in genie_doc
