from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from world_marl.world_model_foundation.collect import (
    adapter_action_mode,
    collect_adapter_sequence,
    collect_world_model_sequence,
    make_single_agent_adapter,
    synthetic_sequence_collector,
    write_json_artifact,
    write_jsonl_metrics,
)
from world_marl.world_model_foundation.metrics import METRIC_KEYS
from world_marl.world_model_foundation.preprocess import normalize_observations
from world_marl.world_model_foundation.replay import (
    WorldModelSequenceBatch,
    synthetic_observation_batch,
)
from world_marl.world_model_foundation.sources import world_model_sources


ROOT = Path(__file__).resolve().parents[1]


class _FakeDiscreteVectorAdapter:
    substrate = "fake:discrete-vector"
    num_envs = 2
    observation_shape = (3,)
    raw_observation_shape = observation_shape
    action_shape = ()
    action_dim = 3
    action_low = None
    action_high = None

    def __init__(self) -> None:
        self._step = 0

    def reset(self) -> np.ndarray:
        self._step = 0
        return np.zeros((self.num_envs, 1, *self.observation_shape), dtype=np.float32)

    def step(self, actions: np.ndarray) -> SimpleNamespace:
        self._step += 1
        flat_actions = np.asarray(actions, dtype=np.int32).reshape((self.num_envs,))
        observations = np.full(
            (self.num_envs, 1, *self.observation_shape),
            fill_value=float(self._step),
            dtype=np.float32,
        )
        rewards = flat_actions.astype(np.float32).reshape((self.num_envs, 1))
        dones = np.zeros((self.num_envs, 1), dtype=np.float32)
        dones[0, 0] = float(self._step == 3)
        return SimpleNamespace(
            observations=observations,
            rewards=rewards,
            dones=dones,
            completed_returns=(),
            completed_lengths=(),
            step_infos=(),
            infos=(),
        )

    def sample_actions(self, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(
            low=0,
            high=self.action_dim,
            size=(self.num_envs, 1),
            dtype=np.int32,
        )


class _FakeContinuousImageAdapter:
    substrate = "fake:continuous-image"
    num_envs = 2
    observation_shape = (4, 5, 3)
    raw_observation_shape = observation_shape
    action_shape = (2,)
    action_dim = 2
    action_low = np.array([-1.0, -0.5], dtype=np.float32)
    action_high = np.array([1.0, 0.5], dtype=np.float32)

    def __init__(self) -> None:
        self._step = 0
        self.seen_actions: list[np.ndarray] = []

    def reset(self) -> np.ndarray:
        self._step = 0
        self.seen_actions.clear()
        return np.zeros((self.num_envs, 1, *self.observation_shape), dtype=np.float32)

    def step(self, actions: np.ndarray) -> SimpleNamespace:
        self._step += 1
        action_batch = np.asarray(actions, dtype=np.float32).reshape(
            (self.num_envs, self.action_dim)
        )
        self.seen_actions.append(action_batch)
        observations = np.full(
            (self.num_envs, 1, *self.observation_shape),
            fill_value=float(self._step) / 10.0,
            dtype=np.float32,
        )
        rewards = action_batch.sum(axis=-1, keepdims=True).astype(np.float32)
        dones = np.zeros((self.num_envs, 1), dtype=np.float32)
        return SimpleNamespace(
            observations=observations,
            rewards=rewards,
            dones=dones,
            completed_returns=(),
            completed_lengths=(),
            step_infos=(),
            infos=(),
        )

    def sample_actions(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(
            low=self.action_low,
            high=self.action_high,
            size=(self.num_envs, self.action_dim),
        ).astype(np.float32)[:, None, :]


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
    assert (
        sources["genie_2"]["announcement_url"]
        == "https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/"
    )
    assert sources["genie"]["paper_url"] == "https://arxiv.org/abs/2402.15391"
    assert sources["genie"]["role"] == "genie1_vq_maskgit_ablation"
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
    genie_doc = (
        ROOT / "src/world_marl/genie2_continuous_jax/ARCHITECTURE.md"
    ).read_text()

    assert "https://arxiv.org/abs/2301.04104" in dreamer_doc
    assert "taken directly from the DreamerV3 paper" in dreamer_doc
    assert "Genie" not in dreamer_doc
    assert "LeJEPA" not in dreamer_doc

    assert (
        "https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/"
        in genie_doc
    )
    assert "Genie 2 is the primary architecture target" in genie_doc
    assert "autoregressive latent diffusion world model" in genie_doc
    assert "continuous latent autoencoder" in genie_doc
    assert "causal transformer dynamics" in genie_doc
    assert "classifier-free guidance" in genie_doc
    assert "continuous LAM" in genie_doc
    assert "LAM infers continuous latent actions" in genie_doc
    assert "VQ/MaskGIT is ablation-only" in genie_doc
    assert "Genie 3 is a capability target" in genie_doc
    assert "latent-to-real-action bridge" in genie_doc
    assert "https://github.com/p-doom/jasmine" in genie_doc
    assert "LeWM/LeJEPA innovations are ablations only" in genie_doc


def test_synthetic_collector_and_artifact_writers(tmp_path: Path) -> None:
    batch = synthetic_sequence_collector(
        env_name="synthetic:image-grid",
        time_steps=4,
        batch_size=2,
        observation_shape=(5, 5, 3),
        action_dim=3,
    )

    assert batch.observations.shape == (4, 2, 5, 5, 3)
    assert batch.metadata["env"] == "synthetic:image-grid"
    assert batch.metadata["collector"] == "synthetic_sequence_collector"

    config_path = write_json_artifact(tmp_path / "config.json", {"train_steps": 2})
    metrics_path = write_jsonl_metrics(
        tmp_path / "metrics.jsonl",
        [{"step": 0, "loss": 1.0}, {"step": 1, "loss": 0.5}],
    )

    assert config_path.read_text().strip() == '{\n  "train_steps": 2\n}'
    assert metrics_path.read_text().count("\n") == 2


def test_adapter_collection_preserves_vector_observation_and_discrete_actions() -> None:
    adapter = _FakeDiscreteVectorAdapter()

    batch = collect_adapter_sequence(
        adapter,
        env_name="fake:discrete-vector",
        time_steps=4,
        seed=7,
    )

    assert adapter_action_mode(adapter) == "discrete"
    assert batch.observations.shape == (4, 2, 3)
    assert batch.actions.shape == (4, 2)
    assert batch.actions.dtype == np.int32
    assert batch.rewards.shape == (4, 2)
    assert batch.continues.shape == (4, 2)
    assert bool(np.all(batch.is_first[0]))
    assert not bool(np.any(batch.is_first[1:]))
    assert bool(batch.is_terminal[2, 0])
    assert batch.continues[2, 0] == 0.0
    assert batch.metadata["env"] == "fake:discrete-vector"
    assert batch.metadata["action_mode"] == "discrete"
    assert batch.metadata["observation_shape"] == (3,)
    assert batch.metadata["action_shape"] == ()
    assert batch.metadata["action_dim"] == 3


def test_adapter_collection_preserves_hwc_observation_and_continuous_actions() -> None:
    adapter = _FakeContinuousImageAdapter()

    batch = collect_adapter_sequence(
        adapter,
        env_name="fake:continuous-image",
        time_steps=3,
        seed=11,
    )

    assert adapter_action_mode(adapter) == "continuous"
    assert batch.observations.shape == (3, 2, 4, 5, 3)
    assert batch.actions.shape == (3, 2, 2)
    assert batch.actions.dtype == np.float32
    assert len(adapter.seen_actions) == 3
    assert np.all(batch.actions >= adapter.action_low - 1e-6)
    assert np.all(batch.actions <= adapter.action_high + 1e-6)
    assert batch.metadata["env"] == "fake:continuous-image"
    assert batch.metadata["action_mode"] == "continuous"
    assert batch.metadata["observation_shape"] == (4, 5, 3)
    assert batch.metadata["action_shape"] == (2,)
    assert batch.metadata["action_dim"] == 2


def test_collect_world_model_sequence_dispatches_synthetic_and_real_adapters() -> None:
    synthetic = collect_world_model_sequence(
        env_name="synthetic:image-grid",
        time_steps=2,
        batch_size=2,
        observation_shape=(3, 3, 1),
        action_dim=2,
    )
    assert synthetic.observations.shape == (2, 2, 3, 3, 1)
    assert synthetic.metadata["collector"] == "synthetic_sequence_collector"

    with pytest.raises(ValueError, match="dmc:<domain>/<task>"):
        make_single_agent_adapter("dmc:cartpole", num_envs=1, max_cycles=2, seed=0)


def test_brax_adapter_can_be_constructed_when_dependency_is_installed() -> None:
    pytest.importorskip("brax")

    adapter = make_single_agent_adapter(
        "brax:reacher",
        num_envs=1,
        max_cycles=2,
        seed=0,
    )
    try:
        assert adapter_action_mode(adapter) == "continuous"
        assert adapter.observation_shape == (11,)
        assert adapter.action_shape == (2,)
    finally:
        close = getattr(adapter, "close", None)
        if close is not None:
            close()
