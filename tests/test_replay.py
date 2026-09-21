"""Sampling, storage and configuration of the maintained dual replay views."""

import numpy as np
import pytest

from majepa.config import MAJEPARunSpec
from majepa.main import _load_configs, _resolve_config_profiles
from majepa.replay import DualViewReplay, ExponentialRecency


def _populate(selector, count):
    for key in range(count):
        selector[key] = ()


def test_recency_draws_follow_the_finite_geometric_distribution():
    selector = ExponentialRecency(0.75, seed=17)
    _populate(selector, 8)
    samples = np.asarray([selector() for _ in range(20_000)])
    observed = np.bincount(samples, minlength=8) / len(samples)
    expected = 0.75 ** np.arange(7, -1, -1)
    expected /= expected.sum()
    np.testing.assert_allclose(observed, expected, atol=0.008, rtol=0)
    np.testing.assert_array_equal(selector.pop_sampled_ages(), 7 - samples)
    assert selector.pop_sampled_ages() == []


def test_uniform_recency_is_seeded_and_respects_fifo_eviction():
    first = ExponentialRecency(1.0, seed=23)
    second = ExponentialRecency(1.0, seed=23)
    for selector in (first, second):
        _populate(selector, 8)
        del selector[0]
        selector[8] = ()
    draws = [first() for _ in range(128)]
    assert draws == [second() for _ in range(128)]
    assert set(draws) == set(range(1, 9))


@pytest.mark.parametrize("decay", [0.0, -0.1, 1.1, np.inf, np.nan])
def test_recency_rejects_invalid_decay(decay):
    with pytest.raises(ValueError, match="decay"):
        ExponentialRecency(decay)


def _replay(directory):
    replay = DualViewReplay(
        length=2,
        capacity=4,
        chunksize=4,
        directory=directory,
        optimized_length=1,
        recency_decay=0.9,
        world_uniform_mix=0.5,
        behavior_recency_decay=0.8,
        behavior_uniform_mix=0.5,
        isolate_report_rng=True,
        seed=7,
    )
    for step in range(8):
        replay.add(
            {
                "value": np.asarray(step, np.int32),
                "action": np.asarray([step % 3, (step + 1) % 3], np.int32),
                "behavior_logprob": np.asarray([-step / 10, -step / 20], np.float32),
                "is_first": np.asarray(step == 0),
                "is_last": np.asarray(False),
                "is_terminal": np.asarray(False),
            }
        )
    return replay


def test_dual_views_retain_aligned_actions_probabilities_and_the_same_items(tmp_path):
    replay = _replay(tmp_path)
    assert isinstance(replay.sampler, ExponentialRecency)
    for selector in (
        replay.sampler,
        replay.world_uniform_sampler,
        replay.behavior_sampler,
        replay.behavior_uniform_sampler,
        replay.report_sampler,
    ):
        assert len(selector) == 4
    assert replay.sampler.decay == 0.9
    assert replay.behavior_sampler.decay == 0.8
    for mode in ("train_world", "train_behavior", "report"):
        for _ in range(8):
            sequence, online = replay._sample(mode)
            values = {key: np.concatenate(parts) for key, parts in sequence.items()}
            t = values["value"]
            assert t.shape == (2,) and int(t[1] - t[0]) == 1
            assert not online
            np.testing.assert_array_equal(
                values["action"], np.stack([t % 3, (t + 1) % 3], -1)
            )
            np.testing.assert_allclose(
                values["behavior_logprob"], np.stack([-t / 10, -t / 20], -1)
            )
    stats = replay.stats()
    assert stats["world_samples"] == stats["behavior_samples"] == 8


def test_world_and_report_draws_do_not_advance_behavior_rng(tmp_path):
    first, second = _replay(tmp_path / "first"), _replay(tmp_path / "second")
    for _ in range(20):
        first._sample("train_world")
        first._sample("report")
        a, _ = first._sample("train_behavior")
        b, _ = second._sample("train_behavior")
        np.testing.assert_array_equal(
            np.concatenate(a["value"]), np.concatenate(b["value"])
        )


def test_run_spec_matches_resolved_reference_replay_settings(tmp_path):
    spec = MAJEPARunSpec(tmp_path, "smac_3s_vs_4z", 3, platform="cpu")
    config = _resolve_config_profiles(_load_configs(), spec.configs)
    manifest = spec.to_dict()
    assert config.replay.sampling == "recent_world_uniform_behavior"
    assert spec.effective_replay_sampling == (
        "50% recent + 50% uniform for independent WM and PPO views"
    )
    assert config.replay.world_uniform_mix == 0.5
    assert config.replay.recency_decay == 0.9998
    assert config.replay.behavior_uniform_mix == 0.5
    assert config.replay.behavior_recency_decay == 0.9998
    assert manifest["command"][manifest["command"].index("--configs") + 1] == "baseline"
