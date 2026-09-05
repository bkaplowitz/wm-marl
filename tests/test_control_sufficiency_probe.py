"""Leakage and alignment gates for the read-only representation probe."""

import json
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from scripts.probe_control_sufficiency import (
    complete_episode_slices,
    load_episodes,
    load_frozen_params,
    main,
    make_extractor,
    ridge_probe,
    short_returns,
    split_episodes,
)
from majepa.main import _load_configs, _resolve_config_profiles


def test_return_targets_start_at_next_reward_and_keep_terminal_tail():
    result = short_returns([0, 2, 4, 100], [False, False, True, False], [1, 3], 0.5)
    np.testing.assert_allclose(result[:3], [[2, 4], [4, 4], [0, 0]])
    assert np.isnan(result[3]).all()
    truncated = short_returns([0, 2, 4], [False, False, False], [1, 3], 0.5)
    np.testing.assert_allclose(truncated[:2, 0], [2, 4])
    assert np.isnan(truncated[:, 1]).all()


def test_episode_ranges_discard_partial_chunks_and_reset_crossings():
    first = [False, False, True, False, True, False, False, True]
    last = [False, True, False, False, False, False, True, False]
    assert list(complete_episode_slices(first, last)) == [(4, 7)]
    identities = np.repeat(np.arange(15), 4)
    partitions = split_episodes(identities, 10)
    np.testing.assert_array_equal(sum(mask.astype(int) for mask in partitions), 1)
    split_ids = [set(identities[mask]) for mask in partitions]
    assert not (
        split_ids[0] & split_ids[1]
        or split_ids[0] & split_ids[2]
        or split_ids[1] & split_ids[2]
    )


def test_ridge_generalizes_known_linear_signal_without_test_target_leakage():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(120, 5))
    y = x[:, :2] @ np.asarray([[2.0], [-3.0]])
    partitions = split_episodes(np.repeat(np.arange(12), 10), 2)
    result = ridge_probe(x, y, partitions, [1e-7, 1e-3, 10], 1000, 0)
    assert result["r2"][0] > 0.999
    altered = y.copy()
    altered[partitions[2]] += 1000.0
    changed = ridge_probe(x, altered, partitions, [1e-7, 1e-3, 10], 1000, 0)
    assert result["alpha"] == changed["alpha"]
    np.testing.assert_allclose(result["validation_rmse"], changed["validation_rmse"])
    assert changed["rmse"][0] > 999


def test_plain_evaluation_export_without_replay_step_ids(tmp_path):
    # Frozen policy calibration exports complete episodes in episodes.npz,
    # without replay UUID filenames or replay-internal step IDs.
    first = np.tile([True, False, False], 6)
    last = np.tile([False, False, True], 6)
    np.savez_compressed(
        tmp_path / "episodes.npz",
        observation=np.zeros((18, 2, 4), np.float32),
        action=np.zeros((18, 2), np.int32),
        action_mask=np.ones((18, 2, 3), bool),
        reward=np.zeros((18, 2), np.float32),
        is_first=first,
        is_last=last,
        is_terminal=last,
    )
    episodes, provenance = load_episodes(tmp_path, 1, 6, 0)
    assert provenance["complete_episodes_available"] == 6
    assert {episode.identity for episode in episodes} == {
        f"episodes.npz:{start}" for start in range(0, 18, 3)
    }
    assert all(len(episode.data["reward"]) == 3 for episode in episodes)


def test_frozen_extractor_reload_is_causal_and_matches_full_prefix(tmp_path):
    config = _resolve_config_profiles(
        _load_configs(), ("smac_vector", "ma_jepa", "debug")
    )
    config = config.update(
        {
            "agent.dyn.parallel_transformer.deter": 8,
            "agent.dyn.parallel_transformer.hidden": 8,
        }
    )
    pure, initialize = make_extractor(config, 6, 4)
    observation = jax.random.normal(jax.random.key(8), (2, 5, 6))
    previous_action = jnp.zeros((2, 5), jnp.int32)
    first = jnp.zeros((2, 5), bool).at[:, 0].set(True)
    active = jnp.ones((2, 5), bool)
    inputs = (observation, previous_action, first, active)
    params = initialize({}, *inputs, seed=3)
    checkpoint = tmp_path / "agent.pkl"
    with checkpoint.open("wb") as stream:
        pickle.dump(
            {"params": jax.device_get(params), "counters": {"actions": 15}}, stream
        )
    loaded, provenance = load_frozen_params(checkpoint, initialize, inputs)
    assert provenance["counters"]["actions"] == 15
    extract = jax.jit(pure)
    _, factual = extract(loaded, *inputs, seed=4)
    _, intervention = extract(
        loaded, observation.at[:, 3:].add(10), previous_action, first, active, seed=4
    )
    for key in factual:
        np.testing.assert_array_equal(factual[key][:, :3], intervention[key][:, :3])
    # Input actions are explicitly shifted; no action chosen at t or later is
    # needed to reconstruct the decision feature at t.
    _, changed_action = extract(
        loaded, observation, previous_action.at[:, 4].set(2), first, active, seed=4
    )
    np.testing.assert_array_equal(
        factual["posterior_mean"][:, :4], changed_action["posterior_mean"][:, :4]
    )


def test_probe_end_to_end_from_checkpoint_and_replay_without_environment(tmp_path):
    config = _resolve_config_profiles(
        _load_configs(), ("smac_vector", "ma_jepa", "debug")
    ).update(
        {
            "agent.num_agents": 2,
            "agent.dyn.parallel_transformer.deter": 8,
            "agent.dyn.parallel_transformer.hidden": 8,
        }
    )
    config.save(str(tmp_path / "config.yaml"))
    _, initialize = make_extractor(config, 6, 4)
    params = initialize(
        {},
        jnp.zeros((2, 1, 6)),
        jnp.zeros((2, 1), jnp.int32),
        jnp.ones((2, 1), bool),
        jnp.ones((2, 1), bool),
        seed=1,
    )
    checkpoint = tmp_path / "ckpt/example"
    checkpoint.mkdir(parents=True)
    (checkpoint / "done").touch()
    (tmp_path / "ckpt/latest").write_text("example")
    with (checkpoint / "agent.pkl").open("wb") as stream:
        pickle.dump({"params": jax.device_get(params), "counters": {}}, stream)
    replay = tmp_path / "replay"
    replay.mkdir()
    length = 6 * 5
    rng = np.random.default_rng(2)
    first = np.zeros(length, bool)
    first[::5] = True
    last = np.zeros(length, bool)
    last[4::5] = True
    np.savez_compressed(
        replay / "example.npz",
        observation=rng.normal(size=(length, 2, 6)).astype(np.float32),
        action=rng.integers(0, 4, size=(length, 2), dtype=np.int32),
        action_mask=np.ones((length, 2, 4), bool),
        reward=np.repeat(rng.uniform(size=(length, 1)).astype(np.float32), 2, axis=1),
        is_first=first,
        is_last=last,
        is_terminal=last,
        stepid=np.arange(length, dtype=np.int64),
    )
    output = tmp_path / "probe.json"
    assert (
        main(
            [
                "--run",
                str(tmp_path),
                "--output",
                str(output),
                "--platform",
                "cpu",
                "--max-episodes",
                "6",
                "--samples-per-episode",
                "2",
                "--max-train-rows",
                "20",
                "--horizons",
                "1",
                "5",
            ]
        )
        == 0
    )
    result = json.loads(output.read_text())
    assert result["replay"]["complete_episodes_available"] == 6
    assert set(result["probes"]) == {
        "raw",
        "encoder",
        "posterior_sample",
        "posterior_mean",
    }
    assert len(result["probes"]["encoder"]["return"]["r2"]) == 2
    assert result["probes"]["posterior_sample"]["observable"]["input_dimensions"] == 16
