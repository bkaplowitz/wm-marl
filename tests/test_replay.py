import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from majepa.replay import DualViewReplay, ExponentialRecency


def test_recency_probabilities_follow_age_weights():
    selector = ExponentialRecency(0.5, seed=17)
    for key in range(3):
        selector[key] = ()
    samples = [selector() for _ in range(4096)]
    np.testing.assert_allclose(
        np.bincount(samples) / 4096, [1 / 7, 2 / 7, 4 / 7], atol=0.025
    )


def test_decay_one_is_uniform_and_reproducible():
    first = ExponentialRecency(1.0, seed=23)
    second = ExponentialRecency(1.0, seed=23)
    for key in range(8):
        first[key] = second[key] = ()
    samples = [first() for _ in range(4096)]
    assert samples == [second() for _ in range(4096)]
    np.testing.assert_allclose(
        np.bincount(samples) / 4096, np.full(8, 1 / 8), atol=0.025
    )


@pytest.mark.parametrize("decay", [0.0, -1.0, 1.1, float("nan"), float("inf")])
def test_recency_rejects_invalid_decay(decay):
    with pytest.raises(ValueError, match="decay"):
        ExponentialRecency(decay)


def test_dual_view_replay_keeps_both_mixtures_on_the_same_items(tmp_path):
    replay = DualViewReplay(
        length=2,
        capacity=4,
        chunksize=4,
        directory=tmp_path,
        optimized_length=1,
        recency_decay=0.9998,
        world_uniform_mix=0.5,
        behavior_recency_decay=0.9998,
        behavior_uniform_mix=0.5,
        seed=7,
    )
    _add_replay_steps(replay, workers=1, steps=8)
    assert len(replay.sampler) == len(replay.behavior_sampler) == 4
    assert set(replay.sampler.steps) == set(replay.behavior_sampler.steps)
    for mode in ("train_world", "train_behavior"):
        values = replay.sample(4, mode)["value"]
        assert values.shape == (4, 2)
        np.testing.assert_array_equal(np.diff(values, axis=1), 1)
        assert values.min() >= 3
        assert values.max() <= 7


def _add_replay_steps(replay, workers=2, steps=12, interleaved=False):
    order = ((worker, step) for worker in range(workers) for step in range(steps))
    if interleaved:
        order = ((worker, step) for step in range(steps) for worker in range(workers))
    for worker, step in order:
        replay.add(
            {
                "value": np.asarray(worker * 100 + step, np.int32),
                "is_first": np.asarray(step == 0),
                "is_last": np.asarray(step == steps - 1),
                "is_terminal": np.asarray(step == steps - 1),
            },
            worker,
        )


@pytest.mark.parametrize("mode", ["train_world", "train_behavior", "report"])
def test_seeded_replay_repeats_fresh_batch_sequence(mode):
    first, second = [
        DualViewReplay(
            length=3,
            capacity=100,
            seed=41,
            isolate_report_rng=True,
            world_uniform_mix=0.5,
            behavior_recency_decay=0.9998,
            behavior_uniform_mix=0.5,
        )
        for _ in range(2)
    ]
    _add_replay_steps(first)
    _add_replay_steps(second)
    for _ in range(12):
        np.testing.assert_array_equal(
            first.sample(4, mode)["value"],
            second.sample(4, mode)["value"],
        )


@pytest.mark.parametrize(
    ("capacity", "workers", "interleaved"),
    [(100, 2, False), (4, 1, False), (100, 2, True)],
)
def test_dual_view_checkpoint_restores_all_sampling_rngs(
    tmp_path, capacity, workers, interleaved
):
    replay = DualViewReplay(
        length=3,
        capacity=capacity,
        chunksize=4,
        directory=tmp_path,
        save_wait=True,
        optimized_length=2,
        recency_decay=0.9998,
        world_uniform_mix=0.5,
        behavior_recency_decay=0.9998,
        behavior_uniform_mix=0.5,
        isolate_report_rng=True,
        seed=43,
    )
    _add_replay_steps(replay, workers=workers, interleaved=interleaved)
    for _ in range(5):
        replay.sample(3, "train_world")
        replay.sample(3, "train_behavior")
    state = replay.save()
    expected = [
        (
            replay.sample(3, "train_world")["value"].copy(),
            replay.sample(3, "train_behavior")["value"].copy(),
            replay.sample(3, "report")["value"].copy(),
        )
        for _ in range(8)
    ]

    restored = DualViewReplay(
        length=3,
        capacity=capacity,
        chunksize=4,
        directory=tmp_path,
        save_wait=True,
        optimized_length=2,
        recency_decay=0.9998,
        world_uniform_mix=0.5,
        behavior_recency_decay=0.9998,
        behavior_uniform_mix=0.5,
        isolate_report_rng=True,
        seed=999,
    )
    restored.load(state)
    actual = [
        (
            restored.sample(3, "train_world")["value"].copy(),
            restored.sample(3, "train_behavior")["value"].copy(),
            restored.sample(3, "report")["value"].copy(),
        )
        for _ in range(8)
    ]
    for expected_pair, actual_pair in zip(expected, actual):
        np.testing.assert_array_equal(expected_pair[0], actual_pair[0])
        np.testing.assert_array_equal(expected_pair[1], actual_pair[1])
        np.testing.assert_array_equal(expected_pair[2], actual_pair[2])


def test_checkpoint_rejects_a_different_retained_item_set(tmp_path):
    replay = DualViewReplay(
        length=3,
        capacity=4,
        chunksize=4,
        directory=tmp_path,
        save_wait=True,
    )
    _add_replay_steps(replay, workers=1)
    state = replay.save()
    restored = DualViewReplay(length=3, capacity=2, directory=tmp_path)
    with pytest.raises(ValueError, match="retained replay items"):
        restored.load(state)


def test_checkpoint_sampling_survives_a_new_process_hash_seed(tmp_path):
    script = """
import pickle
import sys
from pathlib import Path
import numpy as np
from majepa.replay import DualViewReplay

root = Path(sys.argv[1])
replay = DualViewReplay(
    length=3, capacity=4, chunksize=4, directory=root / "replay", save_wait=True,
    world_uniform_mix=0.5, behavior_recency_decay=0.9998,
    behavior_uniform_mix=0.5, isolate_report_rng=True, seed=43,
)
modes = ("train_world", "train_behavior", "report")
if sys.argv[2] == "save":
    for step in range(12):
        replay.add({
            "value": np.asarray(step, np.int32),
            "is_first": np.asarray(step == 0),
            "is_last": np.asarray(step == 11),
            "is_terminal": np.asarray(step == 11),
        })
    (root / "checkpoint.pkl").write_bytes(pickle.dumps(replay.save()))
    np.savez(root / "expected.npz", **{
        mode: replay.sample(12, mode)["value"] for mode in modes
    })
else:
    replay.load(pickle.loads((root / "checkpoint.pkl").read_bytes()))
    with np.load(root / "expected.npz") as expected:
        for mode in modes:
            np.testing.assert_array_equal(replay.sample(12, mode)["value"], expected[mode])
replay.workers.shutdown(wait=True)
"""
    root = Path(__file__).parents[1]
    for phase, hash_seed in (("save", "1"), ("load", "2")):
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), phase],
            env={
                **os.environ,
                "PYTHONHASHSEED": hash_seed,
                "PYTHONPATH": os.pathsep.join(
                    (str(root / "src"), str(root / "external/dreamerv3"))
                ),
            },
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
