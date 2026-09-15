import math

import numpy as np
import pytest

from majepa.config import MAJEPARunSpec
from majepa.replay import DualViewReplay, ExponentialRecency, TruncatedGeometric


def _populate(selector, count):
    for key in range(count):
        selector[key] = ()


def test_truncated_geometric_is_the_equivalent_exponential_selector():
    alpha = 10.0
    capacity = 250_000
    decay = 2.0 ** (-alpha / (capacity - 1))
    truncated = TruncatedGeometric(alpha=alpha, capacity=capacity, seed=17)
    exponential = ExponentialRecency(decay, seed=17)

    assert truncated.alpha == alpha
    assert truncated.capacity == capacity
    assert truncated.decay == pytest.approx(decay)

    _populate(truncated, 100)
    _populate(exponential, 100)
    assert [truncated() for _ in range(128)] == [exponential() for _ in range(128)]


def test_zero_alpha_is_uniform_and_reproducible():
    truncated = TruncatedGeometric(alpha=0.0, capacity=8, seed=23)
    uniform = ExponentialRecency(1.0, seed=23)
    _populate(truncated, 8)
    _populate(uniform, 8)
    assert truncated.decay == 1.0
    assert [truncated() for _ in range(64)] == [uniform() for _ in range(64)]


@pytest.mark.parametrize(
    ("alpha", "capacity", "message"),
    [
        (-1.0, 10, "alpha"),
        (math.inf, 10, "alpha"),
        (10.0, 1, "capacity"),
    ],
)
def test_truncated_geometric_validates_parameters(alpha, capacity, message):
    with pytest.raises(ValueError, match=message):
        TruncatedGeometric(alpha=alpha, capacity=capacity)


def test_full_buffer_has_paper_alpha_slope():
    selector = TruncatedGeometric(alpha=10.0, capacity=250_000)
    # The oldest-to-newest weight ratio is q**(N-1) = 2**-alpha.
    assert selector.decay ** (selector.capacity - 1) == pytest.approx(2.0**-10)


def test_dual_view_replay_routes_world_sampling_to_truncated_geometric(tmp_path):
    replay = DualViewReplay(
        length=2,
        capacity=4,
        chunksize=4,
        directory=tmp_path,
        optimized_length=1,
        world_sampler="truncated_geometric",
        truncated_geometric_alpha=10.0,
        seed=7,
    )
    for step in range(8):
        replay.add(
            {
                "value": np.asarray(step, np.int32),
                "is_first": np.asarray(step == 0),
                "is_last": np.asarray(False),
                "is_terminal": np.asarray(False),
            }
        )

    assert isinstance(replay.sampler, TruncatedGeometric)
    assert len(replay.sampler) == len(replay.behavior_sampler) == 4
    world, _ = replay._sample("train_world")
    behavior, _ = replay._sample("train_behavior")
    assert world["value"][0].shape == behavior["value"][0].shape == (2,)


def test_run_spec_records_truncated_geometric_selection(tmp_path):
    spec = MAJEPARunSpec(
        experiment_dir=tmp_path,
        task="smac_3s_vs_4z",
        num_agents=3,
        platform="cpu",
        replay_sampling="truncated_geometric_world_uniform_behavior",
        world_uniform_mix=0.0,
        truncated_geometric_alpha=10.0,
    )

    assert spec.effective_replay_sampling == (
        "truncated_geometric_world_uniform_behavior"
    )
    assert spec.ctde_manifest["truncated_geometric_alpha"] == 10.0
    assert spec.to_dict()["truncated_geometric_alpha"] == 10.0
    assert spec.command[-8:] == [
        "--replay.sampling",
        "truncated_geometric_world_uniform_behavior",
        "--replay.recency_decay",
        "0.9998",
        "--replay.world_uniform_mix",
        "0.0",
        "--replay.truncated_geometric_alpha",
        "10.0",
    ]
