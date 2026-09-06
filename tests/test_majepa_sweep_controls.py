"""Regression checks for diagnostic RNG isolation and return-scale state."""

from types import SimpleNamespace

import elements
import embodied
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from majepa.models.normalize import Normalize
from majepa.replay import DualViewReplay
from majepa.streams import isolated_report_stream


def test_report_sampling_cannot_change_training_roots():
    def roots(with_reports):
        replay = DualViewReplay(
            length=4,
            capacity=60,
            chunksize=8,
            seed=0,
            isolate_report_rng=True,
        )
        for index in range(80):
            replay.add(
                dict(
                    marker=np.int32(index),
                    is_first=np.bool_(index == 0),
                    is_last=np.bool_(False),
                    is_terminal=np.bool_(False),
                ),
                worker=0,
            )
        sampled = []
        for _ in range(5):
            sampled.append(int(replay.sample(1, "train_behavior")["marker"][0, 0]))
            if with_reports:
                replay.sample(3, "report")
                replay.sample(2, "eval")
        assert len(replay.sampler) == len(replay.behavior_sampler)
        assert len(replay.sampler) == len(replay.report_sampler)
        return sampled

    assert roots(False) == roots(True)


def test_report_transport_cannot_advance_training_batch_counter(monkeypatch):
    from embodied.jax import internal

    monkeypatch.setattr(internal, "device_put", lambda value, _: value)
    monkeypatch.setattr(
        embodied.streams, "Prefetch", lambda source, fn: map(fn, source)
    )
    counters = []
    agent = SimpleNamespace(
        n_batches=elements.Counter(9),
        train_sharded=None,
        train_mirrored=None,
        _seeds=lambda counter, _: counters.append(counter) or np.uint64(counter),
    )
    source = [dict(x=np.ones(2, np.float32)) for _ in range(3)]
    assert len(list(isolated_report_stream(agent, source))) == 3
    assert int(agent.n_batches) == 9
    assert counters == [(1 << 32) + x for x in range(3)]


def test_return_percentiles_ignore_padding_and_empty_batches():
    normalizer = Normalize(
        impl="perc",
        rate=1.0,
        limit=1.0,
        perclo=0.0,
        perchi=100.0,
        debias=False,
        name="normalizer",
    )
    values = jnp.asarray([2.0, 7.0, 9999.0])
    valid = jnp.asarray([True, True, False])

    def update(x, mask):
        return normalizer(x, True, mask)

    state = nj.init(update)({}, values, valid, seed=71)
    state, (_, scale) = nj.pure(update)(state, values, valid, seed=72)
    np.testing.assert_allclose(scale, 5.0)
    after, (_, empty_scale) = nj.pure(update)(
        state,
        values,
        jnp.zeros_like(valid),
        seed=73,
    )
    np.testing.assert_allclose(empty_scale, scale)
    for key in state:
        np.testing.assert_array_equal(state[key], after[key])
