"""Use actual replay/selectors/transport to verify deterministic read ordering."""
import time

import elements
import embodied
import numpy as np

from majepa.replay import DualViewReplay
from majepa.streams import (
    ReplaySnapshotStream, ReplaySnapshotTrace, replay_batch_fingerprints,
    synchronous_report_stream,
)
from majepa.train import _with_prefixed_batch


class Agent:
    def __init__(self):
        self.n_batches = elements.Counter()
        self.train_sharded = self.train_mirrored = None

    def _seeds(self, counter, sharding):
        return np.random.default_rng([2, int(counter)]).integers(
            0, np.iinfo(np.uint32).max, (2,), np.uint32)


class Replay(DualViewReplay):
    def __init__(self):
        super().__init__(length=256, optimized_length=64, chunksize=300,
                         capacity=10000, seed=2, world_uniform_mix=0.5,
                         isolate_report_rng=True)
        self.records = 0
        self.reads = []

    def populate(self, count):
        while self.records < count:
            record = self.records
            self.add({'observation': np.array([record], np.float32),
                      'reward': np.float32(record % 11), 'action': np.int32(record % 7),
                      'is_first': np.bool_(record % 100 == 0),
                      'is_last': np.bool_(record % 100 == 99),
                      'is_terminal': np.bool_(False),
                      'dyn/pair': np.array([0], np.float32),
                      'dyn/stoch': np.array([0], np.float32)})
            self.records += 1

    def sample(self, batch, mode='train'):
        value = super().sample(batch, mode)
        self.reads.append((mode, self.records, len(self)))
        return value

    def writeback(self, version):
        # Include the real cross-chunk step-ID path, not mutation of a mock.
        for chunk in list(self.chunks.values()):
            if chunk.length:
                stepids = chunk.slice(0, chunk.length)['stepid'].copy()[None]
                values = np.full((1, chunk.length, 1), version, np.float32)
                self.update({'stepid': stepids, 'dyn/pair': values, 'dyn/stoch': values})


def paired(replay, report=False):
    def source(mode):
        return embodied.streams.Consec(
            embodied.streams.Stateless(lambda: replay.sample(16, mode)),
            length=64, consec=1, prefix=192, strict=True, contiguous=True)
    return _with_prefixed_batch(
        source('report' if report else 'train_world'),
        source('report' if report else 'train_behavior'), '_behavior_replay/')


def test_preserves_eager_first_window_and_read_before_writeback():
    replay, agent = Replay(), Agent()
    stream = ReplaySnapshotStream(agent, paired(replay))
    for record in range(1, 5001):
        replay.populate(record)
        if len(replay):
            stream.prime(record)
    assert replay.reads == [('train_world', 256, 1), ('train_behavior', 256, 1)]
    assert int(agent.n_batches) == 1

    first, selected_at = stream.take(5000)
    assert selected_at == 256
    for prefix in ['', '_behavior_replay/']:
        np.testing.assert_array_equal(np.asarray(first[prefix + 'observation'])[:, 0, 0], 0)
    assert replay.reads[-2:] == [('train_world', 5000, 4745), ('train_behavior', 5000, 4745)]

    replay.writeback(1)
    replay.populate(5008)
    second, selected_at = stream.take(5008)
    assert selected_at == 5000
    for prefix in ['', '_behavior_replay/']:
        assert np.all(np.asarray(second[prefix + 'dyn/stoch']) == 0)
    third, selected_at = stream.take(5016)
    assert selected_at == 5008
    # New records have initial history0; replay write-back reached older records.
    assert np.any(np.asarray(third['dyn/stoch']) == 1)
    assert int(agent.n_batches) == 4


def test_reports_delays_and_trace_do_not_change_training_batches(tmp_path):
    def execute(extra_reports):
        replay, agent = Replay(), Agent()
        trace = ReplaySnapshotTrace(tmp_path / f'trace{extra_reports}.jsonl', 3)
        stream = ReplaySnapshotStream(agent, paired(replay), trace)
        report = iter(synchronous_report_stream(agent, paired(replay, report=True)))
        replay.populate(256)
        stream.prime(256)
        replay.populate(5000)
        result = []
        for update in range(8):
            step = 5000 + 8 * update
            replay.populate(step)
            if extra_reports:
                next(report)
                time.sleep(0.001)
            data, sampled_at = stream.take(step)
            result.append((sampled_at, replay_batch_fingerprints(data),
                           np.asarray(data['seed']).tolist()))
            replay.writeback(update + 1)
        assert int(agent.n_batches) == 9
        return result, [r for r in replay.reads if r[0] != 'report']

    assert execute(False) == execute(True)
    assert (tmp_path / 'traceFalse.jsonl').read_text() == (tmp_path / 'traceTrue.jsonl').read_text()
    assert len((tmp_path / 'traceFalse.jsonl').read_text().splitlines()) == 3


def test_fingerprints_ignore_only_storage_ids_and_detect_history_changes():
    data = {'observation': np.array([[1, 2]], np.float32),
            'dyn/stoch': np.array([[3, 4]], np.float32),
            'stepid': np.zeros((1, 2, 20), np.uint8)}
    initial = replay_batch_fingerprints(data)
    data['stepid'] += 1
    assert replay_batch_fingerprints(data) == initial
    data['dyn/stoch'][0, 0] += 1
    changed = replay_batch_fingerprints(data)
    assert changed['world']['data_sha256'] == initial['world']['data_sha256']
    assert changed['world']['history_sha256'] != initial['world']['history_sha256']


def test_matches_actual_legacy_transport_with_serialized_prefetch_reads():
    from embodied.jax.agent import Agent as JaxAgent

    old_replay, new_replay = Replay(), Replay()
    old_agent, new_agent = Agent(), Agent()
    old_replay.populate(256)
    new_replay.populate(256)
    legacy = iter(JaxAgent.stream(old_agent, paired(old_replay)))
    fixed = ReplaySnapshotStream(new_agent, paired(new_replay))
    fixed.prime(256)
    # Take the queue item directly so the test, rather than the background
    # scheduler, decides when the following legacy read is allowed to happen.
    previous, _ = legacy.queue.get(timeout=5)
    try:
        for update in range(8):
            step = 5000 + 8 * update
            old_replay.populate(step)
            new_replay.populate(step)
            legacy.requests.release()
            future, _ = legacy.queue.get(timeout=5)
            current, _ = fixed.take(step)
            assert replay_batch_fingerprints(previous) == replay_batch_fingerprints(current)
            np.testing.assert_array_equal(previous['seed'], current['seed'])
            assert int(old_agent.n_batches) == int(new_agent.n_batches)
            old_replay.writeback(update + 1)
            new_replay.writeback(update + 1)
            previous = future
    finally:
        legacy.worker.kill(timeout=0)
        legacy.requests.release()
        legacy.worker.join(timeout=5)
