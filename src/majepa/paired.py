"""Fail-closed common-random-number protocol for architecture comparisons."""
import contextlib
import contextvars
import hashlib
import json
import pickle
import types
from pathlib import Path

import jax
import ninjax as nj
import numpy as np
from embodied.jax import internal

_DOMAIN = contextvars.ContextVar('paired_rng_domain', default='learner')
DOMAINS = {'learner': 101, 'collection': 211, 'evaluation': 307,
           'init_policy': 401, 'init_train': 409, 'init_report': 419}

@contextlib.contextmanager
def rng_domain(root, tag, enabled=True):
    if not enabled:
        yield
        return
    ctx = nj.context()
    old_seed, old_reserve = ctx.seed, ctx.reserve
    ctx.seed, ctx.reserve = jax.random.fold_in(root, tag), []
    try:
        yield
    finally:
        ctx.seed, ctx.reserve = old_seed, old_reserve


def install_rng(agent):
    def seeds(self, counter, sharding):
        domain = _DOMAIN.get()
        base = 17001 if domain == 'evaluation' else int(self.config.seed)
        rng = np.random.default_rng([base, DOMAINS[domain], int(counter)])
        values = rng.integers(0, np.iinfo(np.uint32).max, (2,), np.uint32)
        return internal.device_put(values, sharding)
    agent._seeds = types.MethodType(seeds, agent)
    for name in ['policy', 'init_policy', 'init_train', 'init_report']:
        original = getattr(agent, name)
        def call(*args, _name=name, _original=original, **kwargs):
            domain = _name
            if _name == 'policy':
                mode = kwargs.get('mode', args[2] if len(args) > 2 else 'train')
                domain = 'collection' if mode == 'train' else 'evaluation'
            token = _DOMAIN.set(domain)
            try:
                return _original(*args, **kwargs)
            finally:
                _DOMAIN.reset(token)
        setattr(agent, name, call)


def _leaves(tree):
    return {jax.tree_util.keystr(k): np.asarray(v)
            for k, v in jax.tree_util.tree_flatten_with_path(tree)[0]}


def _hash_arrays(data):
    h = hashlib.sha256()
    for key, value in sorted(data.items()):
        a = np.ascontiguousarray(value)
        h.update(json.dumps([key, str(a.dtype), a.shape]).encode())
        h.update(a.tobytes())
    return h.hexdigest()


class PairAudit:
    def __init__(self, agent, args):
        self.phase = str(args.paired_phase)
        self.root = Path(args.paired_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.logdir = Path(args.logdir)
        self.count = 0
        self.raw = hashlib.sha256()
        self.checked = False
        self.prefill = int(args.world_model_start_step)
        assert self.prefill == int(args.ppo_start_step) == 5000
        assert int(args.envs) == 1
        initial = self.root / 'initial.pkl'
        current = agent.save()
        if self.phase == 'canonical':
            assert bool(agent.model.local_outcomes)
            if initial.exists():
                raise RuntimeError('Refusing to overwrite paired initialization')
            with initial.open('wb') as f:
                pickle.dump(current, f, protocol=5)
        else:
            with initial.open('rb') as f:
                canonical = pickle.load(f)
            source = _leaves(canonical['params'])
            def copy(path, value):
                key = jax.tree_util.keystr(path)
                if key not in source or source[key].shape != np.shape(value):
                    raise ValueError(f'Unmatched initialization leaf: {key}')
                assert source[key].dtype == np.asarray(value).dtype, key
                return source[key]
            params = jax.tree_util.tree_map_with_path(copy, current['params'])
            agent.load({'params': params, 'counters': canonical['counters']})
            loaded = _leaves(agent.save()['params'])
            assert all(np.array_equal(v, source[k]) for k, v in loaded.items())
            current = {'params': params}
        self.init_hash = _hash_arrays(_leaves(current['params']))
        (self.logdir / 'paired_initialization.json').write_text(json.dumps({
            'phase': self.phase, 'matched_shared_leaves': self.phase != 'canonical',
            'parameter_leaves': len(_leaves(current['params'])),
            'parameter_sha256': self.init_hash, 'canonical': str(initial),
            'fresh_optimizer': True}, indent=2))

    def transition(self, tran, worker):
        if self.count >= self.prefill:
            return
        data = {k: v for k, v in tran.items()
                if not k.startswith(('log/', 'finite')) and k != 'stepid'}
        self.raw.update(str(worker).encode())
        self.raw.update(_hash_arrays(data).encode())
        self.count += 1

    def verify(self, stream):
        if self.checked:
            return
        from .streams import replay_batch_fingerprints
        assert self.count == self.prefill, self.count
        record = {'records': self.count, 'prefill_sha256': self.raw.hexdigest(),
                  'first_batch': replay_batch_fingerprints(jax.device_get(stream.pending)),
                  'sampled_at': stream.sampled_at}
        target = self.root / 'prefill.json'
        if self.phase == 'canonical':
            assert not target.exists()
            target.write_text(json.dumps(record, indent=2))
            # The canonical run is the authority for both the initialization
            # snapshot and the first replay read.  Mark the pair ready only
            # after both artifacts have been durably written.
            (self.root / 'verified.json').write_text(json.dumps(record, indent=2))
        else:
            expected = json.loads(target.read_text())
            if record != expected:
                (self.logdir / 'paired_mismatch.json').write_text(json.dumps(
                    {'expected': expected, 'actual': record}, indent=2))
                raise RuntimeError('Paired prefill or first replay batch mismatch; training blocked')
        if self.phase == 'train':
            assert (self.root / 'verified.json').exists(), 'Pair preflight not complete'
        elif self.phase == 'verify':
            (self.root / 'verified.json').write_text(json.dumps(record, indent=2))
        (self.logdir / 'paired_prefill_verified.json').write_text(json.dumps(record, indent=2))
        self.checked = True
