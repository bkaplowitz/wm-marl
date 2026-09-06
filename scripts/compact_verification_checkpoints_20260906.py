"""Keep final/resumable checkpoints and compressed 5k model-weight histories.

Only the new verification's primary-map history is managed here. Historical
5k weights retain every non-optimizer tensor exactly, but are explicitly not
optimizer-resumable. The latest checkpoint is never altered.
"""
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import time

import numpy as np

ROOT = Path('/workspace/majepa_interface_verify_20260906')


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def latest(root):
    name = (root / 'latest').read_text().strip()
    checkpoint = root / name
    assert (checkpoint / 'done').exists()
    assert (checkpoint / 'agent.pkl').is_file()
    return checkpoint


def retain(source, ckpt_root):
    if source.parent == latest(ckpt_root):
        return None
    original_stat = source.stat()
    with (source.parent / 'step.pkl').open('rb') as stream:
        step = int(pickle.load(stream))
    if step % 5000 != 0 or step == 0:
        # Superseded periodic saves are temporary; curve saves remain below.
        if source.parent == latest(ckpt_root):
            return None
        marker = dict(kind='superseded_periodic_save', step=step, at=time.time())
        write(source.parent / 'CHECKPOINT_DELETED.json', marker)
        source.unlink()
        return dict(**marker, path=str(source), reclaimed_bytes=original_stat.st_size)
    with source.open('rb') as stream:
        full = pickle.load(stream)
    params = full['params']
    kept = {k: v for k, v in params.items() if not k.startswith('opt/')}
    assert 0 < len(kept) < len(params)
    compact = {**full, 'params': kept, 'checkpoint_kind': 'all_model_weights_only'}
    archive = source.with_suffix('.pkl.gz')
    partial = source.with_suffix('.pkl.gz.partial')
    assert not archive.exists()
    with gzip.open(partial, 'wb', compresslevel=1) as stream:
        pickle.dump(compact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    with gzip.open(partial, 'rb') as stream:
        restored = pickle.load(stream)
    assert restored['params'].keys() == kept.keys()
    assert all(np.array_equal(kept[k], restored['params'][k]) for k in kept)
    assert source.stat().st_mtime_ns == original_stat.st_mtime_ns
    if source.parent == latest(ckpt_root):
        partial.unlink()
        return None
    checksum = hashlib.sha256()
    with partial.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    partial.replace(archive)
    record = dict(kind='all_model_weights_only', step=step, path=str(source),
        archive=str(archive), sha256=checksum.hexdigest(), at=time.time(),
        retained_tensors=len(kept), tensor_equality_verified=True,
        original_bytes=original_stat.st_size, compressed_bytes=archive.stat().st_size,
        reclaimed_bytes=original_stat.st_size-archive.stat().st_size,
        restore='gzip -dc agent.pkl.gz > agent.pkl; optimizer state is absent')
    write(source.parent / 'WEIGHTS_ONLY.json', record)
    source.unlink()
    return record


def main():
    lock = (ROOT / 'checkpoint_retention.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = ROOT / 'checkpoint_retention.json'
    state = (json.loads(path.read_text()) if path.exists()
             else dict(created_at=time.time(), records=[], errors=[]))
    state.update(pid=os.getpid(), phase='running', updated_at=time.time(),
                 source=__file__, final_checkpoints_untouched=True)
    write(path, state)
    stopping = False

    def stop(_signal, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not stopping:
            queue = json.loads((ROOT / 'queue.json').read_text())
            if time.time() >= queue['decision_deadline']:
                break
            jobs = [j for j in queue['jobs'] if j['kind'] == 'online']
            for job in jobs:
                if job['map'] != '3s_vs_3z':
                    continue
                folder = ROOT / 'runs' / job['name'] / 'train/run/ckpt'
                if not (folder / 'latest').exists():
                    continue
                for done in sorted(folder.glob('*/done')):
                    source = done.parent / 'agent.pkl'
                    if not source.exists():
                        continue
                    record = retain(source, folder)
                    if record:
                        state['records'].append(record)
                        print(json.dumps(record), flush=True)
                    state['updated_at'] = time.time()
                    write(path, state)
            state['updated_at'] = time.time()
            write(path, state)
            if jobs and all(j['status'] in ('complete', 'failed', 'cancelled') for j in jobs):
                break
            time.sleep(30)
    except Exception as error:
        state['errors'].append(dict(at=time.time(), error=repr(error)))
        state['phase'] = 'failed'
        raise
    finally:
        if state['phase'] != 'failed':
            state['phase'] = 'complete' if not stopping else 'stopped'
        state['updated_at'] = time.time()
        write(path, state)
        lock.close()


if __name__ == '__main__':
    main()
