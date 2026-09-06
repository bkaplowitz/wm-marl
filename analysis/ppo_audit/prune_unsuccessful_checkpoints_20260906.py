"""Delete explicitly abandoned/poor checkpoints after the user's authorization.

Preserve all logs, replay, metrics, final evaluations, strong reference models,
and the five original inputs used by the active simulator comparisons.
"""
import hashlib
import json
from pathlib import Path
import os
import time

BASE = Path('/workspace')
DEST = BASE / 'majepa_interface_verify_20260906/checkpoint_cleanup.json'
SELECTED = {
    'majepa_bptt2_matrix_20260905': [
        'bptt2-3s_vs_4z-seed0', 'bptt2-3s_vs_4z-seed1', 'bptt2-3s_vs_4z-seed2',
        'bptt2-3s_vs_5z-seed1', 'bptt2-5m_vs_6m-seed0', 'bptt2-5m_vs_6m-seed1',
        'bptt2-5m_vs_6m-seed2', 'bptt2-MMM2-seed0', 'bptt2-MMM2-seed1',
        'bptt2-MMM2-seed2', 'bptt2-corridor-seed0', 'bptt2-corridor-seed2'],
    'majepa_value_sweep_20260906': [
        'actor1-2s3z-seed0', 'entlow-2s3z-seed0', 'fact-2s3z-seed0',
        'factrep-2s3z-seed0', 'ret-ent-2s3z-seed0', 'retperc-2s3z-seed0',
        'fact-3s_vs_3z-seed0', 'fact-vreg-2s3z-seed0',
        'frep-ent-2s3z-seed0', 'vreg-2s3z-seed0'],
    'majepa_ppo_correction_2663ae5_20260905': [
        'original_base-2s3z-seed1', 'team_return-2s3z-seed0',
        'team_return-2s3z-seed1', 'team_return_anchor-2s3z-seed1'],
    'majepa_prefill_recurrent_ema_20260905': ['prefill_ema-2s3z-seed0', 'prefill_ema-8m-seed0'],
    'majepa_recurrent_capacity_20260905': [
        'recurrent_actor2x256-2s3z-seed0', 'recurrent_wmlr1e4-2s3z-seed0'],
    'majepa_recurrent_combinations_20260905': ['recurrent_fresh_bptt2-2s3z-seed0'],
    'majepa_recurrent_env8_20260905': ['recurrent_env8-2s3z-seed0'],
    'majepa_recurrent_extensions_20260905': [
        'recurrent_ema05-2s3z-seed0', 'recurrent_fresh_history-2s3z-seed0',
        'recurrent_env16-2s3z-seed0'],
}


def write(value):
    tmp = DEST.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(DEST)


def main():
    assert not DEST.exists(), 'Preserve the prior cleanup manifest'
    protected = [BASE / 'majepa_bptt2_matrix_20260905/runs' / f'bptt2-{m}-seed{s}'
                 for m, s in [('3s_vs_3z', 0), ('3s_vs_3z', 1), ('3s_vs_3z', 2),
                              ('3s_vs_5z', 0), ('3s_vs_5z', 2)]]
    targets = []
    selected_runs = []
    for root, names in SELECTED.items():
        for name in names:
            run = BASE / root / 'runs' / name
            assert run not in protected
            selected_runs.append(str(run))
            targets.extend(run.glob('train/run/ckpt/*/agent.pkl'))
    for command in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            text = command.read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        assert not any(run in text for run in selected_runs), f'Checkpoint still in use: {command}'
    # Remove all hard-linked copies of these same checkpoint inodes, otherwise
    # deleting a run's copy alone does not reclaim the quota.
    inodes = {(p.stat().st_dev, p.stat().st_ino) for p in targets}
    aliases = {}
    for root in BASE.glob('majepa*'):
        for folder, _, files in os.walk(root):
            if 'agent.pkl' not in files:
                continue
            p = Path(folder) / 'agent.pkl'
            stat = p.stat()
            key = (stat.st_dev, stat.st_ino)
            if key in inodes:
                assert not any(p.is_relative_to(r) for r in protected)
                aliases.setdefault(key, []).append(p)
    state = dict(authorized_by='User: just delete checkpoints of attempts that were unsuccessful anyways',
                 started_at=time.time(), phase='deleting', reclaimed_bytes=0,
                 protected_inputs=list(map(str, protected)), files=[])
    write(state)
    for paths in aliases.values():
        stat = paths[0].stat()
        assert len(paths) == stat.st_nlink, 'Unaccounted checkpoint hard link'
        digest = hashlib.sha256()
        with paths[0].open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        record = dict(paths=list(map(str, paths)), bytes=stat.st_size,
                      sha256=digest.hexdigest(), deleted=False)
        state['files'].append(record)
        write(state)
        for path in paths:
            assert path.stat().st_ino == stat.st_ino
            marker = path.parent / 'CHECKPOINT_DELETED.json'
            marker.write_text(json.dumps(dict(reason=state['authorized_by'], manifest=str(DEST),
                sha256=record['sha256'], at=time.time(), metrics_and_replay_preserved=True), indent=2)+'\n')
            path.unlink()
        record['deleted'] = True
        state['reclaimed_bytes'] += stat.st_size
        write(state)
        print(json.dumps(dict(deleted=str(paths[0]), reclaimed_bytes=state['reclaimed_bytes'])), flush=True)
    for run in protected:
        assert any(run.glob('train/run/ckpt/*/agent.pkl'))
    state.update(phase='complete', finished_at=time.time())
    write(state)


if __name__ == '__main__':
    main()
