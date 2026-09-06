#!/usr/bin/env python3
"""Bounded replay-coverage and reporting-RNG follow-up after the active matrix."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import fcntl
import json
import os
from pathlib import Path
import signal
import time

import run_interface_verification as reference
import run_ppo_correction_screen as base
from compact_verification_checkpoints_20260906 import retain

SOURCE = Path(__file__).resolve().parents[1]
ROOT = Path('/workspace/majepa_coverage_followup_20260906')
DEPENDENCY = Path('/workspace/majepa_interface_verify_20260906')
GROUP = 'ma-jepa-coverage-followup-20260906'
PYTHON = Path('/workspace/majepa-runtime/bin/python')


@dataclass(frozen=True)
class RunSpec(reference.RunSpec):
    world_uniform_mix: float = 0.5
    isolate_report_rng: bool = True
    reference_arm: str = 'base'

    @property
    def configuration_flags(self):
        flags = list(super().configuration_flags)
        flags[flags.index('--run.isolate_report_rng') + 1] = str(self.isolate_report_rng)
        return tuple(flags) + ('--replay.world_uniform_mix', str(self.world_uniform_mix))


def spec(kind, arm, map_name, seed):
    values = asdict(reference.run_spec(arm, map_name, seed))
    return RunSpec(**{**values, 'arm': f'{kind}-{arm}', 'reference_arm': arm,
        'world_uniform_mix': 0.5 if kind == 'mix' else 0.0,
        'isolate_report_rng': kind != 'reportoff'})


def run_id(run):
    return f'cv2-{run.arm}-{run.map_name}-s{run.seed}'


def job_args(run, gpu, root):
    args = reference.job_args(run, gpu, root)
    args.source = SOURCE
    args.wandb_group = GROUP
    args.wandb_run_prefix = args.wandb_job_id = run_id(run)
    args.screen_label = 'Replay coverage and reporting RNG follow-up'
    args.portserver_address = f'@majepa-coverage-followup-g{gpu}'
    args.portserver_pool = f'{61000 + gpu * 450}-{61449 + gpu * 450}'
    return args


def train_command(args, run, logdir):
    command = reference.train_command(args, run, logdir)
    key = command.index('--logger.filter') + 1
    command[key] += '|replay/world_uniform_'
    return command


def resolve(run, root):
    import elements
    from majepa.main import _load_configs, _resolve_config_profiles
    from smac.env.starcraft2.maps import get_map_params

    assert get_map_params(run.map_name)['n_agents'] == run.num_agents
    original = json.loads((DEPENDENCY / 'resolved.json').read_text())
    original = original[f'{run.reference_arm}-{run.map_name}-seed{run.seed}']
    args = job_args(run, 0, root)
    commands = {'train': train_command(args, run, Path('unused')),
        'final100': base.eval_command(args, run, Path('unused'), Path('checkpoint'))}
    result = {}
    for phase, command in commands.items():
        parsed, rest = elements.Flags(configs=['smac_vector', 'ma_jepa']).parse_known(command[3:])
        config = elements.Flags(_resolve_config_profiles(_load_configs(), parsed.configs)).parse(rest)
        actual = json.loads(json.dumps(config.flat))
        expected = dict(original[phase]['config'])
        expected['replay.world_uniform_mix'] = run.world_uniform_mix
        expected['run.isolate_report_rng'] = run.isolate_report_rng
        if phase == 'train':
            expected['logger.filter'] += '|replay/world_uniform_'
        differences = {k: (expected.get(k, '<missing>'), actual.get(k, '<missing>'))
            for k in set(expected) | set(actual) if expected.get(k) != actual.get(k)}
        if differences:
            raise ValueError(f'Unintended change relative to current paired control: {differences}')
        result[phase] = dict(config=actual, command=command,
            controlled_changes=dict(world_uniform_mix=run.world_uniform_mix,
                isolate_report_rng=run.isolate_report_rng), reference_arm=run.reference_arm)
    return result


def initialize(root):
    expected = (SOURCE / 'SOURCE_SHA256').read_text().strip()
    assert base.source_fingerprint(SOURCE) == expected
    previous = json.loads((DEPENDENCY / 'queue.json').read_text())
    jobs, resolved = [], {}
    for kind, maps in [('mix', ['3s_vs_4z', '2s3z', '3s_vs_3z']),
                       ('reportoff', ['3s_vs_3z'])]:
        for map_name in maps:
            for seed in (0, 1, 2):
                for arm in ('base', 'align'):
                    run = spec(kind, arm, map_name, seed)
                    resolved[run.name] = resolve(run, root)
                    jobs.append(dict(index=len(jobs), name=run.name, kind=kind,
                        map=map_name, seed=seed, reference_arm=arm, status='pending',
                        spec=asdict(run), wandb=f'{reference.URL}/runs/{run_id(run)}-train',
                        wandb_final=f'{reference.URL}/runs/{run_id(run)}-final100'))
    assert len(jobs) == 24
    root.mkdir(exist_ok=False)
    (root / 'workers').mkdir()
    base.atomic_json(root / 'resolved.json', resolved)
    base.atomic_json(root / 'queue.json', dict(created_at=time.time(), jobs=jobs,
        phase='waiting_for_current_matrix', dependency=str(DEPENDENCY), source=str(SOURCE),
        source_sha256=expected, maximum_jobs=24, extra_environment_transitions=1200000,
        claim_deadline=previous['claim_deadline'], decision_deadline=previous['decision_deadline'],
        group=GROUP, final_checkpoint_policy='Every completed model retained, weights only after final100'))
    print(json.dumps(dict(initialized=len(jobs), root=str(root), group=GROUP)), flush=True)


def prune_obsolete(root, job):
    """Keep the latest resumable checkpoint; retain final weights after eval.

    Saving cadence matches the controls. Only superseded artifacts are pruned;
    no checkpoint currently used by training/final evaluation is changed.
    """
    folder = root / 'runs' / job['name'] / 'train/run/ckpt'
    pointer = folder / 'latest'
    if not pointer.exists():
        return
    current = folder / pointer.read_text().strip()
    if not (current / 'done').exists():
        return
    for done in sorted(folder.glob('*/done')):
        path = done.parent / 'agent.pkl'
        # Checkpoint names are ordered UTC timestamps. A newly completed save
        # can appear before the writer advances latest; never prune that save.
        if not path.exists() or done.parent.name >= current.name:
            continue
        if done.parent.name == pointer.read_text().strip():
            continue
        record = dict(path=str(path), bytes=path.stat().st_size, at=time.time(),
            reason='Predeclared latest/final-only storage; newer completed checkpoint exists')
        base.atomic_json(done.parent / 'CHECKPOINT_DELETED.json', record)
        path.unlink()


def worker(root, gpu):
    lock = (root / 'workers' / f'gpu{gpu}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root / 'workers' / f'gpu{gpu}.pid').write_text(str(os.getpid()) + '\n')
    children = base.OwnedChildren()

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        while True:
            snapshot = json.loads((root / 'queue.json').read_text())
            if time.time() >= snapshot['claim_deadline'] or snapshot.get('claims_closed'):
                return
            dependency = json.loads((DEPENDENCY / 'queue.json').read_text())
            if any(j['status'] != 'complete' for j in dependency['jobs']):
                base.atomic_json(root / 'workers' / f'gpu{gpu}.status.json',
                    dict(phase='waiting_for_current_matrix', updated_at=time.time(), pid=os.getpid()))
                time.sleep(30)
                continue
            if base.gpu_processes(gpu):
                time.sleep(15)
                continue
            with reference.state_lock(root) as state:
                if time.time() >= state['claim_deadline'] or state.get('claims_closed'):
                    return
                pending = [j for j in state['jobs'] if j['status'] == 'pending']
                if not pending:
                    if not any(j['status'] == 'running' for j in state['jobs']):
                        state['phase'] = 'complete'
                    return
                job = pending[0]
                job.update(status='running', gpu=gpu, worker_pid=os.getpid(), started_at=time.time())
                state['phase'] = 'running'
                job = dict(job)
            base.atomic_json(root / 'workers' / f'gpu{gpu}.status.json',
                dict(phase='running', name=job['name'], updated_at=time.time(), pid=os.getpid()))
            try:
                assert base.source_fingerprint(SOURCE) == (SOURCE / 'SOURCE_SHA256').read_text().strip()
                with (root / 'workers' / f'{job["name"]}.log').open('x') as stream:
                    p = children.start([str(PYTHON), str(Path(__file__).resolve()), 'job',
                        '--root', str(root), '--index', str(job['index']), '--gpu', str(gpu)],
                        cwd=SOURCE, env=os.environ.copy(), stdout=stream, stderr=stream)
                    while p.poll() is None:
                        prune_obsolete(root, job)
                        time.sleep(20)
                    if p.returncode:
                        raise RuntimeError(f'Training/final100 supervisor exited {p.returncode}')
                outcome = json.loads((root / 'runs' / job['name'] / 'outcome.json').read_text())
                assert outcome['completed'] and outcome['summary']['evaluation_protocol']['episodes'] == 100
                prune_obsolete(root, job)
                checkpoint = Path(outcome['checkpoint'])
                retention = retain(checkpoint / 'agent.pkl', checkpoint.parent, completed_final=True)
                base.atomic_json(root / 'runs' / job['name'] / 'final_weight_retention.json', retention)
                with reference.state_lock(root) as state:
                    state['jobs'][job['index']].update(status='complete', finished_at=time.time(),
                        final_wr=outcome['summary']['win_rate'], retained_weights=retention['archive'])
            except Exception as error:
                with reference.state_lock(root) as state:
                    state['jobs'][job['index']].update(status='failed', error=repr(error), finished_at=time.time())
                raise
    finally:
        children.close()
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('init', 'worker', 'job'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--gpu', type=int, choices=range(6), default=0)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    if args.mode == 'init':
        initialize(args.root)
    elif args.mode == 'worker':
        worker(args.root, args.gpu)
    else:
        state = json.loads((args.root / 'queue.json').read_text())
        job = state['jobs'][args.index]
        run = RunSpec(**job['spec'])
        base.train_command = train_command

        def profile(_args, _run, _env):
            return json.loads((args.root / 'resolved.json').read_text())[run.name]

        base.run_screen(job_args(run, args.gpu, args.root), run, profile,
            extra_manifest=dict(followup=GROUP, dependency=str(DEPENDENCY), fresh_online=True,
                imported_experience=False, predeclared_changes=job['kind']))


if __name__ == '__main__':
    main()
