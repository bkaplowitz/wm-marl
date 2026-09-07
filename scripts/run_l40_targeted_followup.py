#!/usr/bin/env python3
"""One owned queue per L40; preserve failed attempts and the existing deadline."""
from dataclasses import asdict, dataclass
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import run_coverage_followup as coverage
import run_ppo_correction_screen as base

ROOT = Path('/workspace/majepa_targeted_l40_20260907')
SOURCE = Path(__file__).resolve().parents[1]
GROUP = 'ma-jepa-targeted-diagnostics-20260907'


@dataclass(frozen=True)
class Spec(coverage.RunSpec):
    factual: bool = False
    representation_scale: float = 0.0

    @property
    def configuration_flags(self):
        return super().configuration_flags + (
            '--jax.prealloc', 'True',
            '--agent.ppo.factual_value.enabled', str(self.factual),
            '--agent.ppo.factual_value.representation_scale', str(self.representation_scale),
            '--agent.ppo.factual_value.rho_clip', '1.0',
            '--agent.ppo.factual_value.c_clip', '1.0')


def make_spec(arm, map_name, seed):
    values = asdict(coverage.spec('mix', 'align', map_name, seed))
    return Spec(**{**values, 'arm': arm, 'factual': arm == 'fact' or arm == 'factrep',
                   'representation_scale': 0.1 if arm == 'factrep' else 0.0})


def job_args(run, gpu, root):
    args = coverage.job_args(run, gpu, root)
    args.wandb_group = GROUP if gpu == 3 else coverage.GROUP
    args.wandb_run_prefix = args.wandb_job_id = f'l40d-{run.arm}-{run.map_name}-s{run.seed}'
    args.screen_label = 'L40 targeted follow-up, unchanged alignment + mixed replay base'
    args.portserver_address = f'@majepa-l40-targeted-g{gpu}'
    args.portserver_pool = f'{58000 + gpu * 450}-{58449 + gpu * 450}'
    return args


def train_command(args, run, logdir):
    command = coverage.train_command(args, run, logdir)
    if args.gpu == 3:
        command[command.index('--run.checkpoint_at_curve_eval') + 1] = 'True'
    return command


def initialize(root):
    previous = json.loads(Path('/workspace/majepa_mmm_confirmation_l40_20260907/queue.json').read_text())
    root.mkdir(exist_ok=False)
    (root / 'workers').mkdir()
    jobs = []
    for seed in range(3):
        run = make_spec('mem95', 'MMM', seed)
        jobs.append(dict(index=len(jobs), kind='train', gpu=seed, spec=asdict(run),
                         name=run.name, status='pending'))
    # GPU 3 executes the diagnostics before claiming either value intervention.
    jobs.append(dict(index=len(jobs), kind='diagnostics', gpu=3, name='frozen-diagnostics',
                     status='pending', command=[]))
    for map_name in ('3s_vs_4z', '2s3z'):
        for arm in ('control', 'fact', 'factrep'):
            run = make_spec(arm, map_name, 0)
            jobs.append(dict(index=len(jobs), kind='train', gpu=3, spec=asdict(run),
                             name=run.name, status='pending'))
    base.atomic_json(root / 'queue.json', dict(jobs=jobs, created_at=time.time(),
        claim_deadline=previous['claim_deadline'], decision_deadline=previous['decision_deadline'],
        source=str(SOURCE), group=GROUP, gpu_count=4,
        memory_only_change=dict(prealloc=True, memory_fraction=0.95),
        comparison='Each map has a fresh control, factual target, and factual + representation (0.1), seed 0; screening only.'))


def state_change(root, index, **values):
    with (root / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads((root / 'queue.json').read_text())
        state['jobs'][index].update(values)
        base.atomic_json(root / 'queue.json', state)


def job(root, index, gpu):
    entry = json.loads((root / 'queue.json').read_text())['jobs'][index]
    run = Spec(**entry['spec'])
    args = job_args(run, gpu, root)
    base.train_command = train_command

    def profile(_args, _run, _env):
        import elements
        from majepa.main import _load_configs, _resolve_config_profiles
        result = {}
        for phase, command in (
            ('train', train_command(args, run, Path('unused'))),
            ('final100', base.eval_command(args, run, Path('unused'), Path('checkpoint')))):
            parsed, other = elements.Flags(configs=['smac_vector', 'ma_jepa']).parse_known(command[3:])
            config = elements.Flags(_resolve_config_profiles(_load_configs(), parsed.configs)).parse(other)
            assert config.agent.marl.ctde.self_fed.trajectory_kl_scale == 0.1
            assert config.agent.marl.ctde.self_fed.bptt_steps == 2
            assert config.replay.world_uniform_mix == 0.5
            assert config.agent.ppo.factual_value.enabled == run.factual
            assert config.agent.ppo.factual_value.representation_scale == run.representation_scale
            if phase == 'train' and gpu == 3:
                assert config.run.checkpoint_at_curve_eval
            result[phase] = dict(command=command, config=config.flat)
        return result

    base.run_screen(args, run, profile, extra_manifest=dict(
        memory_fraction=os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'],
        previous_attempt='MMM first update exceeded default L40 allocator pool',
        controlled_changes=dict(factual=run.factual, representation_scale=run.representation_scale)))
    if gpu == 3:
        import pickle
        train_root = root / 'runs' / run.name / 'train'
        final = base.latest_checkpoint(train_root, expected_steps=run.steps)
        common = [str(coverage.PYTHON), str(SOURCE / 'scripts/diagnose_policy_interface.py'),
            '--config', str(train_root / 'run/config.yaml')]
        prefix = f'l40d-{run.arm}-{run.map_name}-s{run.seed}'
        diagnostics = [(prefix + '-rank', [
            '--checkpoint', str(final), '--label', f'{run.name}, final 50k model'])]
        if run.arm == 'control':
            completed = []
            for done in final.parent.glob('*/done'):
                with (done.parent / 'step.pkl').open('rb') as stream:
                    step = int(pickle.load(stream))
                if 0 < step < run.steps and (done.parent / 'agent.pkl').exists():
                    completed.append((step, done.parent))
            if not completed:
                raise RuntimeError('Control is missing the queued temporal drift checkpoint')
            step, older = max(completed)
            diagnostics.append((prefix + '-drift', [
                '--checkpoint', str(older), '--compare-checkpoint', str(final), '--drift-only',
                '--label', f'Current combo temporal policy drift: {step} to {run.steps}']))
        for name, flags in diagnostics:
            command = common + flags + ['--output', str(root / 'diagnostics' / name),
                '--wandb-id', name]
            base.atomic_json(root / 'workers' / f'{name}.command.json', dict(command=command))
            env = base.execution_environment(args)
            # Frozen inference does not need the learner's 95% reservation.
            env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
            with (root / 'workers' / f'{name}.log').open('x') as stream:
                subprocess.run(command, cwd=SOURCE, env=env, stdout=stream,
                               stderr=stream, check=True)


def worker(root, gpu):
    lock = (root / 'workers' / f'gpu{gpu}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root / 'workers' / f'gpu{gpu}.pid').write_text(str(os.getpid()))
    children = base.OwnedChildren()
    def stop(_signal, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while True:
            state = json.loads((root / 'queue.json').read_text())
            if time.time() >= state['claim_deadline']:
                return
            pending = [j for j in state['jobs'] if j['gpu'] == gpu and j['status'] == 'pending']
            if not pending:
                return
            entry = pending[0]
            if base.gpu_processes(gpu) or (entry['kind'] == 'diagnostics' and not entry['command']):
                time.sleep(15)
                continue
            command = entry.get('command') or [str(coverage.PYTHON), str(Path(__file__).resolve()),
                'job', '--root', str(root), '--gpu', str(gpu), '--index', str(entry['index'])]
            state_change(root, entry['index'], status='running', worker_pid=os.getpid(), started_at=time.time())
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), XLA_PYTHON_CLIENT_MEM_FRACTION='0.95',
                OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', PYTHONUNBUFFERED='1',
                PYTHONPATH=f'{SOURCE}/src:/workspace/external/dreamerv3',
                PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python', SC2PATH='/workspace/StarCraftII')
            with (root / 'workers' / f'{entry["name"]}.log').open('x') as stream:
                process = children.start(command, cwd=SOURCE, env=env, stdout=stream, stderr=stream)
                state_change(root, entry['index'], child_pid=process.pid)
                result = process.wait()
            state_change(root, entry['index'], status='complete' if result == 0 else 'failed',
                         exit_code=result, finished_at=time.time())
            if result:
                # No blind retries, and no value treatments after failed diagnostics.
                return
    finally:
        children.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('init', 'worker', 'job'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--gpu', type=int, choices=range(4), default=3)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    if args.mode == 'init':
        initialize(args.root)
    elif args.mode == 'worker':
        worker(args.root, args.gpu)
    else:
        job(args.root, args.index, args.gpu)
