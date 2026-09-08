#!/usr/bin/env python3
"""Bounded factual-value retest on the frozen sampled-availability foundation."""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import pickle
import subprocess
import time

import run_sampled_availability as sampled
import run_l40_targeted_followup as target
import run_ppo_correction_screen as base

SOURCE = Path(__file__).resolve().parents[1]
ROOT = Path('/workspace/majepa_value_grounding_retest_20260908')
REFERENCE = Path('/workspace/majepa_sampled_availability_20260907')
GROUP = 'ma-jepa-value-grounding-retest-20260908'
PYTHON = sampled.PYTHON


def spec(arm, map_name, seed):
    assert arm in ('fact', 'factrep')
    return sampled.Spec(**{**asdict(sampled.spec('bernoulli', map_name, seed)),
        'arm': arm, 'factual': True,
        'representation_scale': .1 if arm == 'factrep' else 0.0})


def run_id(run):
    return f'vg1-{run.arm}-{run.map_name}-s{run.seed}'


def job_args(run, gpu, root):
    args = sampled.job_args(run, gpu, root)
    args.wandb_group = GROUP
    args.wandb_run_prefix = args.wandb_job_id = run_id(run)
    args.screen_label = 'Factual value retest on BPTT2, alignment, mixed replay and sampled availability'
    args.portserver_address = f'@majepa-value-retest-g{gpu}'
    args.portserver_pool = f'{51000 + gpu * 450}-{51449 + gpu * 450}'
    return args


def resolve(run, gpu, root):
    import elements
    from majepa.main import _load_configs, _resolve_config_profiles
    from smac.env.starcraft2.maps import get_map_params
    assert get_map_params(run.map_name)['n_agents'] == run.num_agents
    args = job_args(run, gpu, root)
    commands = [('train', sampled.train_command(args, run, Path('unused'))),
        ('final100', base.eval_command(args, run, Path('unused'), Path('checkpoint')))]
    reference_name = sampled.spec('bernoulli', run.map_name, run.seed).name
    reference = json.loads((REFERENCE / 'resolved.json').read_text())[reference_name]
    changes = {'agent.ppo.factual_value.enabled': [False, True]}
    if run.representation_scale:
        changes['agent.ppo.factual_value.representation_scale'] = [0.0, run.representation_scale]
    result = {}
    for phase, command in commands:
        parsed, rest = elements.Flags(configs=['smac_vector', 'ma_jepa']).parse_known(command[3:])
        c = elements.Flags(_resolve_config_profiles(_load_configs(), parsed.configs)).parse(rest)
        before, after = reference[phase]['config'], c.flat
        differences = {k: [before.get(k), after.get(k)] for k in set(before) | set(after)
            if before.get(k) != after.get(k)}
        assert differences == changes, differences
        result[phase] = dict(command=command, config=after, controlled_changes=differences)
    return result


def initialize(root):
    previous = json.loads((REFERENCE / 'queue.json').read_text())
    fingerprint = base.source_fingerprint(SOURCE)
    assert fingerprint == previous['source_sha256']
    assert fingerprint == (SOURCE / 'SOURCE_SHA256').read_text().strip()
    jobs, resolved, references = [], {}, {}
    for seed in (0, 1):
        for gpu, (map_name, arm) in enumerate([
                ('2s3z', 'fact'), ('2s3z', 'factrep'),
                ('3s_vs_4z', 'fact'), ('3s_vs_4z', 'factrep')]):
            run = spec(arm, map_name, seed)
            reference_name = sampled.spec('bernoulli', map_name, seed).name
            reference_job = next(j for j in previous['jobs'] if j['name'] == reference_name)
            assert reference_job['status'] == 'complete'
            folder = REFERENCE / 'runs' / reference_name
            summary = json.loads((folder / 'final100/run/evaluation_summary.json').read_text())
            budget = json.loads((folder / 'budget_verified.json').read_text())
            assert summary['episodes'] == 100 and summary['policy_mode'] == 'eval'
            assert budget == {'environment_steps': 50000, 'learner_updates': 5626}
            references[reference_name] = dict(wandb=reference_job['wandb'],
                wins=summary['wins'], episodes=100, budget=budget,
                source_sha256=previous['source_sha256'])
            resolved[run.name] = resolve(run, gpu, root)
            index = len(jobs)
            jobs.append(dict(index=index, gpu=gpu, name=run.name, kind='train',
                status='pending', spec=asdict(run), reference=reference_name,
                wandb=f'https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/{run_id(run)}-train',
                command=[str(PYTHON), str(Path(__file__).resolve()), 'job',
                    '--root', str(root), '--gpu', str(gpu), '--index', str(index)]))
    root.mkdir(exist_ok=False)
    (root / 'workers').mkdir()
    base.atomic_json(root / 'resolved.json', resolved)
    base.atomic_json(root / 'references.json', references)
    now = time.time()
    base.atomic_json(root / 'queue.json', dict(jobs=jobs, created_at=now, group=GROUP,
        source=str(SOURCE), source_sha256=fingerprint, gpu_count=4,
        claim_deadline=now + 10 * 3600, decision_deadline=now + 14 * 3600,
        maximum_jobs=8, new_training_transitions=400000, expected_updates=5626,
        authorization='8 September: user requested factual-value retest on all four GPUs',
        protocol='2 maps x 2 treatments x seeds 0/1; completed identical-source Bernoulli references; 50k incl 5k prefill; final100 greedy'))
    print(json.dumps(dict(queued=len(jobs), references=references,
        source_sha256=fingerprint, verified_configuration_differences=True)), flush=True)


def job(root, index, gpu):
    entry = json.loads((root / 'queue.json').read_text())['jobs'][index]
    run = sampled.Spec(**entry['spec'])
    args = job_args(run, gpu, root)
    base.train_command = sampled.train_command
    def profile(_args, _run, _env):
        actual = resolve(run, gpu, root)
        expected = json.loads((root / 'resolved.json').read_text())[run.name]
        assert json.loads(json.dumps(actual)) == expected
        return actual
    base.run_screen(args, run, profile, extra_manifest=dict(
        reference=entry['reference'], reference_root=str(REFERENCE),
        factual_targets=True, representation_scale=run.representation_scale,
        off_policy_method='joint V-trace with rho/c caps 1, actual collection-mixture log probabilities',
        exact_dmawm_replica=False))
    train_root = root / 'runs' / run.name / 'train'
    checkpoint = base.latest_checkpoint(train_root, 50000)
    with (checkpoint / 'learner_update_calls.pkl').open('rb') as stream:
        updates = int(pickle.load(stream))
    assert updates == 5626, updates
    base.atomic_json(root / 'runs' / run.name / 'budget_verified.json',
        dict(environment_steps=50000, learner_updates=updates))
    prefix = run_id(run)
    command = [str(PYTHON), str(SOURCE / 'scripts/diagnose_policy_interface.py'),
        '--config', str(train_root / 'run/config.yaml'), '--checkpoint', str(checkpoint),
        '--output', str(root / 'diagnostics' / (prefix + '-rank')),
        '--wandb-id', prefix + '-rank', '--wandb-group', GROUP,
        '--label', f'Factual-value retest {run.arm}; final 50k']
    env = base.execution_environment(args)
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    children = base.OwnedChildren()
    try:
        with (root / 'workers' / (prefix + '-rank.log')).open('x') as stream:
            process = children.start(command, cwd=SOURCE, env=env, stdout=stream, stderr=stream)
            if process.wait():
                raise RuntimeError('Post-training ranking diagnostic failed; final100 is retained')
    finally:
        children.close()


def retention(root):
    from compact_verification_checkpoints_20260906 import retain
    lock = (root / 'retention.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root / 'retention.pid').write_text(str(os.getpid()))
    while True:
        state = json.loads((root / 'queue.json').read_text())
        for folder in root.glob('runs/*/train/run/ckpt'):
            pointer = folder / 'latest'
            if not pointer.exists():
                continue
            current = pointer.read_text().strip()
            if not (folder / current / 'done').exists():
                continue
            for done in sorted(folder.glob('*/done')):
                path = done.parent / 'agent.pkl'
                if done.parent.name < current and path.exists():
                    record = retain(path, folder)
                    if record:
                        print(json.dumps(record), flush=True)
        base.atomic_json(root / 'retention_health.json', dict(pid=os.getpid(), updated_at=time.time()))
        if all(j['status'] in ('complete', 'failed') for j in state['jobs']):
            break
        if time.time() >= state['decision_deadline']:
            break
        time.sleep(60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('init', 'worker', 'job', 'retention'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--gpu', type=int, choices=range(4), default=0)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    if args.mode == 'init':
        initialize(args.root)
    elif args.mode == 'worker':
        target.worker(args.root, args.gpu)
    elif args.mode == 'retention':
        retention(args.root)
    else:
        job(args.root, args.index, args.gpu)
