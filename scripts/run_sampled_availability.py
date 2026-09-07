#!/usr/bin/env python3
"""Paired threshold/Bernoulli availability experiment on four L40 GPUs."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import time

import run_coverage_followup as coverage
import run_l40_targeted_followup as target
import run_ppo_correction_screen as base

SOURCE = Path(__file__).resolve().parents[1]
ROOT = Path('/workspace/majepa_sampled_availability_20260907')
PREVIOUS = Path('/workspace/majepa_parallel_env_stability_20260907')
GROUP = 'ma-jepa-sampled-availability-20260907'
PYTHON = Path('/workspace/majepa-runtime/bin/python')


@dataclass(frozen=True)
class Spec(target.Spec):
    mask_sampling: str = 'threshold'

    @property
    def configuration_flags(self):
        return super().configuration_flags + (
            '--agent.marl.ctde.imagination_mask_sampling', self.mask_sampling)


def spec(mode, map_name, seed):
    return Spec(**{**asdict(target.make_spec(mode, map_name, seed)),
                   'mask_sampling': mode})


def job_args(run, gpu, root):
    args = target.job_args(run, gpu, root)
    args.source = SOURCE
    args.expected_source_sha256 = (SOURCE / 'SOURCE_SHA256').read_text().strip()
    args.wandb_group = GROUP
    args.wandb_run_prefix = args.wandb_job_id = f'am1-{run.mask_sampling}-{run.map_name}-s{run.seed}'
    args.screen_label = 'Paired imagined availability sampling; 1 env, BPTT2, alignment + mixed replay'
    args.portserver_address = f'@majepa-availability-g{gpu}'
    args.portserver_pool = f'{53000 + gpu * 450}-{53449 + gpu * 450}'
    return args


def train_command(args, run, logdir):
    command = coverage.train_command(args, run, logdir)
    command[command.index('--run.checkpoint_at_curve_eval') + 1] = 'True'
    command[command.index('--logger.filter') + 1] += '|imagined_action/'
    return command


def resolve(run, gpu, root):
    import elements
    from majepa.main import _load_configs, _resolve_config_profiles
    args = job_args(run, gpu, root)
    result = {}
    for phase, command in [('train', train_command(args, run, Path('unused'))),
                           ('final100', base.eval_command(args, run, Path('unused'), Path('checkpoint')))]:
        parsed, rest = elements.Flags(configs=['smac_vector', 'ma_jepa']).parse_known(command[3:])
        c = elements.Flags(_resolve_config_profiles(_load_configs(), parsed.configs)).parse(rest)
        assert c.agent.marl.ctde.self_fed.bptt_steps == 2
        assert c.agent.marl.ctde.self_fed.trajectory_kl_scale == .1
        assert c.replay.world_uniform_mix == .5
        assert c.agent.marl.ctde.imagination_mask_sampling == run.mask_sampling
        assert not c.agent.marl.ctde.mask_calibration.enabled
        assert not c.agent.ppo.factual_value.enabled
        assert c.agent.slowvalue.rate == 1 and c.agent.target_encoder.rate == .01
        assert c.run.envs == (1 if phase == 'train' else 4)
        if phase == 'train':
            assert c.run.steps == 50000 and c.run.train_ratio == 128
            assert c.run.world_model_start_step == c.run.ppo_start_step == 5000
        else:
            assert c.run.eval_eps == 100 and c.run.eval_policy_mode == 'eval'
        result[phase] = dict(command=command, config=c.flat)
    return result


def initialize(root):
    root.mkdir(exist_ok=False)
    (root / 'workers').mkdir()
    previous = json.loads((PREVIOUS / 'queue.json').read_text())
    jobs, resolved, comparisons = [], {}, {}
    # The free GPU first evaluates existing one-env checkpoints; no new data
    # from these diagnostics is ever imported into training replay.
    jobs.append(dict(index=0, gpu=3, name='frozen-greedy-vs-sampled', kind='evaluation',
        status='pending', command=[str(PYTHON), str(Path(__file__).resolve()),
            'diagnostics', '--root', str(root), '--gpu', '3']))
    for gpu, (seed, mode) in enumerate([(0, 'threshold'), (0, 'bernoulli'),
                                       (1, 'threshold'), (1, 'bernoulli')]):
        run = spec(mode, '2s3z', seed)
        resolved[run.name] = resolve(run, gpu, root)
        jobs.append(dict(index=len(jobs), gpu=gpu, name=run.name, kind='train',
            status='pending', spec=asdict(run),
            wandb=f'https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/am1-{mode}-2s3z-s{seed}-train',
            command=[str(PYTHON), str(Path(__file__).resolve()), 'job', '--root', str(root),
                     '--gpu', str(gpu), '--index', str(len(jobs))]))
    for seed in (0, 1):
        a = resolved[spec('threshold', '2s3z', seed).name]
        b = resolved[spec('bernoulli', '2s3z', seed).name]
        comparisons[str(seed)] = {}
        for phase in a:
            x, y = a[phase]['config'], b[phase]['config']
            diff = {k: [x.get(k), y.get(k)] for k in set(x) | set(y) if x.get(k) != y.get(k)}
            assert diff == {'agent.marl.ctde.imagination_mask_sampling': ['threshold', 'bernoulli']}, diff
            comparisons[str(seed)][phase] = diff
    base.atomic_json(root / 'resolved.json', resolved)
    base.atomic_json(root / 'comparisons.json', comparisons)
    base.atomic_json(root / 'queue.json', dict(jobs=jobs, group=GROUP, created_at=time.time(),
        source=str(SOURCE), source_sha256=base.source_fingerprint(SOURCE),
        claim_deadline=previous['claim_deadline'], decision_deadline=previous['decision_deadline'],
        gpu_count=4, new_training_transitions=200000, expected_updates=5626,
        protocol='2 seeds x threshold/Bernoulli; 1 env; 50k incl 5k prefill; final100 greedy; sampled100 diagnostic and action ranking afterward'))
    print(json.dumps({'queued': len(jobs), 'group': GROUP, 'paired_config_checks': comparisons}), flush=True)


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def evaluate_modes(root, gpu, run, checkpoint, label, modes):
    args = job_args(run, gpu, root)
    args.wandb_run_prefix = args.wandb_job_id = label
    args.run_spec = run
    folder = root / 'diagnostics' / label
    folder.mkdir(parents=True, exist_ok=False)
    env = base.execution_environment(args)
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    before = file_hash(checkpoint / 'agent.pkl')
    summaries = {}
    children = base.OwnedChildren()
    try:
        with (folder / 'portserver.log').open('x') as stream:
            portserver = children.start([str(PYTHON), str(args.portserver_script),
                '--portserver_static_pool', args.portserver_pool,
                '--portserver_address', args.portserver_address], env=env, stdout=stream, stderr=stream)
            time.sleep(2)
            if portserver.poll() is not None:
                raise RuntimeError('Evaluation portserver failed')
            for mode in modes:
                phase = 'greedy100' if mode == 'eval' else 'sampled100'
                command = base.eval_command(args, run, folder / phase / 'run', checkpoint)
                command[command.index('--run.eval_policy_mode') + 1] = mode
                base.run_child(args, children, folder, phase, command, env)
                summary = json.loads((folder / phase / 'run/evaluation_summary.json').read_text())
                assert summary['episodes'] == 100 and summary['policy_mode'] == mode
                summaries[phase] = summary
    finally:
        children.close()
    assert file_hash(checkpoint / 'agent.pkl') == before, 'Evaluation modified its source checkpoint'
    base.atomic_json(folder / 'comparison.json', dict(checkpoint=str(checkpoint),
        model_file_sha256=before, summaries=summaries, training_changed=False,
        note='Same fixed checkpoint, evaluation episode seeds and quotas; categorical versus greedy actions; no collection exploration mixture'))


def diagnostics(root, gpu):
    previous = Path('/workspace/majepa_targeted_l40_20260907/runs')
    for map_name, name in [('3s_vs_4z', 'control-3s_vs_4z-seed0'), ('MMM', 'mem95-MMM-seed0')]:
        checkpoint = base.latest_checkpoint(previous / name / 'train', 50000)
        evaluate_modes(root, gpu, spec('threshold', map_name, 0), checkpoint,
            f'am1-frozen-{map_name}-s0', ('eval', 'eval_sample'))


def job(root, index, gpu):
    entry = json.loads((root / 'queue.json').read_text())['jobs'][index]
    run = Spec(**entry['spec'])
    args = job_args(run, gpu, root)
    base.train_command = train_command
    def profile(_args, _run, _env):
        actual = resolve(run, gpu, root)
        expected = json.loads((root / 'resolved.json').read_text())[run.name]
        assert json.loads(json.dumps(actual)) == expected, 'Queued configuration changed'
        return actual
    base.run_screen(args, run, profile, extra_manifest=dict(
        sampling_mode=run.mask_sampling, paired_seed=run.seed, benchmark_budget_includes_prefill=True))
    train_root = root / 'runs' / run.name / 'train'
    checkpoint = base.latest_checkpoint(train_root, 50000)
    with (checkpoint / 'learner_update_calls.pkl').open('rb') as stream:
        updates = int(pickle.load(stream))
    assert updates == 5626, updates
    base.atomic_json(root / 'runs' / run.name / 'budget_verified.json',
        dict(environment_steps=50000, learner_updates=updates))
    prefix = f'am1-{run.mask_sampling}-{run.map_name}-s{run.seed}'
    evaluate_modes(root, gpu, run, checkpoint, prefix, ('eval_sample',))
    command = [str(PYTHON), str(SOURCE / 'scripts/diagnose_policy_interface.py'),
        '--config', str(train_root / 'run/config.yaml'), '--checkpoint', str(checkpoint),
        '--output', str(root / 'diagnostics' / (prefix + '-rank')),
        '--wandb-id', prefix + '-rank', '--wandb-group', GROUP,
        '--label', f'Availability sampling {run.mask_sampling}; final 50k']
    env = base.execution_environment(args)
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    with (root / 'workers' / (prefix + '-rank.log')).open('x') as stream:
        subprocess.run(command, cwd=SOURCE, env=env, stdout=stream, stderr=stream, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('init', 'worker', 'job', 'diagnostics'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--gpu', type=int, choices=range(4), default=0)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    if args.mode == 'init':
        initialize(args.root)
    elif args.mode == 'worker':
        target.worker(args.root, args.gpu)
    elif args.mode == 'diagnostics':
        diagnostics(args.root, args.gpu)
    else:
        job(args.root, args.index, args.gpu)
