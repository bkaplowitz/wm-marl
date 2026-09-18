#!/usr/bin/env python3
"""Independent head simplifications on the one-environment localmask reference."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

SOURCE = Path('/workspace/ma_jepa_simplification_20260918')
ROOT = Path('/workspace/majepa_simplification_20260918')
GROUP = 'ma-jepa-localmask-simplification-20260918'
PREDECESSOR = Path('/workspace/majepa_matched_history_20260917/queue.json')
EXPECTED_SHA = (SOURCE / 'SOURCE_SHA256').read_text().strip()
CAPACITY_BYTES = 300_000_000_000
MIN_HEADROOM_BYTES = 20_000_000_000
HISTORICAL_DECAY = 0.9998
MAP_AGENTS = {'3s_vs_4z': 3, '8m': 8, '2s3z': 5}
ARMS = {name: (0.5, HISTORICAL_DECAY) for name in ('nojointmask','nolocaloutcomes','heads256')}
PLAN = [(arm, map_name, seed) for arm in ARMS for seed in (0,1,2) for map_name in ('2s3z','8m','3s_vs_4z')]

# Import the existing, audited queue/runtime engine.  The source path supplied
# to every child process below is the fingerprinted branch copy above.
sys.path.insert(0, '/workspace/majepa_tg_script_20260915')
sys.path.insert(0, str(SOURCE / 'src'))
import run_truncated_geometric_20260915 as tg  # noqa: E402

sys.path.insert(0, str(SOURCE / 'src'))
tg.MAP_AGENTS.update(MAP_AGENTS)
engine = tg.engine
atomic_json = tg.atomic_json


@dataclass(frozen=True)
class Spec(tg.Spec):
    direct_cosine_scale: float = 2.0
    mask_source: str = "joint"
    world_uniform_mix: float = 0.5
    behavior_recency_decay: float = 0.0
    behavior_uniform_mix: float = 0.5

    @property
    def name(self):
        return f'{self.arm}-{self.map_name}-seed{self.seed}'

    @property
    def configuration_flags(self):
        flags = list(super().configuration_flags)
        decay_indices = [
            i for i, value in enumerate(flags)
            if value == '--replay.recency_decay'
        ]
        if not decay_indices:
            raise RuntimeError('missing replay.recency_decay flag')
        flags[decay_indices[-1] + 1] = repr(float(HISTORICAL_DECAY))
        mix_indices = [
            i for i, value in enumerate(flags)
            if value == '--replay.world_uniform_mix'
        ]
        if not mix_indices:
            raise RuntimeError('missing replay.world_uniform_mix flag')
        flags[mix_indices[-1] + 1] = repr(float(self.world_uniform_mix))
        return tuple(flags) + (
            '--replay.behavior_recency_decay',
            repr(float(self.behavior_recency_decay)),
            "--replay.behavior_uniform_mix", repr(float(self.behavior_uniform_mix)),
            '--agent.marl.ctde.imagination_mask_source', self.mask_source,
            '--agent.marl.ctde.compare_mask_heads', str(self.arm != 'nojointmask'),
            '--agent.loss_scales.ctde_multistep_jepa_cosine', str(self.direct_cosine_scale),
            '--agent.simplification.joint_mask', str(self.arm != 'nojointmask'),
            '--agent.simplification.local_outcomes', str(self.arm != 'nolocaloutcomes'),
            '--agent.rewhead.units', str(256 if self.arm == 'heads256' else 512),
            '--agent.conhead.units', str(256 if self.arm == 'heads256' else 512),
            '--agent.maskhead.units', str(256 if self.arm == 'heads256' else 512),
        )


def make_spec(arm: str, map_name: str, seed: int) -> Spec:
    world_mix, behavior_decay = ARMS[arm]
    values = asdict(tg.make_spec(map_name, seed))
    values.update(
        arm=arm,
        mask_source='local',
        envs=1,
        action_margin_scale=0.0 if arm in ('nomargin', 'nodirect') else 0.1,
        direct_cosine_scale=0.0 if arm == 'nodirect' else 2.0,
        map_name=map_name,
        steps=100000 if map_name == '3s_vs_4z' else 50000,
        seed=seed,
        world_uniform_mix=float(world_mix),
        behavior_recency_decay=float(behavior_decay),
    )
    return Spec(**values)


def run_id(run: Spec) -> str:
    return f'si18-{run.arm}-{run.map_name}-s{run.seed}'


def job_args(run: Spec, gpu: int):
    args = tg.base.job_args(run, gpu)
    args.source = SOURCE
    args.expected_source_sha256 = EXPECTED_SHA
    args.experiment_root = ROOT
    args.wandb_group = GROUP
    args.wandb_run_prefix = args.wandb_job_id = run_id(run)
    root_label = (
        'uniform WM' if run.world_uniform_mix == 1.0 else '50/50 WM'
    )
    behavior_label = (
        'uniform PPO roots'
        if not run.behavior_recency_decay
        else '50/50 uniform + recent PPO roots decay=0.9998'
    )
    args.screen_label = (
        f'Localmask simplification {run.arm} / WM4096c64 / Actor512 / '
        f'{root_label} + {behavior_label} / {run.map_name} seed {run.seed}'
    )
    args.portserver_address = f'@majepa-replay-views-g{gpu}'
    args.portserver_pool = f'{59000 + 300 * gpu}-{59299 + 300 * gpu}'
    return args


def train_command(args, run, logdir):
    return tg.train_command(args, run, logdir)


def commands(run: Spec, gpu: int):
    args = job_args(run, gpu)
    return {
        'train': train_command(args, run, Path('unused')),
        'final100': tg.base.previous.base.direct.eval_command(
            args, run, Path('unused'), Path('checkpoint')
        ),
    }


def resolve(run: Spec, gpu: int):
    """Resolve both phases and assert that only replay-view values vary."""
    import elements
    import majepa.main as package
    from majepa.main import _load_configs, _resolve_config_profiles

    if not Path(package.__file__).is_relative_to(SOURCE):
        raise RuntimeError(f'unexpected package source: {package.__file__}')
    common = {
        'agent.opt.lr': 1e-4,
        'agent.marl.ctde.opt.lr': 1e-4,
        'agent.ppo.actor_lr': 3e-5,
        'agent.ppo.critic_lr': 3e-5,
        'agent.dyn.parallel_transformer.deter': 4096,
        'agent.dyn.parallel_transformer.hidden': 512,
        'agent.dyn.parallel_transformer.classes': 64,
        'agent.rewhead.units': 256 if run.arm == 'heads256' else 512,
        'agent.conhead.units': 256 if run.arm == 'heads256' else 512,
        'agent.maskhead.units': 256 if run.arm == 'heads256' else 512,
        'agent.policy.units': 512,
        'agent.value.units': 512,
        'agent.marl.ctde.critic.width': 256,
        'agent.marl.ctde.critic.value_units': 256,
        'agent.loss_scales.ctde_posterior_alignment': 0.05,
        'agent.loss_scales.ctde_multistep_jepa_action': run.action_margin_scale,
        'agent.loss_scales.ctde_multistep_jepa_cosine': run.direct_cosine_scale,
        'agent.marl.ctde.imagination_mask_source': run.mask_source,
        'agent.marl.ctde.compare_mask_heads': run.arm != 'nojointmask',
        'agent.simplification.joint_mask': run.arm != 'nojointmask',
        'agent.simplification.local_outcomes': run.arm != 'nolocaloutcomes',
        'agent.collection_unimix': 0.0,
        'agent.policy.unimix': 0.0,
        'agent.ppo.replay_value_scale': 0.3,
        'agent.marl.ctde.self_fed.trajectory_kl_scale': 0.1,
        'agent.marl.ctde.self_fed.bptt_steps': 2,
        'agent.marl.ctde.imagination_mask_sampling': 'bernoulli',
        'agent.marl.ctde.teammate_belief.enabled': False,
        'agent.marl.ctde.multistep_jepa.belief_context': False,
        'agent.ppo.factual_value.enabled': False,
        'agent.slowvalue.rate': 1.0,
        'agent.target_encoder.rate': 0.01,
        'agent.ppo.clip_epsilon': 0.2,
        'agent.ppo.entropy_coefficient': 0.003,
        'agent.ppo.entropy_schedule.enabled': False,
        'replay.sampling': 'recent_world_uniform_behavior',
        'replay.recency_decay': HISTORICAL_DECAY,
        'replay.world_uniform_mix': float(run.world_uniform_mix),
        'replay.behavior_recency_decay': float(run.behavior_recency_decay),
        'replay.behavior_uniform_mix': float(run.behavior_uniform_mix),
        'run.replay_stream_mode': 'snapshot_staggered',
        'run.replay_startup_behavior_min_starts': 4,
        'run.replay_trace_batches': 16,
        'task': f'smac_{run.map_name}',
        'seed': run.seed,
    }
    result = {}
    for phase, command in commands(run, gpu).items():
        parsed, rest = elements.Flags(
            configs=['smac_vector', 'ma_jepa']
        ).parse_known(command[3:])
        config = elements.Flags(
            _resolve_config_profiles(_load_configs(), parsed.configs)
        ).parse(rest)
        current = json.loads(json.dumps(config.flat))
        expected = dict(common)
        if phase == 'train':
            expected.update({
                'run.steps': run.steps,
                'run.envs': run.envs,
                'run.world_model_start_step': 5000,
                'run.ppo_start_step': 5000,
                'run.train_ratio': 128,
                'run.save_every': 5000,
                'run.checkpoint_at_curve_eval': False,
                'run.curve_eval_interval': 5000,
                'run.curve_eval_eps': 32,
            })
        else:
            expected.update({
                'run.eval_eps': 100,
                'run.eval_policy_mode': 'eval',
            })
        mismatches = {
            key: [value, current.get(key)]
            for key, value in expected.items()
            if current.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f'{run.name}/{phase}: resolved config mismatch: {mismatches}')
        result[phase] = {
            'command': command,
            'config': current,
            'controlled_changes': {
                'replay.world_uniform_mix': float(run.world_uniform_mix),
                'replay.behavior_recency_decay': float(run.behavior_recency_decay),
        'replay.behavior_uniform_mix': float(run.behavior_uniform_mix),
            },
        }
    return result


def storage_check():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'storage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cached = ROOT / 'storage_latest.json'
        if cached.exists():
            record = json.loads(cached.read_text())
            if time.time() - record['at'] < 300 and record['headroom_bytes'] > 40_000_000_000:
                return record
        used = int(
            subprocess.check_output(
                ['du', '-s', '-B1', '/workspace'], text=True
            ).split()[0]
        )
        available = int(os.statvfs('/workspace').f_bavail * os.statvfs('/workspace').f_frsize)
        headroom = min(available, CAPACITY_BYTES - used)
        record = {
            'at': time.time(),
            'capacity_bytes_conservative': CAPACITY_BYTES,
            'used_bytes': used,
            'conservative_headroom_bytes': CAPACITY_BYTES - used,
            'filesystem_available_bytes': available,
            'headroom_bytes': headroom,
            'minimum_headroom_bytes': MIN_HEADROOM_BYTES,
        }
        atomic_json(ROOT / 'storage_latest.json', record)
        if headroom < MIN_HEADROOM_BYTES:
            raise RuntimeError(f'storage guard: {headroom / 1e9:.1f} GB headroom')
        return record


def initialize():
    if ROOT.exists() and (ROOT / 'queue.json').exists():
        raise FileExistsError(f'queue already exists: {ROOT / "queue.json"}')
    if not SOURCE.exists() or not (SOURCE / 'SOURCE_SHA256').exists():
        raise FileNotFoundError(SOURCE / 'SOURCE_SHA256')
    fingerprint = engine.source_fingerprint(SOURCE)
    if fingerprint != EXPECTED_SHA:
        raise RuntimeError(f'source fingerprint mismatch: {fingerprint} != {EXPECTED_SHA}')
    from smac.env.starcraft2.maps import get_map_params
    for map_name, agent_count in MAP_AGENTS.items():
        if get_map_params(map_name)['n_agents'] != agent_count:
            raise RuntimeError(f'agent count mismatch for {map_name}')
        map_path = Path('/workspace/StarCraftII/Maps/SMAC_Maps') / f'{map_name}.SC2Map'
        if not map_path.exists():
            raise FileNotFoundError(map_path)

    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'workers').mkdir(exist_ok=True)
    jobs, resolved = [], {}
    for arm, map_name, seed in PLAN:
        run = make_spec(arm, map_name, seed)
        resolved[run.name] = resolve(run, 0)
        jobs.append({
            'index': len(jobs),
            'gpu': None,
            'name': run.name,
            'status': 'pending',
            'phase': 'localmask_head_simplification',
            'spec': asdict(run),
            'expected_updates': (run.steps - 5000) // 8 + 1,
            'source': str(SOURCE),
            'source_sha256': fingerprint,
            'wandb': f'https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/{run_id(run)}-train',
            'wandb_final': f'https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/{run_id(run)}-final100',
        })
    storage = storage_check()
    atomic_json(ROOT / 'resolved.json', resolved)
    now = time.time()
    atomic_json(ROOT / 'queue.json', {
        'created_at': now,
        'claim_deadline': now + 14 * 24 * 3600,
        'jobs': jobs,
        'group': GROUP,
        'source': str(SOURCE),
        'source_sha256': fingerprint,
        'gpu_count': 4,
        'maximum_jobs': len(jobs),
        'maximum_driver_records': sum(j['spec']['steps'] for j in jobs),
        'expected_updates': '11876 for 100k; 5626 for 50k',
        'predecessor': str(PREDECESSOR),
        'authorization': 'User requested cleanup and all three independent simplification tests; seeds 0/1/2 on 2s3z, 8m, 3s_vs_4z.',
        'protocol': 'One env. Each arm changes only the named head/loss or local-head width. Plain localmask, no shared-mask supervision. Fixed total 50k/100k records, 5k prefill, train_ratio128, local/joint LR1e-4, entropy.003, mixed independent replay, start4, BPTT2, final100. Original env1 driver path retained.',
        'arms': list(ARMS),
        'conditional_followups': [],
        'queue_policy': (
            'Workers wait for predecessor jobs on their own GPU and for '
            'their GPU to be free; pending arms are claimed under one lock.'
        ),
        'storage_at_launch': storage,
    })
    print(json.dumps({
        'group': GROUP,
        'root': str(ROOT),
        'jobs': len(jobs),
        'arms': list(ARMS),
        'maps': list(MAP_AGENTS),
        'seeds': [0, 1, 2],
        'predecessor': str(PREDECESSOR),
        'headroom_gb': storage['headroom_bytes'] / 1e9,
    }), flush=True)


def predecessor_active(gpu=None):
    try:
        state = json.loads(PREDECESSOR.read_text())
    except (OSError, json.JSONDecodeError):
        return True
    return any(job.get('status') != 'complete' for job in state.get('jobs', []))


def update_job(index: int, **values):
    with (ROOT / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads((ROOT / 'queue.json').read_text())
        state['jobs'][index].update(values)
        atomic_json(ROOT / 'queue.json', state)


def run_job(index: int, gpu: int):
    entry = json.loads((ROOT / 'queue.json').read_text())['jobs'][index]
    if entry['gpu'] != gpu or entry['status'] != 'running':
        raise RuntimeError('job was not claimed by this GPU')
    run = Spec(**entry['spec'])
    engine.train_command = train_command
    engine.eval_command = tg.base.previous.base.direct.eval_command

    def profile(_args, _run, _env):
        actual = resolve(run, 0)
        expected = json.loads((ROOT / 'resolved.json').read_text())[run.name]
        if json.loads(json.dumps(actual)) != expected:
            raise RuntimeError(f'saved and actual resolved configurations differ for {run.name}')
        return actual

    try:
        engine.run_screen(
            job_args(run, gpu), run, profile,
            extra_manifest={
                'training_from_scratch': True,
                'world_root_sampler': (
                    'uniform' if run.world_uniform_mix == 1.0
                    else '50/50 uniform + exponential recency 0.9998'
                ),
                'behavior_root_sampler': (
                    'uniform' if not run.behavior_recency_decay
                    else '50/50 uniform + exponential recency 0.9998'
                ),
                'world_uniform_mix': run.world_uniform_mix,
                'behavior_recency_decay': run.behavior_recency_decay,
                'independent_replay_views': True,
                'automatic_promotion': False,
            },
        )
        checkpoint = engine.latest_checkpoint(
            ROOT / 'runs' / run.name / 'train', run.steps
        )
        with (checkpoint / 'learner_update_calls.pkl').open('rb') as handle:
            updates = int(pickle.load(handle))
        if updates != entry['expected_updates']:
            raise RuntimeError(f'unexpected learner budget: {updates}')
        atomic_json(
            ROOT / 'runs' / run.name / 'budget_verified.json',
            {'driver_records': run.steps, 'learner_updates': updates},
        )
        removed = []
        for path in checkpoint.parent.iterdir():
            if path.is_dir() and path != checkpoint and (path / 'done').is_file():
                shutil.rmtree(path)
                removed.append(path.name)
        atomic_json(
            ROOT / 'runs' / run.name / 'checkpoint_retention.json',
            {'kept_final': str(checkpoint), 'removed_intermediate': removed},
        )
        update_job(index, status='complete', finished_at=time.time(), exit_code=0)
    except BaseException as error:
        update_job(index, status='failed', finished_at=time.time(), error=repr(error), exit_code=1)
        raise


def worker(gpu: int):
    gate = json.loads((ROOT / "validation.json").read_text())
    if gate.get("source_sha256") != EXPECTED_SHA or not gate.get("passed"):
        raise RuntimeError("Source has not passed the cleanup and ablation gates")
    lock_path = ROOT / 'workers' / f'gpu{gpu}.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (ROOT / 'workers' / f'gpu{gpu}.pid').write_text(str(os.getpid()) + '\n')
        while True:
            state = json.loads((ROOT / 'queue.json').read_text())
            if time.time() >= state['claim_deadline']:
                return
            blocked = predecessor_active(gpu)
            compute_pids = engine.gpu_processes(gpu)
            if blocked or compute_pids:
                atomic_json(ROOT / 'workers' / f'gpu{gpu}.waiting.json', {
                    'status': 'waiting_for_predecessor_or_gpu',
                    'gpu': gpu,
                    'predecessor_active': blocked,
                    'predecessor_active_any_gpu': predecessor_active(),
                    'compute_pids': compute_pids,
                    'updated_at': time.time(),
                })
                time.sleep(20)
                continue
            with (ROOT / 'queue.lock').open('a') as qlock:
                fcntl.flock(qlock, fcntl.LOCK_EX)
                state = json.loads((ROOT / 'queue.json').read_text())
                pending = next(
                    (j for j in state['jobs'] if j['status'] == 'pending'),
                    None,
                )
                if pending is None:
                    return
                pending.update({
                    'status': 'running',
                    'gpu': gpu,
                    'worker_pid': os.getpid(),
                    'started_at': time.time(),
                })
                atomic_json(ROOT / 'queue.json', state)
                index = pending['index']
            print(json.dumps({
                'gpu': gpu,
                'starting': pending['name'],
                'at': time.time(),
            }), flush=True)
            storage_check()
            run_job(index, gpu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['init', 'worker', 'status'])
    parser.add_argument('--gpu', type=int, choices=range(4), default=0)
    args = parser.parse_args()
    if args.mode == 'init':
        initialize()
    elif args.mode == 'worker':
        worker(args.gpu)
    else:
        print((ROOT / 'queue.json').read_text())

