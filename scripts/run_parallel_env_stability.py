#!/usr/bin/env python3
"""Paired 1-vs-8 SMAC environment screen; three disjoint seed blocks per map."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import subprocess
import time

import run_l40_targeted_followup as target
import run_coverage_followup as coverage
import run_ppo_correction_screen as base

SOURCE = Path(__file__).resolve().parents[1]
ROOT = Path('/workspace/majepa_parallel_env_stability_20260907')
DEPENDENCY = target.ROOT
GROUP = 'ma-jepa-parallel-env-stability-20260907'


def spec(envs, map_name, seed):
    values = asdict(target.make_spec(f'env{envs}', map_name, seed))
    return target.Spec(**{**values, 'envs':envs})


def job_args(run, gpu, root):
    args = target.job_args(run, gpu, root)
    args.wandb_group = GROUP
    args.wandb_run_prefix = args.wandb_job_id = f'pe1-e{run.envs}-{run.map_name}-s{run.seed}'
    args.screen_label = 'Paired 1 vs 8 environments; alignment + mixed replay; equal total data and updates'
    args.portserver_address = f'@majepa-parallel-env-g{gpu}'
    args.portserver_pool = f'{55000+gpu*450}-{55449+gpu*450}'
    return args


def train_command(args, run, logdir):
    command = coverage.train_command(args, run, logdir)
    command[command.index('--run.checkpoint_at_curve_eval')+1] = 'True'
    return command


def resolve(run, gpu, root):
    import elements
    from majepa.main import _load_configs, _resolve_config_profiles
    args = job_args(run,gpu,root)
    result = {}
    for phase,command in [('train',train_command(args,run,Path('unused'))),
                         ('final100',base.eval_command(args,run,Path('unused'),Path('checkpoint')))]:
        parsed,rest=elements.Flags(configs=['smac_vector','ma_jepa']).parse_known(command[3:])
        config=elements.Flags(_resolve_config_profiles(_load_configs(),parsed.configs)).parse(rest)
        assert config.agent.marl.ctde.self_fed.trajectory_kl_scale == .1
        assert config.agent.marl.ctde.self_fed.bptt_steps == 2
        assert config.replay.world_uniform_mix == .5
        assert not config.agent.ppo.factual_value.enabled
        assert config.agent.ppo.factual_value.representation_scale == 0
        assert config.run.envs == (run.envs if phase=='train' else 4)
        assert config.run.train_ratio == 128
        if phase=='train':
            assert config.run.world_model_start_step == config.run.ppo_start_step == 5000
            assert config.run.steps==50000 and config.run.checkpoint_at_curve_eval
        else:
            assert config.run.eval_eps==100
        result[phase]=dict(command=command,config=config.flat)
    return result


def initialize(root):
    import elements
    previous=json.loads((DEPENDENCY/'queue.json').read_text())
    root.mkdir(exist_ok=False)
    (root/'workers').mkdir()
    jobs,resolved,comparisons=[],{},{}
    for gpu,seed in enumerate((0,1000,2000)):
        for envs in (8,1):
            reference_only=envs==1 and seed==0
            maps=('3s_vs_4z','2s3z') if reference_only else ('2s3z','3s_vs_4z')
            for map_name in maps:
                run=spec(envs,map_name,seed)
                resolved[run.name]=resolve(run,gpu,root)
                jobs.append(dict(index=len(jobs),kind='parallel_env',name=run.name,
                    gpu=gpu,status='pending',spec=asdict(run),
                    reference_only=reference_only,
                    wandb=f'https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/' + (
                        f'l40d-control-{map_name}-s0-train' if reference_only else
                        f'pe1-e{envs}-{map_name}-s{seed}-train')))
        for map_name in ('2s3z','3s_vs_4z'):
            one=resolve(spec(1,map_name,seed),gpu,root)
            eight=resolve(spec(8,map_name,seed),gpu,root)
            delta={}
            for phase in one:
                a,b=one[phase]['config'],eight[phase]['config']
                delta[phase]={k:[a.get(k),b.get(k)] for k in set(a)|set(b) if a.get(k)!=b.get(k)}
            assert delta=={'train':{'run.envs':[1,8]},'final100':{}},delta
            comparisons[f'{map_name}-s{seed}']=delta
    # The actual pinned scheduler is called once per individual environment
    # transition, including inside a vector step. It must not scale by env count.
    config=next(iter(resolved.values()))['train']['config']
    counts={}
    for envs in (1,8):
        ratio=elements.when.Ratio(128/(config['batch_size']*config['batch_length']))
        updates=0
        for vector_start in range(0,50000,envs):
            for step in range(vector_start+1,vector_start+envs+1):
                if step>=5000:
                    updates+=ratio(step)
        counts[str(envs)]=updates
    assert counts['1']==counts['8']==5626,counts
    references={}
    for map_name in ('2s3z','3s_vs_4z'):
        name=f'control-{map_name}-seed0'
        entry=next(j for j in previous['jobs'] if j['name']==name)
        references[map_name]=dict(name=name,root=str(DEPENDENCY),status_at_creation=entry['status'],
            wandb=f'https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/l40d-control-{map_name}-s0-final100')
    base.atomic_json(root/'resolved.json',resolved)
    base.atomic_json(root/'comparisons.json',dict(phase_differences=comparisons,
        scheduler_update_counts=counts,seed_blocks=[[s+i for i in range(8)] for s in (0,1000,2000)],
        curve_step_note='8-env driver batches may evaluate up to 8 transitions after intermediate 5k boundaries; final 50k is exact.'))
    base.atomic_json(root/'queue.json',dict(jobs=jobs,group=GROUP,created_at=time.time(),
        claim_deadline=previous['claim_deadline'],decision_deadline=previous['decision_deadline'],
        source=str(SOURCE),source_sha256=base.source_fingerprint(SOURCE),
        expected_updates=5626,reference_controls=references,
        comparison='Two maps x three seeds x two environment counts; reuse two matching L40 seed0 controls; ten new 50k runs.',
        dependency=str(DEPENDENCY),new_training_transitions=500000))
    print(json.dumps(dict(queued=len(jobs),group=GROUP,expected_updates=counts)),flush=True)


def job(root,index,gpu):
    entry=json.loads((root/'queue.json').read_text())['jobs'][index]
    run=target.Spec(**entry['spec'])
    args=job_args(run,gpu,root)
    base.train_command=train_command
    def profile(_args,_run,_env):
        actual=resolve(run,gpu,root)
        expected=json.loads((root/'resolved.json').read_text())[run.name]
        assert json.loads(json.dumps(actual))==expected,'Resolved queued configuration changed'
        return actual
    if entry.get('reference_only'):
        reference_name=f'control-{run.map_name}-seed0'
        while True:
            state=json.loads((DEPENDENCY/'queue.json').read_text())
            reference=next(j for j in state['jobs'] if j['name']==reference_name)
            if reference['status']=='complete':
                break
            if reference['status']=='failed' or time.time()>=state['decision_deadline']:
                raise RuntimeError('Reference control unavailable for paired diagnostics')
            time.sleep(20)
        train_root=DEPENDENCY/'runs'/reference_name/'train'
        (root/'runs'/run.name).mkdir(parents=True)
    else:
        base.run_screen(args,run,profile,extra_manifest=dict(
            parallel_environment_comparison=True,environment_seeds=list(range(run.seed,run.seed+run.envs)),
            paired_seed=run.seed,expected_updates=5626,benchmark_budget_includes_prefill=True))
        train_root=root/'runs'/run.name/'train'
    final=base.latest_checkpoint(train_root,expected_steps=50000)
    with (final/'learner_update_calls.pkl').open('rb') as stream:
        updates=int(pickle.load(stream))
    if updates!=5626:
        raise RuntimeError(f'Update budget mismatch: expected 5626, got {updates}')
    base.atomic_json(root/'runs'/run.name/'budget_verified.json',dict(environment_steps=50000,
        learner_updates=updates,environment_count=run.envs))
    # Use a common transition window, rather than comparing differently spaced
    # wall-clock saves as if they were equally sized parameter updates.
    candidates=[]
    for done in final.parent.glob('*/done'):
        with (done.parent/'step.pkl').open('rb') as stream:
            step=int(pickle.load(stream))
        if 45000<=step<=45008 and (done.parent/'agent.pkl').exists():
            candidates.append((step,done.parent))
    if not candidates:
        raise RuntimeError('Missing predeclared 45k checkpoint for policy-drift diagnostic')
    step,older=min(candidates)
    name=f'pe1-e{run.envs}-{run.map_name}-s{run.seed}-drift'
    command=[str(args.python),str(SOURCE/'scripts/diagnose_policy_interface.py'),
        '--config',str(train_root/'run/config.yaml'),'--checkpoint',str(older),
        '--compare-checkpoint',str(final),'--drift-only','--output',str(root/'diagnostics'/name),
        '--wandb-id',name,'--wandb-group',GROUP,
        '--label',f'Parallel-env comparison: {run.envs} envs; {step} to 50000']
    env=base.execution_environment(args)
    env['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    with (root/'workers'/f'{name}.log').open('x') as stream:
        subprocess.run(command,cwd=SOURCE,env=env,stdout=stream,stderr=stream,check=True)


def worker(root,gpu):
    # Wait for training AND final evaluation, not a transient idle GPU interval.
    while True:
        state=json.loads((root/'queue.json').read_text())
        if time.time()>=state['claim_deadline']:
            return
        previous=json.loads((DEPENDENCY/'queue.json').read_text())
        predecessor=next(j for j in previous['jobs'] if j['name']==f'mem95-MMM-seed{gpu}')
        if predecessor['status']=='complete':
            break
        if predecessor['status']=='failed':
            raise RuntimeError('MMM predecessor failed; preserve it for inspection')
        base.atomic_json(root/'workers'/f'gpu{gpu}.waiting.json',dict(pid=os.getpid(),
            phase='waiting_for_MMM_final100',updated_at=time.time(),predecessor=predecessor['name']))
        time.sleep(20)
    target.worker(root,gpu)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('init','worker'))
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--gpu',type=int,choices=range(3),default=0)
    args=parser.parse_args()
    initialize(args.root) if args.mode=='init' else worker(args.root,args.gpu)
