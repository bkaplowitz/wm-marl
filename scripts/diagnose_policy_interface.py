#!/usr/bin/env python3
"""Frozen policy/value diagnostics with verified real action-prefix replays.

Counterfactuals are accepted only when all public observations, legal masks and
SMAC global states exactly match at every prefix step in a freshly seeded game.
This checks reproducible observable state, not an inaccessible SC2 RNG snapshot.
No training weights or optimizer states are modified. Temporal policy comparisons
use the same recorded histories and posterior RNG for every parameter hybrid.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import pickle
import time
import numpy as np

GROUP = 'ma-jepa-targeted-diagnostics-20260907'


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def load_weights(path):
    if path.is_dir():
        if not (path / 'done').exists():
            raise ValueError(f'Incomplete checkpoint: {path}')
        path = path / ('agent.pkl' if (path / 'agent.pkl').exists() else 'agent.pkl.gz')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as stream:
        payload = pickle.load(stream)
    weights = {key: np.asarray(value) for key, value in payload['params'].items()
               if not key.startswith('opt/')}
    digest = hashlib.sha256()
    for key, value in sorted(weights.items()):
        digest.update(key.encode())
        digest.update(str((value.dtype, value.shape)).encode())
        digest.update(value.tobytes())
    return weights, dict(path=str(path), model_sha256=digest.hexdigest(),
                         counters=payload.get('counters', {}))


def numpy_tree(tree):
    import jax
    return jax.tree.map(lambda value: np.asarray(value), jax.device_get(tree))


class Frozen:
    def __init__(self, config_path, checkpoint, output):
        import elements
        import embodied.jax.nets as nn
        import jax
        import jax.numpy as jnp
        import ninjax as nj
        from ruamel.yaml import YAML
        from majepa.main import _load_configs, _resolve_config_profiles
        from majepa.marl.core import MARLCore
        from majepa.envs.smac import SMACEnv
        from majepa.training.ctde import shared_team_outcomes

        saved = elements.Config(YAML(typ='safe').load(config_path.read_text()))
        defaults = _resolve_config_profiles(_load_configs(), ['smac_vector', 'ma_jepa'])
        self.added_defaults = {k: v for k, v in defaults.flat.items() if k not in saved.flat}
        self.config = defaults.update(saved.flat)
        config = self.config
        self.map_name = config.task.removeprefix('smac_')
        self.gamma = 1 - 1 / config.agent.horizon if config.agent.contdisc else 1.0
        nn.COMPUTE_DTYPE = getattr(jnp, config.jax.compute_dtype)
        self.env_options = dict(difficulty=str(config.env.smac.difficulty),
                                continuing_episode=bool(config.env.smac.continuing_episode))
        env = SMACEnv(self.map_name, seed=170001, **self.env_options)
        self.obs_space = {k: v for k, v in env.obs_space.items() if not k.startswith('log/')}
        act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
        env.close()
        cfg = elements.Config(**config.agent, logdir=str(output), seed=config.seed,
            jax=config.jax, batch_size=config.batch_size, batch_length=config.batch_length,
            replay_context=config.replay_context, replay_sampling=config.replay.sampling,
            ppo_start_step=config.run.ppo_start_step, report_length=config.report_length,
            replica=0, replicas=1)
        model = object.__new__(MARLCore)
        MARLCore.__init__(model, self.obs_space, act_space, cfg)
        self.model = model
        weights, self.provenance = load_weights(checkpoint)
        self.params = jax.device_put(weights)
        self.host_weights = weights

        def pure_jit(fn):
            pure = nj.pure(fn)
            return jax.jit(lambda params, *args, key: pure(
                params, *args, seed=key, create=False, modify=False)[1])

        self.initial = pure_jit(lambda: (model.init_policy(1), model.ctde_joint.initial(1, model.team.size)))

        def infer(carry, observation):
            carry, _, _ = model.policy(carry, observation, mode='eval')
            features = model.team.fold_tree_batch({k: carry[1][k] for k in ('deter', 'stoch')})
            mask = model.team.fold_batch(observation['action_mask'])
            logits = model.policy_distribution(model.feat2tensor(features), 1, mask)['action'].logits
            context = {k: observation[v][:, None] for k, v in
                       [('present', 'agent_present'), ('controllable_alive', 'controllable_alive')]}
            values = model.critic({k: v[:, None] for k, v in features.items()}, 2,
                                  slow=True, context=context).pred()[:, 0]
            return carry, jax.nn.softmax(logits.astype(jnp.float32)), values
        self.infer_fn = pure_jit(infer)

        def joint_step(carry, central, observation, action):
            features = model.team.fold_tree_batch({k: carry[1][k] for k in ('deter', 'stoch')})
            central, _ = model.ctde_joint.step(central,
                model.team.unfold_batch(model.feat2tensor(features)), action[None],
                observation['agent_present'], observation['controllable_alive'],
                observation['is_first'], training=False)
            return central
        self.joint_fn = pure_jit(joint_step)

        def imagined(carry, central, observation, forced_action):
            local = carry[1]
            present, alive = observation['agent_present'], observation['controllable_alive']
            mask, reset = observation['action_mask'], observation['is_first']
            total = jnp.zeros_like(alive, dtype=jnp.float32)
            weight = jnp.ones_like(total)
            root_alive = alive
            for index in range(5):
                features = model.team.fold_tree_batch({k: local[k] for k in ('deter', 'stoch')})
                distribution = model.policy_distribution(model.feat2tensor(features), 1,
                    model.team.fold_batch(mask))['action']
                action = forced_action if index == 0 else distribution.sample(nj.seed())
                cache, deter = model.dyn.advance(model.team.fold_tree_batch(local),
                    {'action': action}, training=False, active=model.team.fold_batch(present))
                central, prediction = model.ctde_joint.step(central,
                    model.team.unfold_batch(model.feat2tensor(features)), action[None],
                    present, alive, reset, training=False)
                next_local, _ = model.dyn.complete_from_observation(cache, deter,
                    model.team.fold_batch(prediction['embedding']), sample=True)
                local = model.team.unfold_tree_batch(next_local)
                hidden = prediction['hidden']
                next_alive = alive & present & (model.ctde_alive(hidden, 2).prob(1) >= .5)
                reward, continuation = shared_team_outcomes(model.ctde_rew(hidden, 2).pred(),
                    model.ctde_con(hidden, 2).prob(1), present, alive, next_alive)
                total += weight * reward
                weight *= continuation
                output = model.ctde_mask(hidden, 2)
                mask = output.output.logit >= 0
                noop = jnp.zeros_like(mask).at[..., 0].set(True)
                mask = jnp.where(mask.any(-1, keepdims=True), mask, noop)
                mask = jnp.where(next_alive[..., None], mask, noop)
                alive, reset = next_alive, jnp.zeros_like(reset)
            features = model.team.fold_tree_batch({k: local[k] for k in ('deter', 'stoch')})
            values = model.critic({k: v[:, None] for k, v in features.items()}, 2, slow=True,
                context={'present': present[:, None], 'controllable_alive': alive[:, None]}).pred()
            total += weight * model.team.unfold_sequence(values)[:, 0]
            return (total * root_alive).sum() / jnp.maximum(root_alive.sum(), 1)
        self.imagined_fn = pure_jit(imagined)

    def env(self, seed):
        from majepa.envs.smac import SMACEnv
        return SMACEnv(self.map_name, seed=seed, **self.env_options)

    def start(self, params=None):
        import jax
        return self.initial(self.params if params is None else params, key=jax.random.PRNGKey(0))

    def observation(self, obs):
        return {k: np.asarray(obs[k])[None] for k in self.obs_space}

    def infer(self, carry, obs, seed, params=None):
        import jax
        return self.infer_fn(self.params if params is None else params, carry,
                            self.observation(obs), key=jax.random.PRNGKey(seed))


def with_action(carry, action):
    return (*carry[:3], {'action': np.asarray(action, np.int32)[None]})


def draw(probability, seed):
    probability = np.asarray(probability)
    uniform = np.random.default_rng(seed).random(len(probability))
    return (uniform[:, None] > np.cumsum(probability, -1)).sum(-1).clip(0, probability.shape[-1]-1).astype(np.int32)


def reset(env):
    return env.step({'reset': True, 'action': np.zeros(env.num_agents, np.int32)})


def step(env, action):
    return env.step({'reset': False, 'action': np.asarray(action, np.int32)})


def state_signature(env, obs):
    # Preserve the values too: array_equal below excludes approximate matches.
    return np.concatenate([np.asarray(env._env.get_state(), np.float32).ravel(),
        obs['observation'].ravel(), obs['action_mask'].astype(np.float32).ravel(),
        obs['controllable_alive'].astype(np.float32),
        np.asarray([obs['is_first'], obs['is_last']], np.float32)])


def make_roots(frozen, count, emit):
    import jax
    roots, banks = [], []
    for episode in range(count * 2):
        seed = 180001 + episode
        env = frozen.env(seed)
        try:
            obs = reset(env)
            carry, central = frozen.start()
            history, signatures, actions = [], [], []
            for t in range(int(env._env.episode_limit) + 1):
                history.append({k: np.asarray(v).copy() for k, v in obs.items()})
                signatures.append(state_signature(env, obs))
                carry, prob, _ = frozen.infer(carry, obs, 9001 + t)
                action = draw(prob, seed + 10000 + t)
                if t in (12, 24, 40, 64) and len(roots) < count and not obs['is_last']:
                    candidates = np.flatnonzero(obs['controllable_alive'] & (obs['action_mask'].sum(-1) > 1))
                    if len(candidates):
                        focal = int(candidates[len(roots) % len(candidates)])
                        roots.append(dict(seed=seed, time=t, focal=focal, history=list(history),
                            signatures=list(signatures), actions=list(actions),
                            carry=carry, central=central, root_action=action.copy()))
                if obs['is_last'] or len(roots) >= count:
                    banks.append(dict(history=list(history), actions=list(actions)))
                    break
                central = frozen.joint_fn(frozen.params, carry, central, frozen.observation(obs),
                    action, key=jax.random.PRNGKey(70001 + t))
                actions.append(action.copy())
                carry = with_action(carry, action)
                obs = step(env, action)
        finally:
            env.close()
        emit({'collection/roots': len(roots), 'collection/episodes': episode + 1})
        if len(roots) >= count:
            return roots, banks
    raise RuntimeError(f'Only {len(roots)} suitable roots found')


def real_branch(frozen, root, action, repeat):
    env = frozen.env(root['seed'])
    try:
        obs = reset(env)
        for t in range(root['time'] + 1):
            if not np.array_equal(state_signature(env, obs), root['signatures'][t]):
                return dict(matched=False, mismatch_time=t)
            if t < root['time']:
                obs = step(env, root['actions'][t])
        carry = with_action(root['carry'], action)
        rewards, masks, values = [], [], []
        for offset in range(int(env._env.episode_limit) + 1):
            obs = step(env, action)
            rewards.append(float(obs['reward'][0]))
            masks.append(obs['action_mask'].astype(np.int8).tolist())
            if obs['is_last']:
                break
            carry, prob, value = frozen.infer(carry, obs, 290001 + root['time'] + offset + repeat * 1000)
            values.append(float(np.asarray(value)[obs['controllable_alive']].mean())
                          if obs['controllable_alive'].any() else 0.0)
            action = draw(prob, 390001 + root['time'] + offset + repeat * 1000)
            carry = with_action(carry, action)
        if not obs['is_last']:
            raise RuntimeError('Real branch exceeded map episode limit')
        discount = frozen.gamma ** np.arange(len(rewards))
        return dict(matched=True, rewards=rewards, real_return=float(np.dot(discount, rewards)),
            reward5=float(np.dot(discount[:5], rewards[:5])),
            successor_bootstrap5=float(frozen.gamma**5 * values[4]) if len(rewards) > 5 else 0.0,
            win=float(obs['log/battle_won']), length=len(rewards), action_masks=masks)
    finally:
        env.close()


def ranking(frozen, roots, args, emit):
    import jax
    rows, groups = [], []
    for index, root in enumerate(roots):
        legal = np.flatnonzero(root['history'][-1]['action_mask'][root['focal']])
        chosen = int(root['root_action'][root['focal']])
        alternatives = [chosen] + [int(x) for x in legal if x != chosen]
        # Include an attack and movement alternative when both are legal.
        alternatives = list(dict.fromkeys([chosen] + [int(x) for x in legal if x >= 6][:1]
                           + [int(x) for x in legal if 2 <= x <= 5][:1] + alternatives))[:args.candidates]
        group = []
        for candidate in alternatives:
            action = root['root_action'].copy()
            action[root['focal']] = candidate
            q = [float(frozen.imagined_fn(frozen.params, root['carry'], root['central'],
                 frozen.observation(root['history'][-1]), action,
                 key=jax.random.PRNGKey(800001 + index * 100 + draw_index)))
                 for draw_index in range(args.imaginations)]
            branches = [real_branch(frozen, root, action, repeat) for repeat in range(args.repeats)]
            valid = [b for b in branches if b['matched']]
            row = dict(root=index, environment_seed=root['seed'], time=root['time'], focal=root['focal'],
                candidate=candidate, joint_action=action.tolist(), imagined_q_mean=float(np.mean(q)),
                imagined_q_std=float(np.std(q)), imagined_q=q, branches=branches,
                accepted=len(valid), rejected=len(branches)-len(valid))
            if len(valid) == args.repeats:
                row['real_return_mean'] = float(np.mean([b['real_return'] for b in valid]))
                group.append(row)
            rows.append(row)
            write(args.output / 'action_values.json', rows)
            emit({'ranking/candidates_completed': len(rows),
                  'ranking/matched_branches': sum(r['accepted'] for r in rows),
                  'ranking/rejected_branches': sum(r['rejected'] for r in rows)})
        if len(group) >= 2:
            predicted = np.array([r['imagined_q_mean'] for r in group])
            actual = np.array([r['real_return_mean'] for r in group])
            pairs = [(i,j) for i in range(len(group)) for j in range(i) if actual[i] != actual[j]]
            groups.append(dict(root=index, candidates=len(group),
                observed_regret=float(actual.max() - actual[predicted.argmax()]),
                pair_order_accuracy=float(np.mean([np.sign(predicted[i]-predicted[j]) ==
                    np.sign(actual[i]-actual[j]) for i,j in pairs])) if pairs else None))
            measured = [g['pair_order_accuracy'] for g in groups if g['pair_order_accuracy'] is not None]
            emit({'ranking/roots_completed':len(groups),
                  'ranking/observed_regret_mean':float(np.mean([g['observed_regret'] for g in groups])),
                  **({'ranking/pair_order_accuracy':float(np.mean(measured))} if measured else {})})
    if not groups:
        raise RuntimeError('No root has two fully matched real action alternatives; ranking is unavailable')
    return dict(roots=groups, accepted_branches=sum(r['accepted'] for r in rows),
                rejected_branches=sum(r['rejected'] for r in rows),
                note='Exploratory ranking of sampled single-agent alternatives with fixed teammate root actions. Few real repeats; observed regret is not noise-corrected. Prefix matches verify observable simulator state, not hidden SC2 RNG.')


def drift(frozen, newer_path, banks, emit):
    import jax
    newer, provenance = load_weights(newer_path)
    old = frozen.host_weights
    for key in old:
        if key not in newer or old[key].shape != newer[key].shape:
            raise ValueError(f'Temporal policy checkpoints have incompatible parameters: {key}')
    world = lambda key: key.startswith(('enc/', 'dyn/'))
    actor = lambda key: key.startswith(('pol/', 'ctde_teammate_actor/'))
    variants = {
        'old': frozen.params,
        'world_only': jax.device_put({k: newer[k] if world(k) else v for k,v in old.items()}),
        'actor_only': jax.device_put({k: newer[k] if actor(k) else v for k,v in old.items()}),
        'both': jax.device_put({k: newer[k] if world(k) or actor(k) else v for k,v in old.items()}),
    }
    rows = []
    for episode, bank in enumerate(banks):
        outputs = {}
        # A repeated old checkpoint is the negative control for RNG isolation.
        for label, params in {**variants, 'old_repeat': frozen.params}.items():
            carry, _ = frozen.start(params)
            probabilities = []
            for t, obs in enumerate(bank['history']):
                carry, probability, _ = frozen.infer(carry, obs, 9001+t, params)
                probabilities.append(np.asarray(probability))
                if t < len(bank['actions']):
                    carry = with_action(carry, bank['actions'][t])
            outputs[label] = np.stack(probabilities)
        if not np.array_equal(outputs['old'], outputs['old_repeat']):
            raise RuntimeError('Identical checkpoint/history/RNG produced different policy probabilities')
        alive = np.stack([o['controllable_alive'] for o in bank['history']])
        oldp = outputs['old'][alive]
        for label in ('world_only', 'actor_only', 'both'):
            p = outputs[label][alive]
            kl = np.sum(oldp * (np.log(np.maximum(oldp,1e-30))-np.log(np.maximum(p,1e-30))),-1)
            rows.append(dict(episode=episode, component=label, kl_mean=float(kl.mean()),
                kl_p95=float(np.quantile(kl,.95)), greedy_flip=float(np.mean(p.argmax(-1)!=oldp.argmax(-1))),
                state_agent_pairs=len(p)))
        emit({'drift/episodes_completed': episode+1})
        for label in ('world_only', 'actor_only', 'both'):
            selected = [r for r in rows if r['component'] == label]
            weights = [r['state_agent_pairs'] for r in selected]
            emit({f'drift/{label}/kl_mean':float(np.average([r['kl_mean'] for r in selected],weights=weights)),
                  f'drift/{label}/greedy_flip':float(np.average([r['greedy_flip'] for r in selected],weights=weights))})
    return dict(old=frozen.provenance, newer=provenance, rows=rows,
        identical_checkpoint_negative_control='exactly equal',
        note='Same raw histories, actual preceding actions, posterior RNG, and legal masks. Encoder/history-only and actor-only checkpoint hybrids separate component contributions; they are not additive. An interval comparison is not a single-update causal attribution.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--compare-checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wandb-id', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--roots', type=int, default=8)
    parser.add_argument('--candidates', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--imaginations', type=int, default=16)
    parser.add_argument('--drift-only', action='store_true')
    args = parser.parse_args()
    if min(args.roots,args.candidates,args.repeats,args.imaginations) < 1:
        parser.error('Diagnostic budgets must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    import wandb
    status = dict(phase='loading', started_at=time.time(),
                  benchmark_eligible=False,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  **{k:str(v) for k,v in vars(args).items()})
    run = wandb.init(entity='osaze-obahor', project='majepa-ppo-treatments', group=GROUP,
        id=args.wandb_id, name=args.wandb_id, job_type='diagnostics', config=status,
        tags=['diagnostic','frozen','L40'])
    def emit(values):
        status.update(updated_at=time.time(), **values)
        write(args.output / 'status.json',status)
        run.log(values)
        print(json.dumps(values),flush=True)
    try:
        frozen = Frozen(args.config, args.checkpoint, args.output)
        write(args.output/'provenance.json', dict(model=frozen.provenance,
            script_sha256=status['script_sha256'],
            default_fields_added=frozen.added_defaults, environment=frozen.env_options,
            gamma=frozen.gamma, policy='stochastic legal policy, no collection unimix'))
        emit({'phase':'collecting_real_histories'})
        roots, banks = make_roots(frozen,args.roots,emit)
        # Raw histories remain available for reanalysis; no device arrays in bank.
        with gzip.open(args.output/'histories.pkl.gz','wb') as f:
            pickle.dump(banks,f)
        result = {}
        if not args.drift_only:
            emit({'phase':'matched_real_action_branches'})
            result['ranking'] = ranking(frozen,roots,args,emit)
        if args.compare_checkpoint:
            emit({'phase':'policy_component_drift'})
            result['drift'] = drift(frozen,args.compare_checkpoint,banks,emit)
        # Frozen nj.pure calls disallow writes and tensor creation.
        write(args.output/'summary.json',result)
        for filename in ('summary.json','provenance.json','action_values.json','histories.pkl.gz'):
            path = args.output / filename
            if path.exists():
                run.save(str(path), base_path=str(args.output), policy='now')
        emit({'phase':'complete'})
        run.summary.update({'completed':True, **result})
        write(args.output/'done.json',{'completed':True})
    except BaseException as error:
        emit({'phase':'failed','error':repr(error)})
        raise
    finally:
        run.finish(exit_code=int(status['phase']=='failed'))


if __name__ == '__main__':
    main()
