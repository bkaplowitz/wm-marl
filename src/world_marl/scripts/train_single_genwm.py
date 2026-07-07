"""Train PPO inside a generative world model on a single-agent environment.

Model-based counterpart of ``train_dmc_jepa`` for the generative arms: fit a
conditional next-observation model (``discrete-transformer`` CTMC flow,
``continuous-transformer`` flow, or ``llada2`` block diffusion) plus a learned
reward/continue head on real replay transitions, then train the policy entirely
on imagined rollouts. Quantile tokenizers are fit once on the initial random
replay and stay frozen (refitting would silently re-map every token id the
model has already learned); later out-of-range observations clip into the edge
bins.

``--collect-steps``/``--online-collect-steps`` are per-env steps (total
transitions = steps x ``--num-envs``), the same units as ``train_dmc_jepa``.
On gymnax/brax, collection and evaluation run on-device through the adapters'
``scan_rollout``; DMC falls back to the host step loop.

The extra ``model-free`` arm skips the world model entirely and spends the
identical real-step budget on PPO over real rollouts — the baseline the
model-based arms must beat on sample efficiency.

Runs ``--num-runs`` seeded runs and applies an improvement gate in
``summarize()``. Exit code 1 means the gate failed, not that the program
crashed. Artifacts mirror ``train_dmc_jepa``: ``config.json`` at the experiment
root, ``outcome.json`` per run, ``summary.json`` at the root, so a policy-level
comparison harness can glob both layouts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from world_marl.genwm import (
    GENWM_ARMS,
    GenWMConfig,
    PPOConfig,
    QuantileTokenizer,
    create_genwm_state,
    create_head_state,
    create_policy_state,
    fit_quantile_tokenizer,
    genwm_train_step,
    head_train_step,
    imagined_rollout,
    ppo_update,
)
from world_marl.world_model_training import _replay_scan_episode_bookkeeping

MODEL_FREE_ARM = "model-free"
ARM_CHOICES = (*GENWM_ARMS, MODEL_FREE_ARM)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        required=True,
        help="dmc:<domain>/<task>, brax:<env>, or gymnax:<env_id>",
    )
    parser.add_argument("--arm", required=True, choices=ARM_CHOICES)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/genwm"))
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--max-cycles", type=int, default=500)
    parser.add_argument(
        "--brax-backend",
        default=None,
        help="Optional Brax physics backend to pass through to brax.envs.create.",
    )
    # Real-env and update budgets (defaults match the JEPA mainline preset).
    parser.add_argument(
        "--collect-steps",
        type=int,
        default=8192,
        help="Per-env real steps for the initial random collection "
        "(total transitions = steps x --num-envs, matching train_dmc_jepa).",
    )
    parser.add_argument("--train-steps", type=int, default=12000)
    parser.add_argument("--policy-train-steps", type=int, default=3000)
    parser.add_argument("--online-iterations", type=int, default=6)
    parser.add_argument(
        "--online-collect-steps",
        type=int,
        default=4096,
        help="Per-env real steps collected per online iteration.",
    )
    parser.add_argument("--online-train-steps", type=int, default=3000)
    parser.add_argument("--online-policy-train-steps", type=int, default=750)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--policy-batch-size", type=int, default=512)
    parser.add_argument("--imag-horizon", type=int, default=15)
    # World-model capacity (defaults match the JEPA mainline preset).
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--wm-learning-rate", type=float, default=1e-3)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--integration-steps", type=int, default=8)
    parser.add_argument("--obs-bins", type=int, default=32)
    parser.add_argument("--action-bins", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--steps-per-block", type=int, default=4)
    # PPO.
    parser.add_argument(
        "--mf-rollout-steps",
        type=int,
        default=128,
        help="model-free arm: on-policy segment length per PPO update "
        "(segment x num-envs must divide evenly into --ppo-num-minibatches).",
    )
    parser.add_argument("--ppo-learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-update-epochs", type=int, default=4)
    parser.add_argument("--ppo-num-minibatches", type=int, default=4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    # Evaluation and gate.
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="Exit 0 even when the improvement gate fails.",
    )
    parser.add_argument("--quiet", action="store_true")
    # Weights & Biases (disabled unless --wandb-project is set).
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    args = parser.parse_args(argv)
    if not args.env.startswith(("dmc:", "brax:", "gymnax:")):
        parser.error(
            "--env must be formatted as dmc:<domain>/<task>, brax:<env>, "
            "or gymnax:<env_id>"
        )
    if args.arm == MODEL_FREE_ARM and args.env.startswith("dmc:"):
        parser.error("the model-free arm needs a scan_rollout adapter (gymnax/brax)")
    return args


def _make_adapter(args: argparse.Namespace, *, seed: int):
    if args.env.startswith("dmc:"):
        from world_marl.envs.dmc_adapter import DMCVectorAdapter, dmc_env_name

        return DMCVectorAdapter(
            dmc_env_name(args.env),
            num_envs=args.num_envs,
            max_cycles=args.max_cycles,
            seed=seed,
        )
    if args.env.startswith("brax:"):
        from world_marl.envs.brax_adapter import BraxVectorAdapter, brax_env_name

        return BraxVectorAdapter(
            brax_env_name(args.env),
            num_envs=args.num_envs,
            max_cycles=args.max_cycles,
            seed=seed,
            backend=args.brax_backend,
        )
    from world_marl.envs.gymnax_adapter import GymnaxVectorAdapter, gymnax_env_name

    return GymnaxVectorAdapter(
        gymnax_env_name(args.env),
        num_envs=args.num_envs,
        max_cycles=args.max_cycles,
        seed=seed,
    )


@jax.jit
def _sample_policy_actions(policy_state, key, observations):
    policy, _ = policy_state.apply_fn({"params": policy_state.params}, observations)
    return policy.sample(seed=key)


@jax.jit
def _mode_policy_actions(policy_state, observations):
    policy, _ = policy_state.apply_fn({"params": policy_state.params}, observations)
    return policy.mode()


def _flat_observations(observations: np.ndarray, num_envs: int) -> np.ndarray:
    return np.asarray(observations, dtype=np.float32)[:, 0].reshape((num_envs, -1))


def _shape_for_step(actions: np.ndarray, action_mode: str) -> np.ndarray:
    if action_mode == "discrete":
        return actions.reshape((-1, 1))
    return actions[:, None, :]


def _policy_actions(
    policy_state,
    flat_obs: np.ndarray,
    key: jax.Array | None,
    *,
    action_mode: str,
    action_low: np.ndarray | None,
    action_high: np.ndarray | None,
) -> np.ndarray:
    if key is None:
        actions = np.asarray(_mode_policy_actions(policy_state, jnp.asarray(flat_obs)))
    else:
        actions = np.asarray(
            _sample_policy_actions(policy_state, key, jnp.asarray(flat_obs))
        )
    if action_mode == "discrete":
        return actions.astype(np.int32)
    return np.clip(actions.astype(np.float32), action_low, action_high)


def _make_scan_action_fns(adapter, action_mode: str) -> dict[str, Any]:
    """Action callbacks in ``scan_rollout``'s 3-arg shape.

    ``scan_rollout`` caches compiled programs by ``id(fn)``, so one set is
    created per run and reused across every collection/eval call.
    """
    action_dim = adapter.action_dim
    if action_mode == "discrete":

        def random_fn(_state, key, obs_flat):
            rows = obs_flat.shape[0]
            actions = jax.random.randint(key, (rows,), 0, action_dim)
            zeros = jnp.zeros((rows,), dtype=jnp.float32)
            return actions.astype(jnp.int32), zeros, zeros, zeros

    else:
        action_low = jnp.asarray(adapter.action_low, dtype=jnp.float32)
        action_high = jnp.asarray(adapter.action_high, dtype=jnp.float32)

        def random_fn(_state, key, obs_flat):
            rows = obs_flat.shape[0]
            actions = jax.random.uniform(
                key, (rows, action_dim), minval=action_low, maxval=action_high
            )
            zeros = jnp.zeros((rows,), dtype=jnp.float32)
            return actions, zeros, zeros, zeros

    def sample_fn(train_state, key, obs_flat):
        policy, values = train_state.apply_fn({"params": train_state.params}, obs_flat)
        actions = policy.sample(seed=key)
        return actions, policy.log_prob(actions), values, policy.entropy()

    def mode_fn(train_state, _key, obs_flat):
        policy, values = train_state.apply_fn({"params": train_state.params}, obs_flat)
        actions = policy.mode()
        return actions, policy.log_prob(actions), values, policy.entropy()

    return {"random": random_fn, "sample": sample_fn, "mode": mode_fn}


def _scan_collect_transitions(
    adapter,
    action_fn,
    policy_state,
    *,
    steps_per_env: int,
    key: jax.Array,
) -> dict[str, np.ndarray]:
    """On-device twin of ``_collect_transitions`` via ``scan_rollout``.

    Next observations are the observation sequence shifted by one plus the
    final carry, which reproduces the host loop's post-reset next-obs at
    terminal steps.
    """
    observations = adapter.reset()
    ys, last_obs_flat = adapter.scan_rollout(
        action_fn,
        policy_state,
        steps_per_env,
        policy_key=key,
        observations=observations,
    )
    obs_seq, action_seq, _lp, _values, _ent, reward_seq, done_seq = ys
    _replay_scan_episode_bookkeeping(adapter, ys, steps_per_env)
    obs = np.asarray(obs_seq, dtype=np.float32)
    next_obs = np.concatenate(
        [obs[1:], np.asarray(last_obs_flat, dtype=np.float32)[None]], axis=0
    )
    total = obs.shape[0] * obs.shape[1]
    actions = np.asarray(action_seq)
    return {
        "observations": obs.reshape((total, -1)),
        "actions": actions.reshape((total, *actions.shape[2:])),
        "rewards": np.asarray(reward_seq, dtype=np.float32).reshape((total,)),
        "dones": np.asarray(done_seq, dtype=np.float32).reshape((total,)),
        "next_observations": next_obs.reshape((total, -1)),
    }


def _scan_eval_return(
    adapter,
    action_fn,
    policy_state,
    *,
    episodes: int,
    key: jax.Array,
) -> float:
    """Scan-based twin of ``_evaluate_policy`` with the same total step bound."""
    returns: list[float] = []
    observations = adapter.reset()
    block = max(1, adapter.max_cycles)
    total_step_budget = max(1, episodes * adapter.max_cycles * 4 // adapter.num_envs)
    max_blocks = max(1, -(-total_step_budget // block))
    for _ in range(max_blocks):
        key, block_key = jax.random.split(key)
        ys, last_obs_flat = adapter.scan_rollout(
            action_fn,
            policy_state,
            block,
            policy_key=block_key,
            observations=observations,
        )
        completed_returns, _ = _replay_scan_episode_bookkeeping(adapter, ys, block)
        returns.extend(float(item[0]) for item in completed_returns)
        observations = np.asarray(last_obs_flat, dtype=np.float32)
        if len(returns) >= episodes:
            break
    if not returns:
        return float("nan")
    return float(np.mean(returns[:episodes]))


def _collect_replay(
    adapter,
    *,
    steps_per_env: int,
    rng: np.random.Generator,
    collect_key: jax.Array,
    action_fns: dict[str, Any] | None,
    policy_state,
    action_mode: str,
) -> dict[str, np.ndarray]:
    if action_fns is not None:
        fn = action_fns["random"] if policy_state is None else action_fns["sample"]
        return _scan_collect_transitions(
            adapter, fn, policy_state, steps_per_env=steps_per_env, key=collect_key
        )
    return _collect_transitions(
        adapter,
        num_steps=steps_per_env,
        rng=rng,
        policy_state=policy_state,
        policy_key=collect_key if policy_state is not None else None,
        action_mode=action_mode,
    )


def _eval_return(
    adapter,
    *,
    episodes: int,
    rng: np.random.Generator,
    eval_key: jax.Array,
    action_fns: dict[str, Any] | None,
    policy_state,
    action_mode: str,
) -> float:
    if action_fns is not None:
        fn = action_fns["random"] if policy_state is None else action_fns["mode"]
        return _scan_eval_return(
            adapter, fn, policy_state, episodes=episodes, key=eval_key
        )
    return _evaluate_policy(
        adapter,
        policy_state,
        episodes=episodes,
        action_mode=action_mode,
        rng=rng,
    )


def _collect_transitions(
    adapter,
    *,
    num_steps: int,
    rng: np.random.Generator,
    policy_state=None,
    policy_key: jax.Array | None = None,
    action_mode: str,
) -> dict[str, np.ndarray]:
    """Collect ``num_steps`` per-env steps on the host (random when state is None)."""
    num_envs = adapter.num_envs
    iterations = max(1, num_steps)
    observations = adapter.reset()
    records: dict[str, list[np.ndarray]] = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "next_observations": [],
    }
    for _ in range(iterations):
        flat = _flat_observations(observations, num_envs)
        if policy_state is None:
            step_actions = adapter.sample_actions(rng)
            actions = step_actions[:, 0]
        else:
            assert policy_key is not None
            policy_key, action_key = jax.random.split(policy_key)
            actions = _policy_actions(
                policy_state,
                flat,
                action_key,
                action_mode=action_mode,
                action_low=adapter.action_low,
                action_high=adapter.action_high,
            )
            step_actions = _shape_for_step(actions, action_mode)
        step = adapter.step(step_actions)
        records["observations"].append(flat)
        records["actions"].append(np.asarray(actions))
        records["rewards"].append(np.asarray(step.rewards[:, 0], dtype=np.float32))
        records["dones"].append(np.asarray(step.dones[:, 0], dtype=np.float32))
        records["next_observations"].append(
            _flat_observations(step.observations, num_envs)
        )
        observations = step.observations
    return {name: np.concatenate(chunks) for name, chunks in records.items()}


def _evaluate_policy(
    adapter,
    policy_state,
    *,
    episodes: int,
    action_mode: str,
    rng: np.random.Generator,
) -> float:
    """Mean return over completed episodes (random policy when state is None)."""
    returns: list[float] = []
    observations = adapter.reset()
    step_limit = max(1, episodes * adapter.max_cycles * 4 // adapter.num_envs)
    for _ in range(step_limit):
        if policy_state is None:
            step_actions = adapter.sample_actions(rng)
        else:
            flat = _flat_observations(observations, adapter.num_envs)
            actions = _policy_actions(
                policy_state,
                flat,
                None,
                action_mode=action_mode,
                action_low=adapter.action_low,
                action_high=adapter.action_high,
            )
            step_actions = _shape_for_step(actions, action_mode)
        step = adapter.step(step_actions)
        returns.extend(float(total[0]) for total in step.completed_returns)
        observations = step.observations
        if len(returns) >= episodes:
            break
    if not returns:
        return float("nan")
    return float(np.mean(returns[:episodes]))


def _action_feature_array(actions: np.ndarray, config: GenWMConfig) -> np.ndarray:
    if config.action_mode == "discrete":
        return np.eye(config.action_dim, dtype=np.float32)[actions.astype(np.int64)]
    return actions.astype(np.float32)


def _fit_models(
    wm_state,
    head_state,
    key: jax.Array,
    data: dict[str, np.ndarray],
    obs_tokenizer: QuantileTokenizer,
    action_tokenizer: QuantileTokenizer | None,
    config: GenWMConfig,
    *,
    steps: int,
    batch_size: int,
    rng: np.random.Generator,
    log_every: int,
    quiet: bool,
    label: str,
):
    """Interleave world-model and reward/continue-head updates on replay data.

    The world model only sees non-terminal transitions (the stored next
    observation at a terminal step is post-reset); the head trains on every
    transition since its targets (r, done) belong to (s, a).
    """
    non_terminal = np.flatnonzero(data["dones"] < 0.5)
    if non_terminal.size == 0:
        raise RuntimeError("replay contains only terminal transitions")
    features = _action_feature_array(data["actions"], config)
    continues = 1.0 - data["dones"]
    total = data["observations"].shape[0]
    wm_loss = float("nan")
    head_metrics: dict[str, Any] = {}
    for step_index in range(steps):
        wm_index = non_terminal[rng.integers(0, non_terminal.size, size=batch_size)]
        key, step_key = jax.random.split(key)
        wm_state, loss = genwm_train_step(
            wm_state,
            step_key,
            jnp.asarray(data["observations"][wm_index]),
            jnp.asarray(data["actions"][wm_index]),
            jnp.asarray(data["next_observations"][wm_index]),
            obs_tokenizer,
            action_tokenizer,
            config,
        )
        head_index = rng.integers(0, total, size=batch_size)
        head_state, metrics = head_train_step(
            head_state,
            jnp.asarray(data["observations"][head_index]),
            jnp.asarray(features[head_index]),
            jnp.asarray(data["rewards"][head_index]),
            jnp.asarray(continues[head_index]),
        )
        if step_index % log_every == 0 or step_index == steps - 1:
            wm_loss = float(loss)
            head_metrics = {name: float(value) for name, value in metrics.items()}
            if not quiet:
                print(
                    f"[{label}] step {step_index + 1}/{steps} "
                    f"wm_loss={wm_loss:.4f} "
                    f"head_loss={head_metrics['head_total_loss']:.4f}",
                    flush=True,
                )
    return wm_state, head_state, wm_loss, head_metrics


def _train_policy(
    policy_state,
    wm_state,
    head_state,
    obs_tokenizer: QuantileTokenizer,
    action_tokenizer: QuantileTokenizer | None,
    start_pool: np.ndarray,
    key: jax.Array,
    config: GenWMConfig,
    ppo_config: PPOConfig,
    *,
    steps: int,
    batch_size: int,
    horizon: int,
    rng: np.random.Generator,
    log_every: int,
    quiet: bool,
    label: str,
):
    metrics: dict[str, Any] = {}
    for step_index in range(steps):
        starts = jnp.asarray(
            start_pool[rng.integers(0, start_pool.shape[0], size=batch_size)]
        )
        key, rollout_key, update_key = jax.random.split(key, 3)
        batch, last_values = imagined_rollout(
            policy_state,
            wm_state,
            head_state,
            obs_tokenizer,
            action_tokenizer,
            starts,
            rollout_key,
            horizon=horizon,
            config=config,
            ppo_config=ppo_config,
        )
        policy_state, step_metrics = ppo_update(
            policy_state, batch, last_values, update_key, ppo_config
        )
        if step_index % log_every == 0 or step_index == steps - 1:
            metrics = {name: float(value) for name, value in step_metrics.items()}
            if not quiet:
                print(
                    f"[{label}] step {step_index + 1}/{steps} "
                    f"ppo_loss={metrics['total_loss']:.4f} "
                    f"entropy={metrics['entropy']:.4f}",
                    flush=True,
                )
    return policy_state, metrics


def run_one(args: argparse.Namespace, *, run_dir: Path, run_index: int) -> dict:
    started = time.time()
    seed = args.seed + 10_000 * run_index
    rng = np.random.default_rng(seed)
    adapter = _make_adapter(args, seed=seed)
    try:
        action_mode = "discrete" if args.env.startswith("gymnax:") else "continuous"
        obs_dim = int(np.prod(adapter.observation_shape))
        config = GenWMConfig(
            arm=args.arm,
            obs_dim=obs_dim,
            action_dim=adapter.action_dim,
            action_mode=action_mode,
            obs_bins=args.obs_bins,
            action_bins=args.action_bins,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            mlp_ratio=args.mlp_ratio,
            learning_rate=args.wm_learning_rate,
            integration_steps=args.integration_steps,
            block_size=args.block_size,
            steps_per_block=args.steps_per_block,
        )
        ppo_config = PPOConfig(
            learning_rate=args.ppo_learning_rate,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            update_epochs=args.ppo_update_epochs,
            num_minibatches=args.ppo_num_minibatches,
        )
        key = jax.random.PRNGKey(seed)
        key, wm_key, head_key, policy_key = jax.random.split(key, 4)
        wm_state = create_genwm_state(wm_key, config)
        head_state = create_head_state(
            head_key, config, learning_rate=args.head_learning_rate
        )
        policy_state = create_policy_state(policy_key, config, ppo_config)

        if not args.quiet:
            print(f"[run {run_index}] collecting {args.collect_steps} random steps")
        data = _collect_transitions(
            adapter,
            num_steps=args.collect_steps,
            rng=rng,
            action_mode=action_mode,
        )
        obs_tokenizer = fit_quantile_tokenizer(
            np.concatenate([data["observations"], data["next_observations"]]),
            args.obs_bins,
        )
        action_tokenizer = None
        if action_mode == "continuous":
            action_tokenizer = fit_quantile_tokenizer(data["actions"], args.action_bins)

        random_return = _evaluate_policy(
            adapter,
            None,
            episodes=args.eval_episodes,
            action_mode=action_mode,
            rng=rng,
        )
        initial_return = _evaluate_policy(
            adapter,
            policy_state,
            episodes=args.eval_episodes,
            action_mode=action_mode,
            rng=rng,
        )
        if not args.quiet:
            print(
                f"[run {run_index}] random={random_return:.2f} "
                f"initial={initial_return:.2f}"
            )

        log_every = max(1, args.train_steps // 10)
        key, fit_key = jax.random.split(key)
        wm_state, head_state, wm_loss, head_metrics = _fit_models(
            wm_state,
            head_state,
            fit_key,
            data,
            obs_tokenizer,
            action_tokenizer,
            config,
            steps=args.train_steps,
            batch_size=args.batch_size,
            rng=rng,
            log_every=log_every,
            quiet=args.quiet,
            label=f"run {run_index} fit",
        )
        key, policy_fit_key = jax.random.split(key)
        policy_state, ppo_metrics = _train_policy(
            policy_state,
            wm_state,
            head_state,
            obs_tokenizer,
            action_tokenizer,
            data["observations"],
            policy_fit_key,
            config,
            ppo_config,
            steps=args.policy_train_steps,
            batch_size=args.policy_batch_size,
            horizon=args.imag_horizon,
            rng=rng,
            log_every=max(1, args.policy_train_steps // 10),
            quiet=args.quiet,
            label=f"run {run_index} policy",
        )

        iteration_returns: list[float] = []
        for iteration in range(args.online_iterations):
            key, collect_key = jax.random.split(key)
            fresh = _collect_transitions(
                adapter,
                num_steps=args.online_collect_steps,
                rng=rng,
                policy_state=policy_state,
                policy_key=collect_key,
                action_mode=action_mode,
            )
            data = {name: np.concatenate([data[name], fresh[name]]) for name in data}
            key, fit_key, policy_fit_key = jax.random.split(key, 3)
            wm_state, head_state, wm_loss, head_metrics = _fit_models(
                wm_state,
                head_state,
                fit_key,
                data,
                obs_tokenizer,
                action_tokenizer,
                config,
                steps=args.online_train_steps,
                batch_size=args.batch_size,
                rng=rng,
                log_every=max(1, args.online_train_steps // 5),
                quiet=args.quiet,
                label=f"run {run_index} online {iteration} fit",
            )
            policy_state, ppo_metrics = _train_policy(
                policy_state,
                wm_state,
                head_state,
                obs_tokenizer,
                action_tokenizer,
                data["observations"],
                policy_fit_key,
                config,
                ppo_config,
                steps=args.online_policy_train_steps,
                batch_size=args.policy_batch_size,
                horizon=args.imag_horizon,
                rng=rng,
                log_every=max(1, args.online_policy_train_steps // 5),
                quiet=args.quiet,
                label=f"run {run_index} online {iteration} policy",
            )
            iteration_return = _evaluate_policy(
                adapter,
                policy_state,
                episodes=args.eval_episodes,
                action_mode=action_mode,
                rng=rng,
            )
            iteration_returns.append(iteration_return)
            if not args.quiet:
                print(
                    f"[run {run_index}] online iteration {iteration}: "
                    f"return={iteration_return:.2f}"
                )

        trained_return = _evaluate_policy(
            adapter,
            policy_state,
            episodes=args.eval_episodes,
            action_mode=action_mode,
            rng=rng,
        )
    finally:
        adapter.close()

    real_env_steps = args.collect_steps + args.online_iterations * (
        args.online_collect_steps
    )
    outcome = {
        "run_index": run_index,
        "seed": seed,
        "arm": args.arm,
        "env": args.env,
        "action_mode": action_mode,
        "obs_dim": obs_dim,
        "action_dim": int(adapter.action_dim),
        "policy_random_mean": random_return,
        "policy_initial_mean": initial_return,
        "policy_trained_mean": trained_return,
        "policy_iteration_returns": iteration_returns,
        "world_model_final_loss": wm_loss,
        "head_final_metrics": head_metrics,
        "ppo_final_metrics": ppo_metrics,
        "real_env_steps": real_env_steps,
        "replay_transitions": int(data["observations"].shape[0]),
        "runtime_seconds": time.time() - started,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "outcome.json").write_text(json.dumps(outcome, indent=2))
    return outcome


def summarize(outcomes: list[dict], *, min_improvement: float) -> dict:
    trained = np.array([o["policy_trained_mean"] for o in outcomes])
    random_ = np.array([o["policy_random_mean"] for o in outcomes])
    initial = np.array([o["policy_initial_mean"] for o in outcomes])
    baseline = max(float(random_.mean()), float(initial.mean()))
    improvement = float(trained.mean()) - baseline
    passed = bool(
        np.all(np.isfinite(trained))
        and np.isfinite(improvement)
        and improvement >= min_improvement
    )
    return {
        "num_runs": len(outcomes),
        "policy_random_mean": float(random_.mean()),
        "policy_initial_mean": float(initial.mean()),
        "policy_trained_mean": float(trained.mean()),
        "improvement_over_baseline": improvement,
        "min_improvement": min_improvement,
        "passed": passed,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    env_slug = args.env.replace(":", "_").replace("/", "_")
    experiment_dir = args.out_dir / f"genwm_{env_slug}_{args.arm}_{stamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        name: (str(value) if isinstance(value, Path) else value)
        for name, value in vars(args).items()
    }
    (experiment_dir / "config.json").write_text(json.dumps(config_payload, indent=2))
    print(f"experiment dir: {experiment_dir}", flush=True)

    outcomes = []
    for run_index in range(args.num_runs):
        outcomes.append(
            run_one(
                args,
                run_dir=experiment_dir / f"run_{run_index:02d}",
                run_index=run_index,
            )
        )

    summary = summarize(outcomes, min_improvement=args.min_improvement)
    (experiment_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["passed"] and not args.allow_fail:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
