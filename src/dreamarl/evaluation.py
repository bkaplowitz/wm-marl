"""Fixed-checkpoint evaluation with explicit deterministic and sampled protocols."""

from __future__ import annotations

import json
from collections import defaultdict
from contextlib import contextmanager
from functools import partial as bind

import elements
import embodied
import numpy as np


@contextmanager
def _preserve_policy_state(agent):
    """Keep diagnostic calls from consuming RNG or pending parameter sync."""

    counter = getattr(agent, "n_actions", None)
    policy_lock = getattr(agent, "policy_lock", None)
    if counter is None or policy_lock is None:
        yield
        return
    with policy_lock:
        with counter.lock:
            saved_counter = counter.value
        has_pending = hasattr(agent, "pending_sync")
        saved_pending = agent.pending_sync if has_pending else None
        if has_pending:
            agent.pending_sync = None
    try:
        yield
    finally:
        with policy_lock:
            with counter.lock:
                counter.value = saved_counter
            if has_pending:
                if agent.pending_sync is not None:
                    raise RuntimeError(
                        "policy synchronization changed concurrently during inline "
                        "evaluation"
                    )
                agent.pending_sync = saved_pending


def evaluate_current_policy(
    agent,
    make_env,
    *,
    episodes: int,
    envs: int,
    debug: bool,
    worker_offset: int = 10_000,
    policy_mode: str = "eval",
) -> dict[str, object]:
    """Evaluate one frozen policy without touching replay or training steps."""

    if episodes < 1:
        raise ValueError("evaluation requires at least one episode")
    if policy_mode not in {"eval", "eval_sample"}:
        raise ValueError(f"unsupported evaluation policy mode: {policy_mode!r}")
    returns = []
    team_returns = []
    agent_returns = []
    accumulators = defaultdict(elements.Agg)

    def log_transition(transition, worker):
        if len(returns) >= episodes:
            return
        episode = accumulators[worker]
        transition["is_first"] and episode.reset()
        rewards = np.asarray(transition["reward"], np.float32)
        episode.add("score", np.float32(rewards.mean()), agg="sum")
        episode.add("team_return_sum", np.float32(rewards.sum()), agg="sum")
        episode.add("agent_scores", rewards, agg="sum")
        if transition["is_last"]:
            result = episode.result()
            returns.append(float(result["score"]))
            team_returns.append(float(result["team_return_sum"]))
            agent_returns.append(np.asarray(result["agent_scores"], np.float32))

    environments = min(envs, episodes)
    functions = [bind(make_env, worker_offset + index) for index in range(environments)]
    driver = embodied.Driver(functions, parallel=not debug)
    driver.on_step(log_transition)

    def policy(*values):
        return agent.policy(*values, mode=policy_mode)

    try:
        with _preserve_policy_state(agent):
            driver.reset(agent.init_policy)
            while len(returns) < episodes:
                driver(policy, steps=10)
    finally:
        driver.close()

    returns_array = np.asarray(returns[:episodes], np.float32)
    team_returns_array = np.asarray(team_returns[:episodes], np.float32)
    agent_array = np.stack(agent_returns[:episodes])
    return {
        "episodes": int(episodes),
        "return_mean": float(returns_array.mean()),
        "return_median": float(np.median(returns_array)),
        "return_std": float(returns_array.std()),
        "return_min": float(returns_array.min()),
        "return_max": float(returns_array.max()),
        "return_p10": float(np.percentile(returns_array, 10)),
        "per_agent_return_mean": float(returns_array.mean()),
        "per_agent_return_median": float(np.median(returns_array)),
        "per_agent_return_std": float(returns_array.std()),
        "per_agent_return_p10": float(np.percentile(returns_array, 10)),
        "team_return_mean": float(team_returns_array.mean()),
        "team_return_median": float(np.median(team_returns_array)),
        "team_return_std": float(team_returns_array.std()),
        "team_return_p10": float(np.percentile(team_returns_array, 10)),
        "agent_return_mean": float(agent_array.mean()),
        "agent_return_std": float(agent_array.std()),
        "returns": returns_array.tolist(),
        "team_returns": team_returns_array.tolist(),
        "per_agent_returns": agent_array.tolist(),
    }


def eval_only(make_agent, make_env, make_logger, args):
    if not args.from_checkpoint:
        raise ValueError("evaluation requires run.from_checkpoint")
    if args.eval_eps < 1:
        raise ValueError("evaluation requires at least one episode")
    if args.eval_policy_mode not in {"eval", "eval_sample"}:
        raise ValueError(
            f"unsupported evaluation policy mode: {args.eval_policy_mode!r}"
        )

    agent = make_agent()
    logger = make_logger()
    logdir = elements.Path(args.logdir)
    logdir.mkdir()
    episodes = defaultdict(elements.Agg)
    returns = []
    team_returns = []
    agent_returns = []

    def log_transition(transition, worker):
        if len(returns) >= args.eval_eps:
            return
        episode = episodes[worker]
        transition["is_first"] and episode.reset()
        rewards = np.asarray(transition["reward"], np.float32)
        episode.add("score", np.float32(rewards.mean()), agg="sum")
        episode.add("team_return_sum", np.float32(rewards.sum()), agg="sum")
        episode.add("agent_scores", rewards, agg="sum")
        episode.add("length", 1, agg="sum")
        if transition["is_last"]:
            result = episode.result()
            score = float(result["score"])
            team_return = float(result["team_return_sum"])
            per_agent = np.asarray(result["agent_scores"], np.float32)
            returns.append(score)
            team_returns.append(team_return)
            agent_returns.append(per_agent)
            logger.add(
                {
                    "score": score,
                    "per_agent_return_mean": score,
                    "team_return_sum": team_return,
                    "length": result["length"],
                    "agent_return_min": per_agent.min(),
                    "agent_return_max": per_agent.max(),
                    "agent_return_std": per_agent.std(),
                },
                prefix="episode",
            )
            logger.write()

    checkpoint = elements.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(args.from_checkpoint, keys=["agent"])

    environments = min(args.envs, args.eval_eps)
    functions = [bind(make_env, index) for index in range(environments)]
    driver = None

    def policy(*values):
        return agent.policy(*values, mode=args.eval_policy_mode)

    try:
        driver = embodied.Driver(functions, parallel=not args.debug)
        driver.on_step(log_transition)
        driver.reset(agent.init_policy)
        while len(returns) < args.eval_eps:
            driver(policy, steps=10)
    finally:
        if driver is not None:
            driver.close()
        logger.close()

    returns_array = np.asarray(returns[: args.eval_eps], np.float32)
    team_returns_array = np.asarray(team_returns[: args.eval_eps], np.float32)
    agent_array = np.stack(agent_returns[: args.eval_eps])
    summary = {
        "episodes": int(args.eval_eps),
        "policy_mode": str(args.eval_policy_mode),
        "return_mean": float(returns_array.mean()),
        "return_median": float(np.median(returns_array)),
        "return_std": float(returns_array.std()),
        "return_min": float(returns_array.min()),
        "return_max": float(returns_array.max()),
        "return_p10": float(np.percentile(returns_array, 10)),
        "per_agent_return_mean": float(returns_array.mean()),
        "per_agent_return_median": float(np.median(returns_array)),
        "per_agent_return_std": float(returns_array.std()),
        "per_agent_return_p10": float(np.percentile(returns_array, 10)),
        "team_return_mean": float(team_returns_array.mean()),
        "team_return_median": float(np.median(team_returns_array)),
        "team_return_std": float(team_returns_array.std()),
        "team_return_p10": float(np.percentile(team_returns_array, 10)),
        "agent_return_mean": float(agent_array.mean()),
        "agent_return_std": float(agent_array.std()),
        "returns": returns_array.tolist(),
        "team_returns": team_returns_array.tolist(),
        "per_agent_returns": agent_array.tolist(),
    }
    (logdir / "evaluation_summary.json").write(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
