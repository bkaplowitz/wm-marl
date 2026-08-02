"""Fixed deterministic evaluation for first-party DreaMARL agents."""

from __future__ import annotations

import json
from collections import defaultdict
from functools import partial as bind

import elements
import embodied
import numpy as np


def eval_only(make_agent, make_env, make_logger, args):
    if not args.from_checkpoint:
        raise ValueError("evaluation requires run.from_checkpoint")
    if args.eval_eps < 1:
        raise ValueError("evaluation requires at least one episode")

    agent = make_agent()
    logger = make_logger()
    logdir = elements.Path(args.logdir)
    logdir.mkdir()
    episodes = defaultdict(elements.Agg)
    returns = []
    agent_returns = []

    def log_transition(transition, worker):
        if len(returns) >= args.eval_eps:
            return
        episode = episodes[worker]
        transition["is_first"] and episode.reset()
        rewards = np.asarray(transition["reward"], np.float32)
        episode.add("score", np.float32(rewards.mean()), agg="sum")
        episode.add("agent_scores", rewards, agg="sum")
        episode.add("length", 1, agg="sum")
        if transition["is_last"]:
            result = episode.result()
            score = float(result["score"])
            per_agent = np.asarray(result["agent_scores"], np.float32)
            returns.append(score)
            agent_returns.append(per_agent)
            logger.add(
                {
                    "score": score,
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
    driver = embodied.Driver(functions, parallel=not args.debug)
    driver.on_step(log_transition)

    def policy(*values):
        return agent.policy(*values, mode="eval")

    driver.reset(agent.init_policy)
    while len(returns) < args.eval_eps:
        driver(policy, steps=10)
    logger.close()

    returns_array = np.asarray(returns[: args.eval_eps], np.float32)
    agent_array = np.stack(agent_returns[: args.eval_eps])
    summary = {
        "episodes": int(args.eval_eps),
        "return_mean": float(returns_array.mean()),
        "return_median": float(np.median(returns_array)),
        "return_std": float(returns_array.std()),
        "return_min": float(returns_array.min()),
        "return_max": float(returns_array.max()),
        "return_p10": float(np.percentile(returns_array, 10)),
        "agent_return_mean": float(agent_array.mean()),
        "agent_return_std": float(agent_array.std()),
        "returns": returns_array.tolist(),
        "per_agent_returns": agent_array.tolist(),
    }
    (logdir / "evaluation_summary.json").write(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
