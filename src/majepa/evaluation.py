"""Fixed-policy evaluation for canonical MA-JEPA runs."""

from __future__ import annotations

import json
from collections import defaultdict
from contextlib import contextmanager
from functools import partial as bind

import elements
import embodied
import numpy as np


_SMAC_SUM_DIAGNOSTICS = {
    "log/legacy_reward": "legacy_return",
    "log/corrected_reward": "corrected_return",
    "log/enemy_damage": "enemy_damage",
    "log/enemy_health_damage": "enemy_health_damage",
    "log/enemy_shield_damage": "enemy_shield_damage",
    "log/enemy_shield_regen": "enemy_shield_regen",
    "log/enemy_deaths_step": "enemy_deaths",
    "log/ally_deaths_step": "ally_deaths",
    "log/action_noop_count": "action_noop_count",
    "log/action_stop_count": "action_stop_count",
    "log/action_move_count": "action_move_count",
    "log/action_attack_count": "action_attack_count",
    "log/action_target_switch_count": "action_target_switch_count",
}

_SMAC_FINAL_DIAGNOSTICS = {
    "log/timeout": "timeout",
    "log/battle_won": "battle_won",
    "log/dead_allies": "dead_allies",
    "log/dead_enemies": "dead_enemies",
    "log/ally_survivors": "ally_survivors",
    "log/enemy_survivors": "enemy_survivors",
}

_RAW_EVALUATION_KEYS = {
    "returns",
    "team_returns",
    "per_agent_returns",
}


def _add_outcome_diagnostics(episode, transition):
    for source, target in _SMAC_SUM_DIAGNOSTICS.items():
        if source in transition:
            episode.add(target, np.float32(transition[source]), agg="sum")
    for source in sorted(transition):
        if source.startswith("log/attack_target_") and source.endswith("_count"):
            episode.add(source.removeprefix("log/"), transition[source], agg="sum")


def _episode_outcome(result, transition):
    outcome = {
        target: float(result[target])
        for target in _SMAC_SUM_DIAGNOSTICS.values()
        if target in result
    }
    outcome.update(
        {
            key: float(value)
            for key, value in result.items()
            if key.startswith("attack_target_") and key.endswith("_count")
        }
    )
    outcome.update(
        {
            target: float(transition[source])
            for source, target in _SMAC_FINAL_DIAGNOSTICS.items()
            if source in transition
        }
    )
    return outcome


def _summarize_outcomes(outcomes):
    if not outcomes:
        return {}
    names = sorted(set.intersection(*(set(item) for item in outcomes)))
    summary = {}
    for name in names:
        values = np.asarray([item[name] for item in outcomes], np.float32)
        summary[f"{name}_mean"] = float(values.mean())
        summary[f"{name}_std"] = float(values.std())
    if "timeout" in names:
        summary["timeout_rate"] = summary.pop("timeout_mean")
        summary.pop("timeout_std")
    if "legacy_return" in names and "corrected_return" in names:
        gaps = np.asarray(
            [item["legacy_return"] - item["corrected_return"] for item in outcomes],
            np.float32,
        )
        summary["legacy_corrected_gap_mean"] = float(gaps.mean())
        summary["legacy_corrected_gap_std"] = float(gaps.std())
    action_names = (
        "action_noop_count",
        "action_stop_count",
        "action_move_count",
        "action_attack_count",
    )
    if all(name in names for name in action_names):
        totals = {name: sum(item[name] for item in outcomes) for name in action_names}
        denominator = max(sum(totals.values()), np.finfo(np.float32).eps)
        for name, value in totals.items():
            label = name.removeprefix("action_").removesuffix("_count")
            summary[f"action_{label}_fraction"] = float(value / denominator)
        attack_total = max(totals["action_attack_count"], np.finfo(np.float32).eps)
        targets = [
            name
            for name in names
            if name.startswith("attack_target_") and name.endswith("_count")
        ]
        for name in targets:
            target = name.removeprefix("attack_target_").removesuffix("_count")
            summary[f"attack_target_{target}_fraction"] = float(
                sum(item[name] for item in outcomes) / attack_total
            )
    return summary


def _evaluation_summary(
    returns,
    team_returns,
    agent_returns,
    battle_wins,
    outcomes,
    episodes,
    *,
    policy_mode=None,
):
    returns_array = np.asarray(returns[:episodes], np.float32)
    team_returns_array = np.asarray(team_returns[:episodes], np.float32)
    agent_array = np.stack(agent_returns[:episodes])
    summary = {
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
    if policy_mode is not None:
        summary["policy_mode"] = str(policy_mode)
    if len(battle_wins) == episodes:
        summary["win_rate"] = float(np.mean(battle_wins))
        summary["wins"] = int(np.sum(battle_wins))
    summary.update(_summarize_outcomes(outcomes[:episodes]))
    return summary


def _validate_standalone_evaluation(summary, records, expected_episodes):
    """Reject incomplete or internally inconsistent held-out evaluations."""

    expected_episodes = int(expected_episodes)
    if int(summary.get("episodes", -1)) != expected_episodes:
        raise ValueError("evaluation summary does not match the episode quota")
    for key in sorted(_RAW_EVALUATION_KEYS):
        values = summary.get(key, ())
        if len(values) != expected_episodes:
            raise ValueError(f"evaluation summary field {key!r} is incomplete")
    if len(records) != expected_episodes:
        raise ValueError("raw evaluation records do not match the episode quota")
    if [record["episode"] for record in records] != list(range(expected_episodes)):
        raise ValueError("raw evaluation episode indices are not contiguous")


def _write_evaluation_records(logdir, records):
    """Write one lossless JSON record per completed evaluation episode."""

    path = logdir / "evaluation_episodes.jsonl"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evaluation records: {path}")
    with path.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    return path


@contextmanager
def _preserve_policy_state(agent):
    """Keep inline evaluation from consuming RNG or pending parameter sync."""

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
                        "policy synchronization changed during inline evaluation"
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

    if episodes < 1 or envs < 1:
        raise ValueError("evaluation requires positive episode and environment counts")
    if policy_mode not in {"eval", "eval_sample"}:
        raise ValueError(f"unsupported evaluation policy mode: {policy_mode!r}")
    returns = []
    team_returns = []
    agent_returns = []
    battle_wins = []
    outcomes = []
    accumulators = defaultdict(elements.Agg)
    environments = min(envs, episodes)
    quotas = np.full(environments, episodes // environments, np.int32)
    quotas[: episodes % environments] += 1
    completed = np.zeros(environments, np.int32)

    def log_transition(transition, worker):
        if completed[worker] >= quotas[worker]:
            return
        episode = accumulators[worker]
        transition["is_first"] and episode.reset()
        rewards = np.asarray(transition["reward"], np.float32)
        episode.add("score", np.float32(rewards.mean()), agg="sum")
        episode.add("team_return_sum", np.float32(rewards.sum()), agg="sum")
        episode.add("agent_scores", rewards, agg="sum")
        _add_outcome_diagnostics(episode, transition)
        if transition["is_last"]:
            completed[worker] += 1
            result = episode.result()
            returns.append(float(result["score"]))
            team_returns.append(float(result["team_return_sum"]))
            agent_returns.append(np.asarray(result["agent_scores"], np.float32))
            if "log/battle_won" in transition:
                battle_wins.append(float(transition["log/battle_won"]))
            outcome = _episode_outcome(result, transition)
            if outcome:
                outcomes.append(outcome)

    functions = [bind(make_env, worker_offset + index) for index in range(environments)]
    driver = embodied.Driver(functions, parallel=not debug)
    driver.on_step(log_transition)

    def policy(*values):
        return agent.policy(*values, mode=policy_mode)

    try:
        with _preserve_policy_state(agent):
            driver.reset(agent.init_policy)
            while int(completed.sum()) < episodes:
                driver(policy, steps=10)
    finally:
        driver.close()
    return _evaluation_summary(
        returns, team_returns, agent_returns, battle_wins, outcomes, episodes
    )


def eval_only(make_agent, make_env, make_logger, args):
    """Evaluate one explicit checkpoint and write its complete summary."""

    if not args.from_checkpoint:
        raise ValueError("evaluation requires run.from_checkpoint")
    if args.eval_eps < 1 or args.envs < 1:
        raise ValueError("evaluation requires positive episode and environment counts")
    if args.eval_policy_mode not in {"eval", "eval_sample"}:
        raise ValueError(
            f"unsupported evaluation policy mode: {args.eval_policy_mode!r}"
        )
    worker_offset = int(args.eval_worker_offset)
    if worker_offset < 0:
        raise ValueError("evaluation worker offset must be nonnegative")

    agent = make_agent()
    logger = make_logger()
    logdir = elements.Path(args.logdir)
    logdir.mkdir()
    episodes = defaultdict(elements.Agg)
    returns = []
    team_returns = []
    agent_returns = []
    battle_wins = []
    outcomes = []
    records = []
    environments = min(args.envs, args.eval_eps)
    quotas = np.full(environments, args.eval_eps // environments, np.int32)
    quotas[: args.eval_eps % environments] += 1
    completed = np.zeros(environments, np.int32)

    def log_transition(transition, worker):
        if completed[worker] >= quotas[worker]:
            return
        episode = episodes[worker]
        transition["is_first"] and episode.reset()
        rewards = np.asarray(transition["reward"], np.float32)
        episode.add("score", np.float32(rewards.mean()), agg="sum")
        episode.add("team_return_sum", np.float32(rewards.sum()), agg="sum")
        episode.add("agent_scores", rewards, agg="sum")
        episode.add("length", 1, agg="sum")
        _add_outcome_diagnostics(episode, transition)
        if transition["is_last"]:
            worker_episode = int(completed[worker])
            completed[worker] += 1
            result = episode.result()
            score = float(result["score"])
            team_return = float(result["team_return_sum"])
            per_agent = np.asarray(result["agent_scores"], np.float32)
            returns.append(score)
            team_returns.append(team_return)
            agent_returns.append(per_agent)
            if "log/battle_won" in transition:
                battle_wins.append(float(transition["log/battle_won"]))
            outcome = _episode_outcome(result, transition)
            if outcome:
                outcomes.append(outcome)
            records.append(
                {
                    "schema_version": 1,
                    "episode": len(records),
                    "return": score,
                    "team_return": team_return,
                    "per_agent_returns": per_agent.tolist(),
                    "battle_won": (
                        float(transition["log/battle_won"])
                        if "log/battle_won" in transition
                        else None
                    ),
                    "outcome": outcome,
                    "metadata": {
                        "worker": int(worker),
                        "worker_index": worker_offset + int(worker),
                        "worker_episode": worker_episode,
                    },
                }
            )
            logger.add(
                {
                    "score": score,
                    "per_agent_return_mean": score,
                    "team_return_sum": team_return,
                    "length": result["length"],
                    "agent_return_min": per_agent.min(),
                    "agent_return_max": per_agent.max(),
                    "agent_return_std": per_agent.std(),
                    **outcome,
                },
                prefix="episode",
            )
            logger.write()

    checkpoint = elements.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(args.from_checkpoint, keys=["agent"])
    functions = [bind(make_env, worker_offset + index) for index in range(environments)]
    driver = None
    summary = None

    def policy(*values):
        return agent.policy(*values, mode=args.eval_policy_mode)

    try:
        driver = embodied.Driver(functions, parallel=not args.debug)
        driver.on_step(log_transition)
        driver.reset(agent.init_policy)
        while int(completed.sum()) < args.eval_eps:
            driver(policy, steps=10)
        driver.close()
        driver = None

        summary = _evaluation_summary(
            returns,
            team_returns,
            agent_returns,
            battle_wins,
            outcomes,
            args.eval_eps,
            policy_mode=args.eval_policy_mode,
        )
        summary["evaluation_protocol"] = {
            "episodes": int(args.eval_eps),
            "envs": int(environments),
            "worker_offset": worker_offset,
            "worker_indices": [worker_offset + index for index in range(environments)],
            "policy_mode": str(args.eval_policy_mode),
        }
        _validate_standalone_evaluation(summary, records, args.eval_eps)
        _write_evaluation_records(logdir, records)
        (logdir / "evaluation_summary.json").write(
            json.dumps(summary, indent=2, sort_keys=True)
        )
        logger.add(
            {
                key: value
                for key, value in summary.items()
                if key not in _RAW_EVALUATION_KEYS
                and isinstance(value, (int, float, np.number))
            },
            prefix="final_eval",
        )
        logger.write()
    finally:
        if driver is not None:
            driver.close()
        logger.close()
    assert summary is not None
    print(json.dumps(summary, indent=2, sort_keys=True))
