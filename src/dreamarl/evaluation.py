"""Fixed-checkpoint evaluation with explicit deterministic and sampled protocols."""

from __future__ import annotations

import json
from collections import defaultdict
from contextlib import contextmanager
from functools import partial as bind
from pathlib import Path

import elements
import embodied
import jax
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
        target_names = [
            name
            for name in names
            if name.startswith("attack_target_") and name.endswith("_count")
        ]
        attack_total = max(totals["action_attack_count"], np.finfo(np.float32).eps)
        for name in target_names:
            target = name.removeprefix("attack_target_").removesuffix("_count")
            summary[f"attack_target_{target}_fraction"] = float(
                sum(item[name] for item in outcomes) / attack_total
            )
    return summary


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
    if len(battle_wins) == episodes:
        summary["win_rate"] = float(np.mean(battle_wins))
        summary["wins"] = int(np.sum(battle_wins))
    summary.update(_summarize_outcomes(outcomes[:episodes]))
    return summary


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
    battle_wins = []
    outcomes = []
    jecc_controller = {
        "decisions": 0,
        "controllable": 0,
        "forced": 0,
        "changes": 0,
        "noops": 0,
    }
    environments = min(args.envs, args.eval_eps)
    quotas = np.full(environments, args.eval_eps // environments, np.int32)
    quotas[: args.eval_eps % environments] += 1
    completed = np.zeros(environments, np.int32)
    controller = (
        _FocalInterventionController(
            args.probe_model,
            episode_limit=int(args.probe_episode_limit),
        )
        if bool(args.probe_controller)
        else None
    )

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

    functions = [bind(make_env, index) for index in range(environments)]
    driver = None

    def policy(*values):
        carry, actions, outputs = agent.policy(*values, mode=args.eval_policy_mode)
        if controller is not None:
            observations = values[1]
            actions = controller(
                observations,
                actions,
                outputs,
                enabled=completed < quotas,
            )
            carry = _replace_carry_action(carry, actions)
        if "jecc_controller/focal" in outputs:
            focal = (
                np.asarray(outputs["jecc_controller/focal"], bool)
                & (completed < quotas)[:, None]
            )
            controllable = focal & np.asarray(
                outputs.get("jecc_controller/controllable", focal), bool
            )
            jecc_controller["decisions"] += int(focal.sum())
            jecc_controller["controllable"] += int(controllable.sum())
            jecc_controller["forced"] += int((focal & ~controllable).sum())
            jecc_controller["changes"] += int(
                np.asarray(outputs["jecc_controller/changed"], bool)[controllable].sum()
            )
            jecc_controller["noops"] += int(
                np.asarray(outputs["jecc_controller/noop"], bool)[controllable].sum()
            )
        return carry, actions, outputs

    try:
        driver = embodied.Driver(functions, parallel=not args.debug)
        driver.on_step(log_transition)
        driver.reset(agent.init_policy)
        while int(completed.sum()) < args.eval_eps:
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
    if len(battle_wins) == args.eval_eps:
        summary["win_rate"] = float(np.mean(battle_wins))
        summary["wins"] = int(np.sum(battle_wins))
    summary.update(_summarize_outcomes(outcomes[: args.eval_eps]))
    if controller is not None:
        summary.update(controller.metrics())
    if jecc_controller["decisions"]:
        decisions = jecc_controller["decisions"]
        controllable = jecc_controller["controllable"]
        summary.update(
            {
                "jecc_controller_decisions": decisions,
                "jecc_controller_controllable_decisions": controllable,
                "jecc_controller_forced_decisions": jecc_controller["forced"],
                "jecc_controller_forced_fraction": (
                    jecc_controller["forced"] / decisions
                ),
                "jecc_controller_action_changes": jecc_controller["changes"],
                "jecc_controller_change_fraction": (
                    jecc_controller["changes"] / max(controllable, 1)
                ),
                "jecc_controller_noop_fraction": (
                    jecc_controller["noops"] / max(controllable, 1)
                ),
            }
        )
    (logdir / "evaluation_summary.json").write(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _replace_carry_action(carry, actions):
    """Keep recurrent action history equal to the action sent to the env."""

    previous = dict(carry[-1])
    for key, value in actions.items():
        host = np.asarray(value)
        previous[key] = [
            jax.device_put(host[index].astype(item.dtype), item.sharding)
            for index, item in enumerate(previous[key])
        ]
    return (*carry[:-1], previous)


class _FocalInterventionController:
    """Privileged frozen critic that changes exactly one agent per battle."""

    def __init__(self, model_path, *, episode_limit):
        with np.load(str(model_path)) as model:
            metadata = json.loads(str(model["metadata"]))
            self.layers = []
            index = 0
            while f"critic_{index}_kernel" in model:
                self.layers.append(
                    (
                        np.asarray(model[f"critic_{index}_kernel"], np.float32),
                        np.asarray(model[f"critic_{index}_bias"], np.float32),
                    )
                )
                index += 1
        if not self.layers:
            raise ValueError(f"probe model contains no critic layers: {model_path}")
        self.table = np.asarray(metadata["latent_table"], np.float32)
        self.mean = np.asarray(metadata["critic_input_mean"], np.float32)
        self.std = np.asarray(metadata["critic_input_std"], np.float32)
        self.action_count = int(metadata["action_count"])
        self.episode_limit = int(episode_limit)
        self.steps = None
        self.changes = 0
        self.noops = 0
        self.decisions = 0

    def __call__(self, observations, actions, outputs, *, enabled=None):
        proposed = np.asarray(actions["action"], np.int32)
        masks = np.asarray(observations["action_mask"], bool)
        stochastic = np.asarray(outputs["dyn/stoch"])
        indices = stochastic.argmax(-1).astype(np.int64)
        batch, agents, variables = indices.shape
        if self.steps is None or len(self.steps) != batch:
            self.steps = np.zeros((batch,), np.int32)
        first = np.asarray(observations["is_first"], bool)
        self.steps[first] = 0
        variable = np.arange(variables)[None, None, :]
        embedded = self.table[variable, indices].reshape(batch, agents, -1)
        timestep = np.broadcast_to(
            (self.steps / max(self.episode_limit, 1))[:, None, None],
            (batch, agents, 1),
        ).astype(np.float32)
        states = np.concatenate(
            [embedded, np.asarray(observations["observation"], np.float32), timestep],
            axis=-1,
        )
        onehot = np.eye(self.action_count, dtype=np.float32)[proposed]
        changed = proposed.copy()
        if enabled is None:
            enabled = np.ones(batch, bool)
        for worker in range(batch):
            if not enabled[worker]:
                continue
            focal = worker % agents
            order = [focal, *[index for index in range(agents) if index != focal]]
            ordered_actions = onehot[worker, order].copy()
            ordered_actions[0] = 0.0
            inputs = np.concatenate(
                [states[worker, order].reshape(-1), ordered_actions.reshape(-1)]
            )
            values = self._predict(((inputs - self.mean) / self.std)[None])[0]
            values = np.where(masks[worker, focal], values, -np.inf)
            selection = int(np.argmax(values))
            self.changes += int(selection != proposed[worker, focal])
            self.noops += int(selection == 0)
            self.decisions += 1
            changed[worker, focal] = selection
        self.steps += 1
        return dict(actions, action=changed)

    def _predict(self, inputs):
        value = np.asarray(inputs, np.float32)
        for kernel, bias in self.layers[:-1]:
            value = value @ kernel + bias
            value = value / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))
        kernel, bias = self.layers[-1]
        return value @ kernel + bias

    def metrics(self):
        return {
            "controller_decisions": int(self.decisions),
            "controller_action_changes": int(self.changes),
            "controller_change_fraction": float(self.changes / max(self.decisions, 1)),
            "controller_noop_fraction": float(self.noops / max(self.decisions, 1)),
        }


def _probe_episode_samples(records, episode_id):
    """Convert one frozen rollout into aligned state-action/outcome samples."""

    if len(records) < 2 or not records[-1]["is_last"]:
        raise ValueError("probe episodes must include reset and terminal transitions")
    final = records[-1]
    arrays = {
        "observation": [],
        "latent_index": [],
        "action": [],
        "action_mask": [],
        "episode_id": [],
        "timestep": [],
        "steps_remaining": [],
        "damage_h5": [],
        "damage_h15": [],
        "corrected_h5": [],
        "corrected_h15": [],
        "enemy_death_h5": [],
        "enemy_death_h15": [],
        "ally_death_h15": [],
        "terminal_h15": [],
        "episode_enemy_deaths": [],
        "episode_ally_survivors": [],
        "episode_win": [],
        "episode_timeout": [],
    }
    for timestep, record in enumerate(records[:-1]):
        future5 = records[timestep + 1 : timestep + 6]
        future15 = records[timestep + 1 : timestep + 16]
        arrays["observation"].append(record["observation"])
        arrays["latent_index"].append(record["latent_index"])
        arrays["action"].append(record["action"])
        arrays["action_mask"].append(record["action_mask"])
        arrays["episode_id"].append(episode_id)
        arrays["timestep"].append(timestep)
        arrays["steps_remaining"].append(len(records) - timestep - 1)
        arrays["damage_h5"].append(sum(item["enemy_damage"] for item in future5))
        arrays["damage_h15"].append(sum(item["enemy_damage"] for item in future15))
        arrays["corrected_h5"].append(sum(item["corrected_reward"] for item in future5))
        arrays["corrected_h15"].append(
            sum(item["corrected_reward"] for item in future15)
        )
        arrays["enemy_death_h5"].append(
            float(any(item["enemy_deaths"] > 0 for item in future5))
        )
        arrays["enemy_death_h15"].append(
            float(any(item["enemy_deaths"] > 0 for item in future15))
        )
        arrays["ally_death_h15"].append(
            float(any(item["ally_deaths"] > 0 for item in future15))
        )
        arrays["terminal_h15"].append(float(any(item["is_last"] for item in future15)))
        arrays["episode_enemy_deaths"].append(final["dead_enemies"])
        arrays["episode_ally_survivors"].append(final["ally_survivors"])
        arrays["episode_win"].append(final["battle_won"])
        arrays["episode_timeout"].append(final["timeout"])
    return {key: np.asarray(value) for key, value in arrays.items()}


def collect_smac_probe(make_agent, make_env, args):
    """Collect immutable frozen-policy trajectories for outcome-credit probes."""

    if not args.from_checkpoint:
        raise ValueError("SMAC probe collection requires run.from_checkpoint")
    if not args.probe_output:
        raise ValueError("SMAC probe collection requires run.probe_output")
    if args.eval_eps < 1:
        raise ValueError("SMAC probe collection requires at least one episode")
    if args.eval_policy_mode not in {"eval", "eval_sample"}:
        raise ValueError(f"unsupported probe policy mode: {args.eval_policy_mode!r}")

    agent = make_agent()
    checkpoint = elements.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(args.from_checkpoint, keys=["agent"])
    completed = []
    active = defaultdict(list)

    def record_transition(transition, worker):
        if len(completed) >= args.eval_eps:
            return
        if transition["is_first"]:
            active[worker] = []
        if "dyn/stoch" not in transition:
            raise KeyError("probe collection requires dyn/stoch policy outputs")
        active[worker].append(
            {
                "observation": np.asarray(transition["observation"], np.float32),
                "latent_index": np.asarray(transition["dyn/stoch"])
                .argmax(-1)
                .astype(np.uint8),
                "action": np.asarray(transition["action"], np.int16),
                "action_mask": np.asarray(transition["action_mask"], bool),
                "enemy_damage": float(transition.get("log/enemy_damage", 0.0)),
                "corrected_reward": float(transition.get("log/corrected_reward", 0.0)),
                "enemy_deaths": float(transition.get("log/enemy_deaths_step", 0.0)),
                "ally_deaths": float(transition.get("log/ally_deaths_step", 0.0)),
                "dead_enemies": float(transition.get("log/dead_enemies", 0.0)),
                "ally_survivors": float(transition.get("log/ally_survivors", 0.0)),
                "battle_won": float(transition.get("log/battle_won", 0.0)),
                "timeout": float(transition.get("log/timeout", 0.0)),
                "is_last": bool(transition["is_last"]),
            }
        )
        if transition["is_last"] and len(completed) < args.eval_eps:
            completed.append(active.pop(worker))

    environments = min(args.envs, args.eval_eps)
    functions = [bind(make_env, index) for index in range(environments)]
    driver = embodied.Driver(functions, parallel=not args.debug)

    def policy(*values):
        return agent.policy(*values, mode=args.eval_policy_mode)

    try:
        driver.on_step(record_transition)
        driver.reset(agent.init_policy)
        while len(completed) < args.eval_eps:
            driver(policy, steps=10)
    finally:
        driver.close()

    samples = [
        _probe_episode_samples(records, episode_id)
        for episode_id, records in enumerate(completed[: args.eval_eps])
    ]
    dataset = {
        key: np.concatenate([sample[key] for sample in samples], axis=0)
        for key in samples[0]
    }
    dataset["num_episodes"] = np.asarray(args.eval_eps, np.int32)
    dataset["checkpoint"] = np.asarray(str(args.from_checkpoint))
    dataset["policy_mode"] = np.asarray(str(args.eval_policy_mode))
    output = Path(str(args.probe_output)).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **dataset)
    result = {
        "output": str(output),
        "episodes": int(args.eval_eps),
        "samples": int(dataset["action"].shape[0]),
        "wins": int(sum(records[-1]["battle_won"] for records in completed)),
        "mean_enemy_deaths": float(
            np.mean([records[-1]["dead_enemies"] for records in completed])
        ),
        "mean_ally_survivors": float(
            np.mean([records[-1]["ally_survivors"] for records in completed])
        ),
    }
    Path(str(output) + ".json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
