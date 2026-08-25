"""First-party Embodied training loop with a guaranteed final checkpoint."""

import collections
import json
import time
from functools import partial as bind

import elements
import embodied
import jax
import numpy as np

from .evaluation import evaluate_current_policy


_RAW_EVALUATION_KEYS = {
    "returns",
    "team_returns",
    "per_agent_returns",
    "battle_wins",
    "outcomes",
    "episode_metadata",
}


def _write_evaluation_episodes(logdir, step, summary):
    """Append lossless per-episode evaluation data for publication plots."""

    returns = summary.get("returns", ())
    team_returns = summary.get("team_returns", ())
    per_agent_returns = summary.get("per_agent_returns", ())
    battle_wins = summary.get("battle_wins", ())
    outcomes = summary.get("outcomes", ())
    metadata = summary.get("episode_metadata", ())
    count = int(summary["episodes"])
    if not (len(returns) == len(team_returns) == len(per_agent_returns) == count):
        raise ValueError("evaluation episode arrays do not match the episode quota")

    path = logdir / "evaluation_episodes.jsonl"
    with path.open("a") as stream:
        for index in range(count):
            record = {
                "schema_version": 1,
                "environment_steps": int(step),
                "episode": index,
                "return": float(returns[index]),
                "team_return": float(team_returns[index]),
                "per_agent_returns": per_agent_returns[index],
                "battle_won": (
                    float(battle_wins[index]) if index < len(battle_wins) else None
                ),
                "outcome": outcomes[index] if index < len(outcomes) else {},
                "metadata": metadata[index] if index < len(metadata) else {},
            }
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def _save_checkpoint(checkpoint, attempts=4):
    """Retry transient filesystem failures without hiding a persistent error."""

    for attempt in range(attempts):
        try:
            checkpoint.save()
            return
        except OSError:
            if attempt + 1 == attempts:
                raise
            time.sleep(5 * (attempt + 1))


def _with_policy_reference(primary, reference):
    """Attach an independently sampled replay batch for actor trust."""

    reference = iter(reference)
    for batch in primary:
        other = next(reference)
        yield {
            **batch,
            **{f"_policy_reference/{key}": value for key, value in other.items()},
        }


def train(make_agent, make_replay, make_env, make_stream, make_logger, args):
    agent = make_agent()
    replay = make_replay()
    logger = make_logger()

    logdir = elements.Path(args.logdir)
    step = logger.step
    usage = elements.Usage(**args.usage)
    train_agg = elements.Agg()
    epstats = elements.Agg()
    episodes = collections.defaultdict(elements.Agg)
    policy_fps = elements.FPS()
    train_fps = elements.FPS()

    batch_steps = args.batch_size * args.batch_length
    should_train = elements.when.Ratio(args.train_ratio / batch_steps)
    should_log = embodied.LocalClock(args.log_every)
    should_report = embodied.LocalClock(args.report_every)
    should_save = embodied.LocalClock(args.save_every)
    next_curve_eval = (
        int(args.curve_eval_interval) if args.curve_eval_interval else None
    )

    @elements.timer.section("logfn")
    def logfn(tran, worker):
        episode = episodes[worker]
        tran["is_first"] and episode.reset()
        rewards = np.asarray(tran["reward"], np.float32)
        mean_reward = np.float32(rewards.mean())
        team_reward = np.float32(rewards.sum())
        episode.add("score", mean_reward, agg="sum")
        episode.add("per_agent_return_mean", mean_reward, agg="sum")
        episode.add("team_return_sum", team_reward, agg="sum")
        episode.add("agent_scores", rewards, agg="sum")
        episode.add("length", 1, agg="sum")
        episode.add("rewards", mean_reward, agg="stack")
        for key, value in tran.items():
            if value.dtype == np.uint8 and value.ndim == 3:
                if worker == 0:
                    episode.add(f"policy_{key}", value, agg="stack")
            elif key.startswith("log/"):
                assert value.ndim == 0, (key, value.shape, value.dtype)
                episode.add(key + "/avg", value, agg="avg")
                episode.add(key + "/max", value, agg="max")
                episode.add(key + "/sum", value, agg="sum")
        if tran["is_last"]:
            result = episode.result()
            agent_scores = np.asarray(result.pop("agent_scores"), np.float32)
            logger.add(
                {
                    "score": result.pop("score"),
                    "per_agent_return_mean": result.pop("per_agent_return_mean"),
                    "team_return_sum": result.pop("team_return_sum"),
                    "length": result.pop("length"),
                    "agent_return_min": agent_scores.min(),
                    "agent_return_max": agent_scores.max(),
                    "agent_return_std": agent_scores.std(),
                },
                prefix="episode",
            )
            rewards = result.pop("rewards")
            if len(rewards) > 1:
                result["reward_rate"] = (
                    np.abs(rewards[1:] - rewards[:-1]) >= 0.01
                ).mean()
            epstats.add(result)

    functions = [bind(make_env, index) for index in range(args.envs)]
    driver = embodied.Driver(functions, parallel=not args.debug)
    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(lambda tran, _: policy_fps.step())
    driver.on_step(replay.add)
    driver.on_step(logfn)

    policy_reference = any(key.startswith("_policy_reference/") for key in agent.spaces)
    train_source = make_stream(replay, "train")
    report_source = make_stream(replay, "report")
    if policy_reference:
        train_source = _with_policy_reference(
            train_source,
            make_stream(replay, "train"),
        )
        report_source = _with_policy_reference(
            report_source,
            make_stream(replay, "report"),
        )
    stream_train = iter(agent.stream(train_source))
    stream_report = iter(agent.stream(report_source))
    carry_train = [agent.init_train(args.batch_size)]
    carry_report = agent.init_report(args.batch_size)

    def trainfn(tran, worker):
        del tran, worker
        if len(replay) < args.batch_size * args.batch_length:
            return
        for _ in range(should_train(step)):
            with elements.timer.section("stream_next"):
                batch = next(stream_train)
            if "_environment_step" in agent.spaces:
                reference = batch["is_first"]
                batch["_environment_step"] = jax.device_put(
                    np.full(reference.shape, int(step), np.int32),
                    reference.sharding,
                )
            carry_train[0], outputs, metrics = agent.train(carry_train[0], batch)
            train_fps.step(batch_steps)
            if "replay" in outputs:
                replay.update(outputs["replay"])
            train_agg.add(metrics, prefix="train")

    driver.on_step(trainfn)

    checkpoint = elements.Checkpoint(
        logdir / "ckpt",
        keep=None if bool(args.checkpoint_at_curve_eval) else 1,
    )
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.replay = replay
    if checkpoint.exists():
        checkpoint.load()

    print("Start training loop")

    def policy(*values):
        return agent.policy(*values, mode="train")

    driver.reset(agent.init_policy)
    while step < args.steps:
        driver(policy, steps=10)

        if should_report(step) and len(replay):
            aggregate = elements.Agg()
            for _ in range(args.consec_report * args.report_batches):
                batch = next(stream_report)
                if "_environment_step" in agent.spaces:
                    reference = batch["is_first"]
                    batch["_environment_step"] = jax.device_put(
                        np.full(reference.shape, int(step), np.int32),
                        reference.sharding,
                    )
                carry_report, metrics = agent.report(carry_report, batch)
                aggregate.add(metrics)
            logger.add(aggregate.result(), prefix="report")

        if should_log(step):
            logger.add(
                {
                    "environment_steps": int(step),
                    "agent_steps": int(step) * int(args.num_agents),
                },
                prefix="counters",
            )
            logger.add(train_agg.result())
            logger.add(epstats.result(), prefix="epstats")
            logger.add(replay.stats(), prefix="replay")
            logger.add(usage.stats(), prefix="usage")
            logger.add({"fps/policy": policy_fps.result()})
            logger.add({"fps/train": train_fps.result()})
            logger.add({"timer": elements.timer.stats()["summary"]})
            logger.write()

        if should_save(step):
            _save_checkpoint(checkpoint)

        if next_curve_eval is not None and int(step) >= next_curve_eval:
            summary = evaluate_current_policy(
                agent,
                make_env,
                episodes=int(args.curve_eval_eps),
                envs=int(args.eval_envs),
                debug=bool(args.debug),
                worker_offset=int(args.curve_eval_seed_offset),
                policy_mode=str(args.curve_eval_policy_mode),
            )
            _write_evaluation_episodes(logdir, step, summary)
            logger.add(
                {
                    key: value
                    for key, value in summary.items()
                    if key not in _RAW_EVALUATION_KEYS
                },
                prefix="eval",
            )
            logger.write()
            if bool(args.checkpoint_at_curve_eval):
                _save_checkpoint(checkpoint)
            while next_curve_eval <= int(step):
                next_curve_eval += int(args.curve_eval_interval)

    if bool(args.final_save):
        _save_checkpoint(checkpoint)
    logger.close()
