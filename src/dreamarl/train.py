"""First-party Embodied training loop with a guaranteed final checkpoint."""

import collections
from functools import partial as bind

import elements
import embodied
import jax
import numpy as np

from .evaluation import evaluate_current_policy


def train(make_agent, make_replay, make_env, make_stream, make_logger, args):
    agent = make_agent()
    replay = make_replay()
    if bool(args.load_replay):
        replay.load()
        if not len(replay):
            raise RuntimeError("continuation replay was requested but no items loaded")
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

    stream_train = iter(agent.stream(make_stream(replay, "train")))
    stream_report = iter(agent.stream(make_stream(replay, "report")))
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

    checkpoint = elements.Checkpoint(logdir / "ckpt")
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.replay = replay
    if args.from_checkpoint:
        elements.checkpoint.load(
            args.from_checkpoint,
            {"agent": bind(agent.load, regex=args.from_checkpoint_regex)},
        )
    checkpoint.load_or_save()

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
            checkpoint.save()

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
            logger.add(
                {
                    key: value
                    for key, value in summary.items()
                    if key not in {"returns", "team_returns", "per_agent_returns"}
                },
                prefix="eval",
            )
            logger.write()
            while next_curve_eval <= int(step):
                next_curve_eval += int(args.curve_eval_interval)

    if bool(args.final_save):
        checkpoint.save()
    logger.close()
