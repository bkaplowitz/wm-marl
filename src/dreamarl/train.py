"""First-party Embodied training loop with a guaranteed final checkpoint."""

import collections
import time
from functools import partial as bind

import elements
import embodied
import jax
import numpy as np

from .evaluation import evaluate_current_policy


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


def _with_prefixed_batch(primary, secondary, prefix):
    """Pair independent batches without merging their batch or time axes."""

    if not prefix or not prefix.endswith("/"):
        raise ValueError(f"reserved replay prefix must end in '/': {prefix!r}")
    secondary = iter(secondary)
    for batch in primary:
        other = next(secondary)
        if batch["is_first"].shape != other["is_first"].shape:
            raise ValueError(
                "paired replay views must have identical BxT shapes, got "
                f"{batch['is_first'].shape} and {other['is_first'].shape}"
            )
        attached = {f"{prefix}{key}": value for key, value in other.items()}
        overlap = batch.keys() & attached.keys()
        if overlap:
            raise ValueError(f"reserved replay prefix collision: {sorted(overlap)}")
        yield {
            **batch,
            **attached,
        }


def _with_policy_reference(primary, reference):
    """Attach an independently sampled replay batch for actor trust."""

    return _with_prefixed_batch(primary, reference, "_policy_reference/")


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
    minimum_replay_size = batch_steps
    actor_critic_start_step = int(args.actor_critic_start_step)
    if actor_critic_start_step < 0:
        raise ValueError("actor_critic_start_step must be nonnegative")
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
    behavior_replay = any(key.startswith("_behavior_replay/") for key in agent.spaces)
    dual_view = bool(getattr(replay, "dual_view", False))
    if behavior_replay != dual_view:
        raise ValueError(
            "dual-view replay and learner behavior-batch spaces must be enabled "
            "together"
        )

    train_source = make_stream(replay, "train_world" if dual_view else "train")
    report_source = make_stream(replay, "report")
    if dual_view:
        train_source = _with_prefixed_batch(
            train_source,
            make_stream(replay, "train_behavior"),
            "_behavior_replay/",
        )
        # JAX requires every advertised input space for report compilation as
        # well. TeamAxisAdapter.report strips this transport-only copy, so the
        # primary report batch and metrics retain their prior semantics.
        report_source = _with_prefixed_batch(
            report_source,
            make_stream(replay, "report"),
            "_behavior_replay/",
        )
    if policy_reference:
        train_source = _with_policy_reference(
            train_source,
            make_stream(replay, "train_behavior" if dual_view else "train"),
        )
        report_source = _with_policy_reference(
            report_source,
            make_stream(replay, "report"),
        )
    stream_train = iter(agent.stream(train_source))
    stream_report = iter(agent.stream(report_source))
    carry_train = [agent.init_train(args.batch_size)]
    carry_report = agent.init_report(args.batch_size)
    learner_update_calls = elements.Counter()
    world_only_update_calls = elements.Counter()
    actor_critic_update_calls = elements.Counter()
    first_learner_environment_step = elements.Counter(-1)
    first_actor_critic_environment_step = elements.Counter(-1)

    def trainfn(tran, worker):
        del tran, worker
        current_step = int(step)
        # Do not call Ratio before replay eligibility. This prevents prefill
        # collection from creating a learner-update backlog.
        if len(replay) < minimum_replay_size:
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
            learner_update_calls.increment()
            if int(first_learner_environment_step) < 0:
                first_learner_environment_step.increment(current_step + 1)
            if current_step < actor_critic_start_step:
                world_only_update_calls.increment()
            else:
                actor_critic_update_calls.increment()
                if int(first_actor_critic_environment_step) < 0:
                    first_actor_critic_environment_step.increment(current_step + 1)
            train_fps.step(batch_steps * (2 if dual_view else 1))
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
    checkpoint.learner_update_calls = learner_update_calls
    checkpoint.world_only_update_calls = world_only_update_calls
    checkpoint.actor_critic_update_calls = actor_critic_update_calls
    checkpoint.first_learner_environment_step = first_learner_environment_step
    checkpoint.first_actor_critic_environment_step = first_actor_critic_environment_step
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
                    "learner_update_calls": int(learner_update_calls),
                    "world_only_update_calls": int(world_only_update_calls),
                    "actor_critic_update_calls": int(actor_critic_update_calls),
                },
                prefix="counters",
            )
            logger.add(
                {
                    "actor_critic_start_step": actor_critic_start_step,
                    "minimum_replay_eligible_starts": minimum_replay_size,
                    "first_learner_environment_step": int(
                        first_learner_environment_step
                    ),
                    "first_actor_critic_environment_step": int(
                        first_actor_critic_environment_step
                    ),
                    "world_model_active": float(len(replay) >= minimum_replay_size),
                    "actor_critic_active": float(
                        int(step) >= actor_critic_start_step
                        and len(replay) >= minimum_replay_size
                    ),
                    "configured_train_ratio": float(args.train_ratio),
                    "optimizer_calls_per_environment_step": float(
                        args.train_ratio / batch_steps
                    ),
                },
                prefix="schedule",
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
            logger.add(
                {
                    key: value
                    for key, value in summary.items()
                    if key not in {"returns", "team_returns", "per_agent_returns"}
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
