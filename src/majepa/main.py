import os
from functools import partial as bind

import elements
import embodied
import portal

from .configuration import load_config


def _validate_script(script: str, num_agents: int) -> None:
    if num_agents < 2:
        raise ValueError("MA-JEPA requires at least two agents")
    if script not in {"train", "eval_only"}:
        raise ValueError("MA-JEPA supports only train and eval_only")


def main(argv=None):
    from .marl.agent import MARLCore

    [elements.print(line) for line in MARLCore.banner]

    parsed, other = elements.Flags(configs=["baseline"], config="").parse_known(argv)
    config = load_config(parsed.configs, parsed.config or None)
    config = elements.Flags(config).parse(other)
    config = config.update(
        logdir=(config.logdir.format(timestamp=elements.timestamp()))
    )

    if "JOB_COMPLETION_INDEX" in os.environ:
        config = config.update(replica=int(os.environ["JOB_COMPLETION_INDEX"]))
    print("Replica:", config.replica, "/", config.replicas)

    logdir = elements.Path(config.logdir)
    print("Logdir:", logdir)
    print("Run script:", config.script)
    _validate_script(str(config.script), int(config.agent.num_agents))
    logdir.mkdir()
    config.save(logdir / "config.yaml")

    def init():
        elements.timer.global_timer.enabled = config.logger.timer

    portal.setup(
        errfile=config.errfile and logdir / "error",
        clientkw=dict(logging_color="cyan"),
        serverkw=dict(logging_color="cyan"),
        initfns=[init],
        ipv6=config.ipv6,
    )

    args = elements.Config(
        **config.run,
        replica=config.replica,
        replicas=config.replicas,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        report_length=config.report_length,
        consec_train=config.consec_train,
        consec_report=config.consec_report,
        replay_context=config.replay_context,
        num_agents=config.agent.num_agents,
    )

    if config.script == "train":
        from . import train as first_party_train

        first_party_train.train(
            bind(make_agent, config),
            bind(make_replay, config, "replay"),
            bind(make_env, config),
            bind(make_stream, config),
            bind(make_logger, config),
            args,
        )

    elif config.script == "eval_only":
        from . import evaluation

        evaluation.eval_only(
            bind(make_agent, config),
            bind(make_env, config),
            bind(make_logger, config),
            args,
        )

    else:
        raise NotImplementedError(config.script)


def make_agent(config):
    from .marl.agent import MARLCore as Algorithm

    env = make_env(config, 0)
    if env.num_agents != config.agent.num_agents:
        raise ValueError(
            f"Environment exposes {env.num_agents} agents but the resolved "
            f"configuration declares {config.agent.num_agents}."
        )
    obs_space = {
        key: value for key, value in env.obs_space.items() if not key.startswith("log/")
    }
    act_space = {k: v for k, v in env.act_space.items() if k != "reset"}
    env.close()
    if config.random_agent:
        return embodied.RandomAgent(obs_space, act_space)
    cpdir = elements.Path(config.logdir)
    cpdir = cpdir.parent if config.replicas > 1 else cpdir
    agent = Algorithm(
        obs_space,
        act_space,
        elements.Config(
            **config.agent,
            logdir=config.logdir,
            seed=config.seed,
            jax=config.jax,
            batch_size=config.batch_size,
            batch_length=config.batch_length,
            replay_context=config.replay_context,
            replay_sampling=str(config.replay.sampling),
            ppo_start_step=int(config.run.ppo_start_step),
            report_length=config.report_length,
            replica=config.replica,
            replicas=config.replicas,
        ),
    )

    return agent


def make_logger(config):
    step = elements.Counter()
    logdir = config.logdir
    multiplier = config.env.get(config.task.split("_")[0], {}).get("repeat", 1)
    outputs = []
    outputs.append(elements.logger.TerminalOutput(config.logger.filter, "Agent"))
    for output in config.logger.outputs:
        if output == "jsonl":
            outputs.append(elements.logger.JSONLOutput(logdir, "metrics.jsonl"))
            outputs.append(
                elements.logger.JSONLOutput(logdir, "scores.jsonl", "episode/score")
            )
        elif output == "tensorboard":
            outputs.append(elements.logger.TensorBoardOutput(logdir, config.logger.fps))
        elif output == "wandb":
            name = os.environ.get("WANDB_NAME") or "/".join(logdir.split("/")[-4:])
            outputs.append(elements.logger.WandBOutput(name, config=config.flat))
        elif output == "scope":
            outputs.append(elements.logger.ScopeOutput(elements.Path(logdir)))
        else:
            raise NotImplementedError(output)
    logger = elements.Logger(step, outputs, multiplier)
    return logger


def make_replay(config, folder, mode="train"):
    batlen = config.batch_length if mode == "train" else config.report_length
    consec = config.consec_train if mode == "train" else config.consec_report
    capacity = config.replay.size if mode == "train" else config.replay.size / 10
    length = consec * batlen + config.replay_context
    assert config.batch_size * length <= capacity

    directory = elements.Path(config.logdir) / folder
    if config.replicas > 1:
        directory /= f"{config.replica:05}"
    kwargs = dict(
        length=length,
        capacity=int(capacity),
        online=config.replay.online,
        chunksize=config.replay.chunksize,
        directory=directory,
    )

    sampling = str(config.replay.sampling)
    world_uniform_mix = float(config.replay.world_uniform_mix)
    if world_uniform_mix and sampling != "recent_world_uniform_behavior":
        raise ValueError("world_uniform_mix requires dual-view replay")
    if mode == "train" and sampling != "recent_world_uniform_behavior":
        raise ValueError("The maintained learner requires dual-view replay")
    if mode == "train" and sampling == "recent_world_uniform_behavior":
        from .replay import DualViewReplay

        return DualViewReplay(
            **kwargs,
            optimized_length=int(consec * batlen),
            recency_decay=float(config.replay.recency_decay),
            seed=int(config.seed),
            world_uniform_mix=world_uniform_mix,
            behavior_recency_decay=float(config.replay.behavior_recency_decay),
            behavior_uniform_mix=float(config.replay.behavior_uniform_mix),
        )
    if sampling != "uniform" and mode == "train":
        raise ValueError(f"unsupported replay sampling: {sampling!r}")
    return embodied.replay.Replay(**kwargs)


def make_env(config, index, **overrides):
    suite, task = config.task.split("_", 1)
    kwargs = config.env.get(suite, {})
    kwargs.update(overrides)
    if kwargs.pop("use_seed", False):
        kwargs["seed"] = int(config.seed) + int(index)
    if suite == "smac":
        from .envs.smac import SMACEnv

        env = SMACEnv(task, **kwargs)
    else:
        raise ValueError(f"MA-JEPA supports only SMAC: {config.task!r}")
    return wrap_env(env, config)


def wrap_env(env, config):
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.UnifyDtypes(env)
    env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env


def make_stream(config, replay, mode):
    train_mode = mode in {"train", "train_world", "train_behavior"}
    fn = bind(replay.sample, config.batch_size, mode)
    stream = embodied.streams.Stateless(fn)
    stream = embodied.streams.Consec(
        stream,
        length=(config.batch_length if train_mode else config.report_length),
        consec=(config.consec_train if train_mode else config.consec_report),
        prefix=config.replay_context,
        strict=train_mode,
        contiguous=True,
    )
    return stream


if __name__ == "__main__":
    main()
