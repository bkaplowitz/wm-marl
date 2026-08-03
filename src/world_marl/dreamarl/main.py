import os
import pathlib
from functools import partial as bind

import elements
import embodied
import numpy as np
import portal
import ruamel.yaml as yaml

folder = pathlib.Path(__file__).parent


def main(argv=None):
    from .agent import Agent

    [elements.print(line) for line in Agent.banner]

    configs = elements.Path(folder / "configs.yaml").read()
    configs = yaml.YAML(typ="safe").load(configs)
    parsed, other = elements.Flags(configs=["defaults"]).parse_known(argv)
    config = elements.Config(configs["defaults"])
    for name in parsed.configs:
        config = config.update(configs[name])
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
    if not config.script.endswith(("_env", "_replay")):
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

    elif config.script == "train_eval":
        embodied.run.train_eval(
            bind(make_agent, config),
            bind(make_replay, config, "replay"),
            bind(make_replay, config, "eval_replay", "eval"),
            bind(make_env, config),
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

    elif config.script == "parallel":
        embodied.run.parallel.combined(
            bind(make_agent, config),
            bind(make_replay, config, "replay"),
            bind(make_replay, config, "replay_eval", "eval"),
            bind(make_env, config),
            bind(make_env, config),
            bind(make_stream, config),
            bind(make_logger, config),
            args,
        )

    elif config.script == "parallel_env":
        is_eval = config.replica >= args.envs
        embodied.run.parallel.parallel_env(
            bind(make_env, config), config.replica, args, is_eval
        )

    elif config.script == "parallel_envs":
        is_eval = config.replica >= args.envs
        embodied.run.parallel.parallel_envs(
            bind(make_env, config), bind(make_env, config), args
        )

    elif config.script == "parallel_replay":
        embodied.run.parallel.parallel_replay(
            bind(make_replay, config, "replay"),
            bind(make_replay, config, "replay_eval", "eval"),
            bind(make_stream, config),
            args,
        )

    else:
        raise NotImplementedError(config.script)


def make_agent(config):
    from .agent import Agent

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
    return Agent(
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
            report_length=config.report_length,
            replica=config.replica,
            replicas=config.replicas,
        ),
    )


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
        elif output == "expa":
            exp = logdir.split("/")[-4]
            run = "/".join(logdir.split("/")[-3:])
            proj = "embodied" if logdir.startswith(("/cns/", "gs://")) else "debug"
            outputs.append(
                elements.logger.ExpaOutput(
                    exp, run, proj, config.logger.user, config.flat
                )
            )
        elif output == "wandb":
            name = os.environ.get("WANDB_NAME") or "/".join(logdir.split("/")[-4:])
            outputs.append(elements.logger.WandBOutput(name))
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

    if config.replay.fracs.uniform < 1 and mode == "train":
        assert config.jax.compute_dtype in ("bfloat16", "float32"), (
            "Gradient scaling for low-precision training can produce invalid loss "
            "outputs that are incompatible with prioritized replay."
        )
        recency = 1.0 / np.arange(1, capacity + 1) ** config.replay.recexp
        selectors = embodied.replay.selectors
        kwargs["selector"] = _SelectorMixture(
            dict(
                uniform=selectors.Uniform(),
                priority=selectors.Prioritized(**config.replay.prio),
                recency=selectors.Recency(recency),
            ),
            config.replay.fracs,
        )

    return embodied.replay.Replay(**kwargs)


class _SelectorMixture(embodied.replay.selectors.Mixture):
    """Mixture selector with the collection-size contract Replay requires."""

    def __len__(self):
        return len(self.selectors[0]) if self.selectors else 0


def make_env(config, index, **overrides):
    suite, task = config.task.split("_", 1)
    if suite != "meltingpot":
        raise ValueError(
            f"DreaMARL currently supports Melting Pot tasks, got {config.task!r}"
        )
    from .meltingpot import MeltingPotEnv

    kwargs = config.env.get(suite, {})
    kwargs.update(overrides)
    if kwargs.pop("use_seed", False):
        kwargs["seed"] = hash((config.seed, index)) % (2**32 - 1)
    return wrap_env(MeltingPotEnv(task, **kwargs), config)


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
    fn = bind(replay.sample, config.batch_size, mode)
    stream = embodied.streams.Stateless(fn)
    stream = embodied.streams.Consec(
        stream,
        length=config.batch_length if mode == "train" else config.report_length,
        consec=config.consec_train if mode == "train" else config.consec_report,
        prefix=config.replay_context,
        strict=(mode == "train"),
        contiguous=True,
    )

    return stream


if __name__ == "__main__":
    try:
        import wandb

        print(f"WandB version: {wandb.__version__}")
        print(f"WandB init available: {hasattr(wandb, 'init')}")
    except ImportError as e:
        print(f"WandB import failed: {e}")
    main()
