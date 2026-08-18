import os
import pathlib
from copy import deepcopy
from functools import partial as bind

import elements
import embodied
import numpy as np
import portal
import ruamel.yaml as yaml

folder = pathlib.Path(__file__).parent


def _merge_dicts(base, updates):
    """Recursively add an optional configuration layer."""

    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_configs(extra_config_path=None):
    configs = yaml.YAML(typ="safe").load(elements.Path(folder / "configs.yaml").read())
    if extra_config_path:
        extra = yaml.YAML(typ="safe").load(elements.Path(extra_config_path).read())
        configs["defaults"] = _merge_dicts(
            configs["defaults"], extra.pop("defaults", {})
        )
        configs.update(extra)
    return configs


def _worker_seed(seed: int, index: int) -> int:
    """Match the pinned DreamerV3 environment-worker seed mapping."""

    return hash((int(seed), int(index))) % (2**32 - 1)


def _validate_script(script: str, num_agents: int) -> None:
    if num_agents > 1 and (
        script == "train_eval" or script.startswith("parallel")
    ):
        raise ValueError(
            f"script={script} uses generic single-agent reporting and is not "
            "supported for MARL runs; use script=train with explicit curve "
            "evaluation"
        )


def main(argv=None, extra_config_path=None):
    from .marl.core import MARLCore

    [elements.print(line) for line in MARLCore.banner]

    configs = _load_configs(extra_config_path)
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
    _validate_script(str(config.script), int(config.agent.num_agents))
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

    dual_replay = str(config.replay.sampling) == "recent_world_uniform_behavior"
    run_args = dict(config.run)
    if dual_replay:
        run_args["train_ratio"] = 2 * float(run_args["train_ratio"])
    args = elements.Config(
        **run_args,
        replica=config.replica,
        replicas=config.replicas,
        logdir=config.logdir,
        batch_size=config.batch_size * (2 if dual_replay else 1),
        replay_ready_size=config.batch_size * config.batch_length,
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

    elif config.script == "utility_probe":
        from . import diagnostics

        if not config.run.probe_source:
            raise ValueError("utility_probe requires run.probe_source")
        replay_config = config.update(logdir=str(config.run.probe_source))
        diagnostics.utility_probe(
            bind(make_agent, config),
            bind(make_replay, replay_config, "replay"),
            bind(make_stream, config),
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
    if config.agent.get("ablation", False):
        from .ablations.algorithm import AblationAlgorithm as Algorithm
    else:
        from .marl.core import MARLCore as Algorithm

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
    dual_replay = str(config.replay.sampling) == "recent_world_uniform_behavior"
    return Algorithm(
        obs_space,
        act_space,
        elements.Config(
            **config.agent,
            logdir=config.logdir,
            seed=config.seed,
            jax=config.jax,
            batch_size=config.batch_size * (2 if dual_replay else 1),
            batch_length=config.batch_length,
            replay_context=config.replay_context,
            report_length=config.report_length,
            replica=config.replica,
            replicas=config.replicas,
            replay_sampling=config.replay.sampling,
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

    sampling = str(config.replay.sampling)
    if mode == "train" and sampling == "recent_world_uniform_behavior":
        from .replay import RecentWorldUniformBehaviorReplay

        if int(capacity) != 50_000:
            raise ValueError(
                "recent_world_uniform_behavior requires replay.size=50000"
            )
        return RecentWorldUniformBehaviorReplay(
            **kwargs,
            recency_decay=float(config.replay.recency_decay),
            seed=int(config.seed),
        )
    if sampling != "uniform" and mode == "train":
        raise ValueError(f"unsupported replay sampling: {sampling!r}")
    return embodied.replay.Replay(**kwargs)


def make_env(config, index, **overrides):
    suite, task = config.task.split("_", 1)
    kwargs = config.env.get(suite, {})
    kwargs.update(overrides)
    if kwargs.pop("use_seed", False):
        kwargs["seed"] = _worker_seed(config.seed, index)
    if suite == "meltingpot":
        from .envs.meltingpot import MeltingPotEnv

        env = MeltingPotEnv(task, **kwargs)
    elif suite == "dmc":
        from .envs.dmc import make_dmc
        from .envs.single_agent import SingletonAgentEnv

        if "seed" not in kwargs:
            raise ValueError("DMC requires env.dmc.use_seed=True")
        env = SingletonAgentEnv(make_dmc(task, **kwargs))
    else:
        raise ValueError(f"unsupported DreaMARL task: {config.task!r}")
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
    length = config.batch_length if mode == "train" else config.report_length
    consec = config.consec_train if mode == "train" else config.consec_report

    def make_one(sample_mode, batch_size):
        source = embodied.streams.Stateless(
            bind(replay.sample, batch_size, sample_mode)
        )
        return embodied.streams.Consec(
            source,
            length=length,
            consec=consec,
            prefix=config.replay_context,
            strict=(mode == "train"),
            contiguous=True,
        )

    dual = str(config.replay.sampling) == "recent_world_uniform_behavior"
    if mode == "train" and dual:
        stream = embodied.streams.Zip(
            [
                make_one("train_world", config.batch_size),
                make_one("train_behavior", config.batch_size),
            ]
        )

        def add_roles(data):
            shape = data["is_first"].shape
            if shape[0] != 2 * int(config.batch_size):
                raise ValueError(f"unexpected dual replay batch shape: {shape}")
            data = dict(data)
            data["replay_sample_role"] = np.concatenate(
                [
                    np.ones((config.batch_size, shape[1]), np.int8),
                    np.full((config.batch_size, shape[1]), 2, np.int8),
                ],
                axis=0,
            )
            return data

        return embodied.streams.Map(stream, add_roles)

    batch_size = config.batch_size * (2 if dual else 1)
    stream = make_one(mode, batch_size)
    if dual:
        stream = embodied.streams.Map(
            stream,
            lambda data: {
                **data,
                "replay_sample_role": np.zeros(
                    data["is_first"].shape, np.int8
                ),
            },
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
