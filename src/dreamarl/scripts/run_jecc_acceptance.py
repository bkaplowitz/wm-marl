"""Run the frozen hard-map acceptance gate for JECC."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dreamarl.runtime import repository_root


def _latest_checkpoint(experiment):
    latest = experiment / "run" / "ckpt" / "latest"
    checkpoint = latest.parent / latest.read_text().strip()
    if not (checkpoint / "done").is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _stage_replay(source, destination):
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(exist_ok=True)
        else:
            try:
                os.link(path, target)
            except OSError:
                shutil.copy2(path, target)


def _run(command, env, logfile):
    print(" ".join(command))
    with logfile.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            cwd=repository_root(),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            output.write(line)
            output.flush()
        returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def _evaluation_command(
    python,
    manifest,
    logdir,
    checkpoint,
    *,
    seed,
    episodes,
    envs,
    platform,
    old_probe=None,
    jecc=False,
):
    command = [
        str(python),
        "-m",
        "dreamarl.main",
        "--logdir",
        str(logdir),
        "--configs",
        *manifest["configs"],
        "--task",
        manifest["task"],
        "--seed",
        str(seed),
        "--agent.num_agents",
        str(manifest["num_agents"]),
        "--script",
        "eval_only",
        "--run.from_checkpoint",
        str(checkpoint),
        "--run.eval_eps",
        str(episodes),
        "--run.eval_policy_mode",
        "eval",
        "--run.envs",
        str(envs),
        "--jax.platform",
        platform,
        "--jax.precompile",
        "False",
        "--jax.profiler",
        "False",
        "--logger.outputs",
        "jsonl",
        "scope",
    ]
    if old_probe is not None:
        command.extend(
            [
                "--run.probe_controller",
                "True",
                "--run.probe_model",
                str(old_probe),
            ]
        )
    if jecc:
        command.extend(
            [
                "--agent.marl.stage",
                "jecc",
                "--agent.behavior_optimizer",
                "separated",
                "--agent.behavior_objective",
                "reinforce",
                "--agent.marl.jecc.pretrain_only",
                "True",
                "--agent.marl.jecc.diagnostic_controller",
                "True",
            ]
        )
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_experiment", type=Path)
    parser.add_argument("old_probe_model", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--episodes", type=int, default=96)
    parser.add_argument("--envs", type=int, default=6)
    parser.add_argument("--eval-seed", type=int, default=30123)
    parser.add_argument("--platform", choices=("cpu", "cuda", "tpu"), default="cuda")
    parser.add_argument("--python", type=Path)
    args = parser.parse_args(argv)

    source = args.source_experiment.expanduser().resolve()
    probe = args.old_probe_model.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest = json.loads((source / "launch.json").read_text())
    python = (
        args.python.expanduser() if args.python is not None else Path(sys.executable)
    )
    source_checkpoint = _latest_checkpoint(source)
    pretrain_logdir = output / "pretrain" / "run"
    _stage_replay(source / "run" / "replay", pretrain_logdir / "replay")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(Path(manifest["infrastructure_root"])),
            str(repository_root() / "src"),
            env.get("PYTHONPATH", ""),
        ]
    )
    pretrain = [
        str(python),
        "-m",
        "dreamarl.main",
        "--logdir",
        str(pretrain_logdir),
        "--configs",
        *manifest["configs"],
        "--task",
        manifest["task"],
        "--seed",
        str(manifest["seed"]),
        "--agent.num_agents",
        str(manifest["num_agents"]),
        "--agent.marl.stage",
        "jecc",
        "--agent.behavior_optimizer",
        "separated",
        "--agent.behavior_objective",
        "reinforce",
        "--agent.marl.jecc.pretrain_only",
        "True",
        "--script",
        "jecc_pretrain",
        "--run.from_checkpoint",
        str(source_checkpoint),
        "--run.from_checkpoint_regex",
        "^(?!opt/).*",
        "--run.jecc_pretrain_updates",
        str(args.updates),
        "--replay.sampling",
        str(manifest["replay_sampling"]),
        "--jax.platform",
        args.platform,
        "--jax.precompile",
        "False",
        "--jax.profiler",
        "False",
        "--logger.outputs",
        "jsonl",
        "scope",
    ]
    if manifest["replay_sampling"] == "recent":
        pretrain.extend(["--replay.size", "50000", "--replay.online", "False"])
    _run(pretrain, env, output / "pretrain.log")
    jecc_checkpoint = _latest_checkpoint(output / "pretrain")

    evaluations = {
        "b0": (source_checkpoint, None, False),
        "observational_critic": (source_checkpoint, probe, False),
        "jecc": (jecc_checkpoint, None, True),
    }
    results = {}
    for name, (checkpoint, old_probe, jecc) in evaluations.items():
        logdir = output / "evaluation" / name
        command = _evaluation_command(
            python,
            manifest,
            logdir,
            checkpoint,
            seed=args.eval_seed,
            episodes=args.episodes,
            envs=args.envs,
            platform=args.platform,
            old_probe=old_probe,
            jecc=jecc,
        )
        _run(command, env, output / f"evaluation_{name}.log")
        results[name] = json.loads((logdir / "evaluation_summary.json").read_text())

    baseline_wins = int(results["b0"].get("wins", 0))
    jecc_wins = int(results["jecc"].get("wins", 0))
    baseline_valid = baseline_wins > 0
    catastrophic = baseline_valid and jecc_wins == 0
    gate_passed = baseline_valid and jecc_wins > 0
    summary = {
        "source_checkpoint": str(source_checkpoint),
        "jecc_checkpoint": str(jecc_checkpoint),
        "updates": args.updates,
        "protocol": {
            "episodes": args.episodes,
            "workers": args.envs,
            "seed": args.eval_seed,
            "focal_agent": "worker modulo agent count",
            "episodes_per_worker": args.episodes // args.envs,
        },
        "gate_passed": gate_passed,
        "baseline_valid": baseline_valid,
        "catastrophic_zero_win_collapse": catastrophic,
        "results": results,
    }
    (output / "acceptance_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return int(not gate_passed)


if __name__ == "__main__":
    raise SystemExit(main())
