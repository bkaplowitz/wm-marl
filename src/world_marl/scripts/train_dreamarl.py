"""Train the maintained first-party DreaMARL algorithm on CoinGame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from world_marl.dreamarl.config import DreaMARLConfig
from world_marl.dreamarl.environments import CoinGameAdapter
from world_marl.dreamarl.experiment import (
    DreaMARLExperimentConfig,
    run_dreamarl_experiment,
)
from world_marl.logging import RunLogger, WandbConfig, to_jsonable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-env-steps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--max-episode-steps", type=int, default=64)
    parser.add_argument("--initial-random-steps", type=int, default=4_096)
    parser.add_argument("--initial-learner-updates", type=int, default=64)
    parser.add_argument("--collect-steps", type=int, default=16)
    parser.add_argument("--learner-updates-per-collect", type=int, default=16)
    parser.add_argument("--evaluation-interval", type=int, default=10_000)
    parser.add_argument("--evaluation-episodes", type=int, default=128)
    parser.add_argument("--evaluation-num-envs", type=int, default=64)
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="world-marl")
    parser.add_argument("--wandb-entity", default="osaze-obahor")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = CoinGameAdapter(max_episode_steps=args.max_episode_steps)
    model = DreaMARLConfig(
        max_agents=adapter.spec.num_agents,
        action_dim=adapter.spec.action_dim,
    )
    run_name = f"dreamarl_coin_game_seed{args.seed}"
    run_dir = Path(args.run_dir or f"runs/{run_name}")
    experiment = DreaMARLExperimentConfig(
        seed=args.seed,
        total_environment_steps=args.total_env_steps,
        num_envs=args.num_envs,
        max_episode_steps=args.max_episode_steps,
        initial_random_steps=args.initial_random_steps,
        initial_learner_updates=args.initial_learner_updates,
        collect_steps=args.collect_steps,
        learner_updates_per_collect=args.learner_updates_per_collect,
        evaluation_interval=args.evaluation_interval,
        evaluation_episodes=args.evaluation_episodes,
        evaluation_num_envs=args.evaluation_num_envs,
        checkpoint_interval=args.checkpoint_interval,
        run_dir=str(run_dir),
        resume_from=args.resume_from,
    )
    wandb = (
        WandbConfig(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            group="dreamarl_coin_game",
            tags=("dreamarl", "coin_game", "marl"),
            mode=args.wandb_mode,
            config={
                "algorithm": "DreaMARL",
                "model": model.to_dict(),
                "experiment": to_jsonable(experiment),
            },
        )
        if args.wandb
        else None
    )
    logger = RunLogger(run_dir, wandb_config=wandb)
    try:
        summary = run_dreamarl_experiment(model, experiment, logger)
    except BaseException:
        logger.close(exit_code=1)
        raise
    logger.close()
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
