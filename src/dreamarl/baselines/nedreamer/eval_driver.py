"""Evaluate one official NE-Dreamer checkpoint inside its isolated runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(args.upstream_root.resolve()))
    import torch
    from hydra import compose, initialize_config_dir

    import tools
    from dreamer import Dreamer
    from envs import make_envs

    tools.set_seed_everywhere(args.training_seed)
    with initialize_config_dir(
        version_base=None,
        config_dir=str((args.upstream_root / "configs").resolve()),
    ):
        config = compose(
            config_name="configs",
            overrides=[
                f"env.task={args.task}",
                "model.rep_loss=ne_dreamer",
                f"device={args.device}",
                f"seed={args.training_seed}",
                f"env.seed={args.eval_seed}",
                "env.env_num=1",
                f"env.eval_episode_num={args.episodes}",
                f"trainer.eval_episode_num={args.episodes}",
                "trainer.eval_video_every=0",
                "trainer.s3_bucket=null",
                "model.imagination_decoding.enabled=false",
            ],
        )
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)
    agent = Dreamer(config.model, obs_space, act_space).to(args.device)
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    agent.load_state_dict(payload["agent_state_dict"])
    agent.eval()

    done = torch.ones(args.episodes, dtype=torch.bool, device=args.device)
    once_done = torch.zeros(args.episodes, dtype=torch.bool, device=args.device)
    decisions = torch.zeros(args.episodes, dtype=torch.int32, device=args.device)
    returns = torch.zeros(args.episodes, dtype=torch.float32, device=args.device)
    state = agent.get_initial_state(args.episodes)
    action = state["prev_action"].clone()
    with torch.no_grad():
        while not once_done.all():
            decisions += ~done * ~once_done
            transition_cpu, done_cpu = eval_envs.step(
                action.detach().cpu(), done.detach().cpu()
            )
            transition = transition_cpu.to(args.device, non_blocking=True)
            done = done_cpu.to(args.device)
            transition["action"] = action
            action, state = agent.act(transition, state, eval=True)
            returns += transition["reward"][:, 0] * ~once_done
            once_done |= done

    action_repeat = int(config.env.action_repeat)
    rows = [
        {
            "episode": index,
            "episode_return": float(returns[index].cpu()),
            "agent_decisions": int(decisions[index].cpu()),
            "real_environment_transitions": int(decisions[index].cpu()) * action_repeat,
        }
        for index in range(args.episodes)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    for collection in (train_envs.envs, eval_envs.envs):
        for env in collection:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
