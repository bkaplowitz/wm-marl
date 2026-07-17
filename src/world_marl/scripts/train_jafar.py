"""Train and evaluate the source-derived Jafar world-model arm."""

from world_marl.latent_action_world_model.runner import main as run_arm


def main(argv: list[str] | None = None) -> int:
    return run_arm("jafar", argv)


if __name__ == "__main__":
    raise SystemExit(main())
