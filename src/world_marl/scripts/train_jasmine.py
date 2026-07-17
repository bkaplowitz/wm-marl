"""Train and evaluate the source-derived Jasmine world-model arm."""

from world_marl.latent_action_world_model.runner import main as run_arm


def main(argv: list[str] | None = None) -> int:
    return run_arm("jasmine", argv)


if __name__ == "__main__":
    raise SystemExit(main())
