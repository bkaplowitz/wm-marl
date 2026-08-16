"""Runtime entrypoint that adds isolated ablation configurations."""

from pathlib import Path

from ..main import main


if __name__ == "__main__":
    main(extra_config_path=Path(__file__).with_name("configs.yaml"))
