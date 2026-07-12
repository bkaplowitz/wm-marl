"""Configure accelerator determinism before importing the JEPA trainer."""

from __future__ import annotations

import sys

from world_marl.determinism import configure_deterministic_environment


def main() -> None:
    requested = (
        "--deterministic-compute" in sys.argv
        and "--no-deterministic-compute" not in sys.argv
    )
    if requested:
        configure_deterministic_environment()

    from world_marl.scripts.train_dmc_jepa import main as train_main

    train_main()
