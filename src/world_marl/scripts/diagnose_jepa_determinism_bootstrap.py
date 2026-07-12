"""Configure accelerator determinism before importing the JEPA diagnostic."""

from __future__ import annotations

import sys

from world_marl.determinism import configure_deterministic_environment


def main() -> None:
    if "--no-deterministic-compute" not in sys.argv:
        configure_deterministic_environment()

    from world_marl.scripts.diagnose_jepa_determinism import main as diagnostic_main

    diagnostic_main()
