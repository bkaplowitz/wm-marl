"""Source and infrastructure verification for first-party DreaMARL."""

from __future__ import annotations

import hashlib
from pathlib import Path

from world_marl.baselines.dreamer_cdp.config import default_upstream_root
from world_marl.baselines.dreamer_cdp.launcher import verify_upstream


ALGORITHM_FILES = (
    "agent.py",
    "axes.py",
    "configs.yaml",
    "environments.py",
    "main.py",
    "rssm.py",
    "transformer_rssm.py",
)


def algorithm_root() -> Path:
    """Return the source directory executed by the DreaMARL launcher."""

    return Path(__file__).resolve().parent


def algorithm_entrypoint() -> Path:
    return algorithm_root() / "main.py"


def runtime_fingerprint() -> str:
    digest = hashlib.sha256()
    for name in ALGORITHM_FILES:
        digest.update(name.encode())
        digest.update((algorithm_root() / name).read_bytes())
    return digest.hexdigest()[:12]


def verify_first_party_source() -> dict[str, str]:
    """Verify the infrastructure pin and hash the executable source tree."""

    revision = verify_upstream(default_upstream_root())
    missing = [
        name for name in ALGORITHM_FILES if not (algorithm_root() / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"DreaMARL source files are missing: {missing}")
    return {
        "infrastructure_commit": revision,
        "algorithm_fingerprint": runtime_fingerprint(),
    }
