"""Load and verify the pinned JEPA Transformer research foundation."""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path
from typing import Any


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def source_manifest_path() -> Path:
    return repository_root() / "configs" / "jepatransformer" / "sources.toml"


def protocol_manifest_path() -> Path:
    return (
        repository_root() / "configs" / "jepatransformer" / "visual_dmc_protocol.toml"
    )


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _git_output(checkout: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_foundation(root: str | Path | None = None) -> dict[str, Any]:
    """Verify source revisions and the fixed visual-DMC protocol."""
    root = Path(root or repository_root()).resolve()
    sources = _load_toml(root / "configs" / "jepatransformer" / "sources.toml")
    protocol = _load_toml(
        root / "configs" / "jepatransformer" / "visual_dmc_protocol.toml"
    )

    verified_sources = []
    for source in sources["implementations"]:
        checkout = root / source["local_path"]
        if not checkout.is_dir():
            raise FileNotFoundError(
                f"missing pinned source {source['id']}: {checkout}; "
                "run 'git submodule update --init --recursive'"
            )
        revision = _git_output(checkout, "rev-parse", "HEAD")
        if revision != source["commit"]:
            raise RuntimeError(
                f"source revision mismatch for {source['id']}: "
                f"expected {source['commit']}, found {revision}"
            )
        status = _git_output(checkout, "status", "--porcelain", "--untracked-files=no")
        if status:
            raise RuntimeError(f"pinned source {source['id']} has local modifications")
        verified_sources.append(
            {
                "id": source["id"],
                "commit": revision,
                "local_path": source["local_path"],
            }
        )

    observations = protocol["observations"]
    if observations["upstream_profile"] != "dmc_vision":
        raise ValueError("visual protocol must use the upstream dmc_vision profile")
    if [observations["height"], observations["width"]] != [64, 64]:
        raise ValueError("visual protocol must use 64x64 observations")
    if observations["proprio"]:
        raise ValueError("visual protocol must not expose proprioceptive observations")
    if protocol["evaluation"]["checkpoint_policy"] != "latest":
        raise ValueError("evaluation must use the latest policy")
    if protocol["evaluation"]["checkpoint_search"]:
        raise ValueError("checkpoint search is not permitted")

    return {
        "source_snapshot_date": sources["snapshot_date"],
        "sources": verified_sources,
        "protocol": protocol["name"],
        "observation_profile": observations["upstream_profile"],
        "phase_1_tasks": protocol["phase_1"]["tasks"],
        "phase_1_seeds": protocol["phase_1"]["seeds"],
    }
