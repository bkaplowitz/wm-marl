"""Saved September 17 SC2 staging commands, reusable on fresh campaign volumes."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def stage(root, *, dry_run=False):
    root = Path(root)
    commands = [
        [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "3",
            "--output",
            str(root / "SC2.4.10.zip"),
            "https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip",
        ],
        ["unzip", "-q", "-P", "iagreetotheeula", str(root / "SC2.4.10.zip"), "-d", str(root)],
        [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "3",
            "--output",
            str(root / "SMAC_Maps.zip"),
            "https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip",
        ],
        ["unzip", "-q", str(root / "SMAC_Maps.zip"), "-d", str(root / "StarCraftII/Maps")],
    ]
    if dry_run:
        return commands
    if root.exists():
        raise FileExistsError(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(root.parent).free / 1e9
    if free_gb < 40:
        raise OSError(f"SC2 staging needs 40 GB free for archives, extraction and headroom; {free_gb:.1f} GB available at {root.parent}")
    root.mkdir(parents=True, exist_ok=False)
    for command in commands:
        print(json.dumps(command), flush=True)
        subprocess.run(command, check=True)
    hashes = {}
    for name, expected in {
        "Versions/Base75689/SC2_x64": "115f65a49191c504f6336f7d5b7818b23dbba3d7a45cacf9142c86c959abaa41",
        "Maps/SMAC_Maps/2s3z.SC2Map": "928d21db081910096628ecc14627caa972940a7ccb99cc3448de28f7ec770a12",
    }.items():
        with (root / "StarCraftII" / name).open("rb") as handle:
            hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
        if hashes[name] != expected:
            raise ValueError(f"SC2 asset checksum mismatch: {name}")
    receipt = {
        "completed": True,
        "sc2path": str(root / "StarCraftII"),
        "hashes": hashes,
    }
    (root / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(stage(args.directory, dry_run=args.dry_run), indent=2))
