from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .config import DreamerProfile, ObservationMode
from .oracle import (
    OracleManifest,
    TensorSpec,
    _FIXTURE_CASES,
    _PROFILE_REVISIONS,
    _canonical_fixture_command,
    _canonical_fixture_request,
    _fixture_coordinates,
    _git_show,
    _sha256_bytes,
    _sha256_path,
    _source_hashes_for,
    _write_deterministic_npz,
)


Array = npt.NDArray[np.generic]
_PARSER_REGISTRY: dict[str, Callable[[argparse._SubParsersAction[Any]], None]] = {}


def _register_parser(
    name: str,
) -> Callable[
    [Callable[[argparse._SubParsersAction[Any]], None]],
    Callable[[argparse._SubParsersAction[Any]], None],
]:
    if type(name) is not str:
        raise TypeError("fixture parser name must be an exact string")

    def decorator(
        builder: Callable[[argparse._SubParsersAction[Any]], None],
    ) -> Callable[[argparse._SubParsersAction[Any]], None]:
        if name in _PARSER_REGISTRY:
            raise ValueError(f"fixture parser already registered: {name}")
        _PARSER_REGISTRY[name] = builder
        return builder

    return decorator


def _validate_reference(
    reference_checkout: str | Path,
    source_revision: str,
    source_name: str,
) -> Path:
    checkout = Path(reference_checkout)
    if not checkout.is_dir():
        raise ValueError(f"official checkout does not exist: {checkout}")
    for path, expected in _source_hashes_for(source_name, source_revision).items():
        actual = _sha256_bytes(_git_show(checkout, source_revision, path))
        if actual != expected:
            raise ValueError(f"official source hash mismatch: {path}")
    return checkout


def _canonical_request(
    *,
    profile: DreamerProfile | str,
    observation_mode: ObservationMode | str,
    source_revision: str,
    fixture_stem: str,
) -> str:
    if type(source_revision) is not str:
        raise TypeError("source revision must be an exact string")
    resolved_profile = DreamerProfile(profile)
    resolved_mode = ObservationMode(observation_mode)
    expected_profile, expected_mode, _case, _source, _dtype, _seed = (
        _fixture_coordinates(fixture_stem)
    )
    if resolved_profile is not expected_profile or resolved_mode is not expected_mode:
        raise ValueError("fixture stem does not match requested profile and mode")
    if source_revision != _PROFILE_REVISIONS[expected_profile]:
        raise ValueError("source revision does not match profile authority")
    return _canonical_fixture_request(fixture_stem)


def _write_pair(
    *,
    output_dir: str | Path,
    fixture_stem: str,
    arrays: Mapping[str, Array],
    manifest: OracleManifest,
) -> tuple[Path, Path]:
    if any(type(name) is not str for name in arrays):
        raise TypeError("array keys must be exact strings")
    _fixture_coordinates(fixture_stem)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("fixture output directory must be a real directory")
    fixture_path = destination / f"{fixture_stem}.npz"
    manifest_path = destination / f"{fixture_stem}.manifest.json"
    if fixture_path.exists() or fixture_path.is_symlink():
        raise ValueError(f"fixture destination already exists: {fixture_path}")
    if manifest_path.exists() or manifest_path.is_symlink():
        raise ValueError(f"fixture destination already exists: {manifest_path}")
    if manifest.fixture_file != fixture_path.name:
        raise ValueError("manifest fixture filename does not match fixture stem")
    expected_schema = {
        name: TensorSpec(tuple(value.shape), value.dtype.name)
        for name, value in sorted(arrays.items())
    }
    if dict(manifest.tensor_schema) != expected_schema:
        raise ValueError("manifest tensor schema does not match generated arrays")
    manifest.validate()

    fixture_published = False
    manifest_published = False
    try:
        with tempfile.TemporaryDirectory(
            dir=destination,
            prefix=f".{fixture_stem}.",
            suffix=".stage",
        ) as staging:
            stage_dir = Path(staging)
            staged_fixture = stage_dir / fixture_path.name
            staged_manifest = stage_dir / manifest_path.name
            _write_deterministic_npz(staged_fixture, arrays)
            if staged_fixture.is_symlink() or not staged_fixture.is_file():
                raise ValueError("staged fixture must be a regular file")
            if manifest.fixture_sha256 != _sha256_path(staged_fixture):
                raise ValueError(
                    "manifest fixture hash does not match generated fixture"
                )
            manifest.save(staged_manifest)
            if staged_manifest.is_symlink() or not staged_manifest.is_file():
                raise ValueError("staged manifest must be a regular file")
            reloaded = OracleManifest.load(
                staged_manifest,
                fixture_path=staged_fixture,
            )
            if reloaded.to_dict() != manifest.to_dict():
                raise ValueError("staged manifest does not roundtrip exactly")
            os.link(staged_fixture, fixture_path)
            fixture_published = True
            os.link(staged_manifest, manifest_path)
            manifest_published = True
        return fixture_path, manifest_path
    except BaseException:
        if manifest_published:
            manifest_path.unlink(missing_ok=True)
        if fixture_published:
            fixture_path.unlink(missing_ok=True)
        raise


def refresh_manifest(args: argparse.Namespace) -> Path:
    profile = DreamerProfile(args.profile)
    mode = ObservationMode(args.observation_mode)
    revision = args.source_revision
    stem = args.fixture_stem
    if type(revision) is not str or type(stem) is not str:
        raise TypeError("fixture revision and stem must be exact strings")
    expected_profile, expected_mode, _case, source, _dtype, _seed = (
        _fixture_coordinates(stem)
    )
    if profile is not expected_profile:
        raise ValueError("fixture stem profile does not match requested profile")
    if mode is not expected_mode:
        raise ValueError("fixture stem observation mode does not match requested mode")
    if revision != _PROFILE_REVISIONS[profile]:
        raise ValueError("source revision does not match profile authority")

    output_dir = Path(args.output_dir)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError("fixture output directory must be a real directory")
    manifest_path = output_dir / f"{stem}.manifest.json"
    fixture_path = output_dir / f"{stem}.npz"
    if manifest_path.parent != output_dir or fixture_path.parent != output_dir:
        raise ValueError("fixture paths must be immediate output-directory children")
    if manifest_path.is_symlink() or fixture_path.is_symlink():
        raise ValueError("fixture pair must not contain symlinks")
    if not manifest_path.is_file() or not fixture_path.is_file():
        raise ValueError(f"fixture pair does not exist: {stem}")

    previous = OracleManifest.load(manifest_path, fixture_path=fixture_path)
    expected_request = _canonical_request(
        profile=profile,
        observation_mode=mode,
        source_revision=revision,
        fixture_stem=stem,
    )
    if previous.generator_request != expected_request:
        raise ValueError("prior generator request does not match fixture coordinates")
    if previous.generator_command != _canonical_fixture_command(stem):
        raise ValueError("prior generator command does not match fixture coordinates")
    _validate_reference(args.reference_checkout, revision, source)
    previous.save(manifest_path)
    return manifest_path


@_register_parser("refresh-manifest")
def _register_refresh_manifest_parser(
    subparsers: argparse._SubParsersAction[Any],
) -> None:
    parser = subparsers.add_parser(
        "refresh-manifest",
        help="refresh metadata for an existing immutable NPZ fixture",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--profile", choices=[item.value for item in DreamerProfile], required=True
    )
    parser.add_argument(
        "--observation-mode",
        choices=[item.value for item in ObservationMode],
        required=True,
    )
    parser.add_argument("--reference-checkout", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixture-stem", choices=tuple(_FIXTURE_CASES), required=True)
    parser.set_defaults(handler=refresh_manifest)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DreamerV3 numerical fixtures",
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for builder in _PARSER_REGISTRY.values():
        builder(subparsers)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_PARSER_REGISTRY",
    "_canonical_request",
    "_parse_args",
    "_register_parser",
    "_register_refresh_manifest_parser",
    "_validate_reference",
    "_write_pair",
    "main",
    "refresh_manifest",
]
