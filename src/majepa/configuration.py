"""Resolve named profiles or reload the exact settings saved by a run."""

from copy import deepcopy
from pathlib import Path

import elements
from ruamel.yaml import YAML


def _merge(base, updates):
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(profiles=("baseline",), filename=None):
    if filename:
        if list(profiles) != ["baseline"]:
            raise ValueError("Choose a saved config or named profiles, not both")
        return elements.Config(
            YAML(typ="safe").load(Path(filename).expanduser().read_text())
        )
    path = Path(__file__).with_name("configs.yaml")
    profiles_by_name = YAML(typ="safe").load(path.read_text())
    resolved = profiles_by_name["defaults"]
    for name in profiles:
        if name not in profiles_by_name or name == "defaults":
            raise ValueError(f"Unknown profile: {name!r}")
        resolved = _merge(resolved, profiles_by_name[name])
    return elements.Config(resolved)
