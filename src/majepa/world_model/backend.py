"""Backend-neutral construction contract for a MA-JEPA world model."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


ModuleType = type[Any]
FeatureTensor = Callable[[Mapping[str, Any]], Any]
ReplayEntries = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class WorldModelBackend:
    """Factories and feature semantics required by the unchanged agent loop."""

    name: str
    encoders: Mapping[str, ModuleType]
    decoders: Mapping[str, ModuleType]
    dynamics: Mapping[str, ModuleType]
    feature_tensor: FeatureTensor
    replay_entries: ReplayEntries

    def encoder(self, name: str) -> ModuleType:
        return self._resolve("encoder", self.encoders, name)

    def dynamics_model(self, name: str) -> ModuleType:
        return self._resolve("dynamics", self.dynamics, name)

    def decoder(self, name: str) -> ModuleType:
        return self._resolve("decoder", self.decoders, name)

    def _resolve(
        self, kind: str, modules: Mapping[str, ModuleType], name: str
    ) -> ModuleType:
        try:
            return modules[name]
        except KeyError as error:
            available = ", ".join(sorted(modules)) or "none"
            raise ValueError(
                f"world-model backend {self.name!r} has no {kind} {name!r}; "
                f"available: {available}"
            ) from error
