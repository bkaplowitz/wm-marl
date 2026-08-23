"""Model registry used only by the ablation launcher."""

from types import MappingProxyType

from ..models import visual as canonical_visual
from ..world_model.backend import WorldModelBackend
from ..world_model.transformer import parallel_backend
from . import visual
from .rssm import rssm_backend


def world_model_backend(name: str) -> WorldModelBackend:
    if name == "rssm":
        return rssm_backend()
    if name != "parallel_transformer":
        raise ValueError(f"unknown ablation backend: {name}")
    base = parallel_backend()
    return WorldModelBackend(
        name=base.name,
        encoders=MappingProxyType(
            {
                "simple": canonical_visual.Encoder,
                "vit": visual.ViTEncoder,
            }
        ),
        decoders=MappingProxyType({"simple": visual.Decoder}),
        dynamics=base.dynamics,
        feature_tensor=base.feature_tensor,
        replay_entries=base.replay_entries,
    )
