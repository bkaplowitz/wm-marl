from __future__ import annotations

from typing import Any


def world_model_sources() -> dict[str, dict[str, Any]]:
    return {
        "dreamer_v3": {
            "paper": "DreamerV3: Mastering Diverse Domains through World Models",
            "paper_url": "https://arxiv.org/abs/2301.04104",
        },
        "genie_2": {
            "announcement": "Genie 2: A large-scale foundation world model",
            "announcement_url": "https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/",
            "role": "primary continuous latent diffusion target",
        },
        "genie": {
            "paper": "Genie: Generative Interactive Environments",
            "paper_url": "https://arxiv.org/abs/2402.15391",
            "role": "genie1_vq_maskgit_ablation",
        },
        "genie_3": {
            "announcement": "Genie 3: A new frontier for world models",
            "announcement_url": "https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/",
            "role": "capability target; complete architecture is not public",
        },
        "jasmine": {
            "paper": "Jasmine",
            "paper_url": "https://arxiv.org/abs/2510.27002",
            "repo_url": "https://github.com/p-doom/jasmine",
            "role": "implementation reference",
        },
        "jafar": {
            "repo_url": "https://github.com/FLAIROx/jafar",
            "role": "implementation reference",
        },
        "dm_control": {
            "repo_url": "https://github.com/google-deepmind/dm_control",
            "point_mass_url": "https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/point_mass.py",
            "pixels_wrapper_url": "https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/wrappers/pixels.py",
            "observation_mode": "official_pixel_wrapper",
        },
        "mujoco_playground": {
            "repo_url": "https://github.com/google-deepmind/mujoco_playground",
            "role": "JAX-native DMC-style control adapter",
            "physics_backend": "mjx",
            "observation_mode": "vector",
        },
    }


visual_model_sources = world_model_sources
