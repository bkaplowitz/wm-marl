from __future__ import annotations

from typing import Any


def world_model_sources() -> dict[str, dict[str, Any]]:
    return {
        "dreamer_v3": {
            "paper": "DreamerV3: Mastering Diverse Domains through World Models",
            "paper_url": "https://arxiv.org/abs/2301.04104",
        },
        "jasmine": {
            "paper": "Jasmine",
            "paper_url": "https://arxiv.org/abs/2510.27002",
            "repo_url": "https://github.com/p-doom/jasmine",
            "commit": "420859bc99eecf6b07a7e9edf65d5d145935f1e1",
            "role": "continuous MAE and diffusion-forcing implementation source",
        },
        "jafar": {
            "repo_url": "https://github.com/FLAIROx/jafar",
            "commit": "5ff9fc7d5d744c8c2797ba3ad0a095ed7f2e2665",
            "role": "VQ-VAE and MaskGIT implementation source",
        },
        "dm_control": {
            "repo_url": "https://github.com/google-deepmind/dm_control",
            "point_mass_url": "https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/point_mass.py",
            "pixels_wrapper_url": "https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/wrappers/pixels.py",
            "observation_mode": "official_pixel_wrapper",
        },
        "mujoco_playground": {
            "repo_url": "https://github.com/google-deepmind/mujoco_playground",
            "role": "JAX-native Cartpole vision evaluation",
            "physics_backend": "mjx_warp",
            "renderer_backend": "mjwarp_batch_renderer",
            "observation_mode": "pixels",
        },
    }


visual_model_sources = world_model_sources
