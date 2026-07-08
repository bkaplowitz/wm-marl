from __future__ import annotations

from typing import Any


def world_model_sources() -> dict[str, dict[str, Any]]:
    return {
        "dreamer_v3": {
            "paper": "DreamerV3: Mastering Diverse Domains through World Models",
            "paper_url": "https://arxiv.org/abs/2301.04104",
        },
        "genie": {
            "paper": "Genie: Generative Interactive Environments",
            "paper_url": "https://arxiv.org/abs/2402.15391",
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
    }


visual_model_sources = world_model_sources
