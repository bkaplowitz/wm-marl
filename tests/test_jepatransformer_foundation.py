from __future__ import annotations

import tomllib

from world_marl.jepa_transformer.foundation import (
    protocol_manifest_path,
    source_manifest_path,
    verify_foundation,
)


def test_foundation_verifies_all_required_source_pins():
    result = verify_foundation()
    assert result["observation_profile"] == "dmc_vision"
    assert result["phase_1_tasks"] == ["dmc_walker_walk", "dmc_cheetah_run"]
    assert result["phase_1_seeds"] == [0, 1]
    assert {source["id"] for source in result["sources"]} == {
        "dreamerv3",
        "dreamer-cdp",
        "ne-dreamer",
    }


def test_visual_protocol_has_no_checkpoint_search():
    with protocol_manifest_path().open("rb") as handle:
        protocol = tomllib.load(handle)
    assert protocol["observations"] == {
        "upstream_profile": "dmc_vision",
        "kind": "rgb",
        "height": 64,
        "width": 64,
        "action_repeat": 1,
        "proprio": False,
    }
    assert protocol["evaluation"]["checkpoint_policy"] == "latest"
    assert protocol["evaluation"]["checkpoint_search"] is False
    assert protocol["accounting"]["count_evaluation_transitions_separately"]


def test_source_manifest_records_exact_code_revisions():
    with source_manifest_path().open("rb") as handle:
        sources = tomllib.load(handle)
    implementations = {item["id"]: item for item in sources["implementations"]}
    assert implementations["dreamerv3"]["commit"] == (
        "e3f02248693a79dc8b0ebd62c93683888ddaccfe"
    )
    assert len(implementations["dreamer-cdp"]["commit"]) == 40
    assert len(implementations["ne-dreamer"]["commit"]) == 40
