"""Extraction tests for plot_wm_runtime (comparison.json runtime rows)."""

import json

from world_marl.scripts.plot_wm_runtime import collect_runtime_rows


def _write_comparison(dirpath, rows):
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "comparison.json").write_text(json.dumps({"rows": rows}))


def test_collect_runtime_rows_across_seeds(tmp_path):
    _write_comparison(
        tmp_path / "seed0" / "wm_comparison_a",
        [
            {
                "env": "brax:reacher",
                "arm": "jepa",
                "runtime_seconds": 3200.0,
                "policy_trained_mean": -36.0,
            },
            {
                "env": "brax:reacher",
                "arm": "llada2",
                "runtime_seconds": 900.0,
                "policy_trained_mean": -81.0,
            },
        ],
    )
    _write_comparison(
        tmp_path / "seed1" / "wm_comparison_b",
        [
            {
                "env": "brax:reacher",
                "arm": "jepa",
                "runtime_seconds": 3300.0,
                "policy_trained_mean": -40.0,
            }
        ],
    )
    rows = collect_runtime_rows(
        [
            tmp_path / "seed0" / "wm_comparison_a",
            tmp_path / "seed1" / "wm_comparison_b",
        ],
        None,
    )
    assert set(rows) == {"brax_reacher"}
    assert set(rows["brax_reacher"]) == {"jepa", "llada2"}
    jepa = rows["brax_reacher"]["jepa"]
    assert [row["seed"] for row in jepa] == ["seed0", "seed1"]
    assert [row["runtime_seconds"] for row in jepa] == [3200.0, 3300.0]
    assert rows["brax_reacher"]["llada2"][0]["trained_mean"] == -81.0


def test_collect_runtime_rows_env_filter_and_missing_file(tmp_path):
    _write_comparison(
        tmp_path / "seed0" / "wm_comparison_a",
        [
            {
                "env": "brax:reacher",
                "arm": "jepa",
                "runtime_seconds": 3200.0,
                "policy_trained_mean": -36.0,
            },
            {
                "env": "gymnax:CartPole-v1",
                "arm": "jepa",
                "runtime_seconds": 100.0,
                "policy_trained_mean": 400.0,
            },
        ],
    )
    no_file = tmp_path / "seed1" / "wm_comparison_b"
    no_file.mkdir(parents=True)
    rows = collect_runtime_rows(
        [tmp_path / "seed0" / "wm_comparison_a", no_file], ["brax_reacher"]
    )
    assert set(rows) == {"brax_reacher"}
    assert len(rows["brax_reacher"]["jepa"]) == 1
