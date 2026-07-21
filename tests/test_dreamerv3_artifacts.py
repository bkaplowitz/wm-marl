from __future__ import annotations

import json

import numpy as np

from world_marl.baselines.dreamerv3.artifacts import (
    EpisodeScore,
    bin_episode_scores,
    extract_episode_scores,
    load_official_reference,
    normalize_evaluation_artifacts,
    normalize_training_artifacts,
    read_jsonl,
    summarize_returns,
)
from world_marl.baselines.dreamerv3.config import default_upstream_root


def _write_scores(path, rows):
    path.parent.mkdir(parents=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_jsonl_reader_tolerates_partial_final_line(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text(
        '{"step": 10, "episode/score": 2.0}\n{"step":',
        encoding="utf-8",
    )
    records = read_jsonl(path)
    assert records == [{"step": 10, "episode/score": 2.0}]


def test_extract_bin_and_summarize_scores():
    scores = extract_episode_scores(
        [
            {"step": 5_000, "episode/score": 1.0},
            {"step": 9_000, "episode/score": 3.0},
            {"step": 15_000, "episode/score": 5.0},
        ]
    )
    assert scores[0] == EpisodeScore(5_000, 1.0)
    curve = bin_episode_scores(scores, bin_size=10_000)
    assert curve == [
        {
            "env_steps": 10_000,
            "episode_return_mean": 2.0,
            "episode_return_std": 1.0,
            "episodes": 2,
        },
        {
            "env_steps": 20_000,
            "episode_return_mean": 5.0,
            "episode_return_std": 0.0,
            "episodes": 1,
        },
    ]
    summary = summarize_returns([0.0, 10.0, 20.0, 30.0])
    assert summary["mean"] == 15.0
    assert summary["cvar10"] == 0.0


def test_official_reacher_reference_is_loaded_from_pinned_upstream():
    reference = load_official_reference(
        default_upstream_root(), task="dmc_reacher_easy"
    )
    assert reference is not None
    assert reference["seeds"] == [0, 1, 2, 3, 4]
    assert reference["env_steps"][-1] == 490_000
    assert np.isclose(reference["mean"][-1], 961.8)


def test_training_normalization_writes_shared_artifacts(tmp_path):
    experiment = tmp_path / "run"
    _write_scores(
        experiment / "upstream" / "scores.jsonl",
        [
            {"step": 5_000, "episode/score": 100.0},
            {"step": 15_000, "episode/score": 300.0},
        ],
    )
    upstream = tmp_path / "empty-upstream"
    upstream.mkdir()
    summary = normalize_training_artifacts(
        experiment,
        upstream_root=upstream,
        task="dmc_reacher_easy",
        seed=7,
        train_steps_budget=20_000,
    )
    assert summary["online_training_episodes"]["mean"] == 200.0
    assert summary["latest_checkpoint_train_env_steps"] is None
    assert (experiment / "normalized" / "training_curve.json").is_file()
    assert (experiment / "normalized" / "training_curve.png").is_file()


def test_evaluation_normalization_counts_eval_steps_separately(tmp_path):
    eval_dir = tmp_path / "eval"
    _write_scores(
        eval_dir / "upstream" / "scores.jsonl",
        [
            {"step": 1_001, "episode/score": 950.0},
            {"step": 2_002, "episode/score": 850.0},
            {"step": 3_003, "episode/score": 100.0},
        ],
    )
    summary = normalize_evaluation_artifacts(
        eval_dir,
        requested_episodes=2,
        train_env_steps=500_000,
        success_threshold=900.0,
    )
    assert summary["returns"] == [950.0, 850.0]
    assert summary["success_rate"] == 0.5
    assert summary["eval_env_steps"] == 2_002
    assert summary["total_real_env_steps"] == 502_002
