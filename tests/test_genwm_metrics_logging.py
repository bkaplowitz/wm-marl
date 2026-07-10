"""Structured metrics.jsonl emission from train_single_genwm trainers.

The trainers write through the shared ``RunLogger`` (the same writer the jepa
harness uses), one record per log point: ``phase``/``step``/``total`` plus the
trainer's metric values.
"""

import json

import jax
import numpy as np

from world_marl.genwm import GenWMConfig, create_genwm_state, create_head_state
from world_marl.genwm import fit_quantile_tokenizer
from world_marl.logging import RunLogger
from world_marl.scripts.train_single_genwm import _fit_models

OBS_DIM = 4


def _tiny_config() -> GenWMConfig:
    return GenWMConfig(
        arm="llada2",
        obs_dim=OBS_DIM,
        action_dim=3,
        action_mode="discrete",
        obs_bins=5,
        action_bins=0,
        model_dim=8,
        num_heads=2,
        num_layers=1,
        mlp_ratio=2,
        block_size=1,
        steps_per_block=1,
    )


def _tiny_replay(transitions: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "observations": rng.normal(size=(transitions, OBS_DIM)).astype(np.float32),
        "actions": rng.integers(0, 3, size=(transitions,)).astype(np.int32),
        "rewards": rng.normal(size=(transitions,)).astype(np.float32),
        "dones": np.zeros(transitions, dtype=np.float32),
        "next_observations": rng.normal(size=(transitions, OBS_DIM)).astype(np.float32),
    }


def test_fit_models_writes_metrics_records(tmp_path):
    config = _tiny_config()
    data = _tiny_replay(16)
    obs_tokenizer = fit_quantile_tokenizer(
        np.concatenate([data["observations"], data["next_observations"]]),
        config.obs_bins,
    )
    wm_state = create_genwm_state(jax.random.PRNGKey(0), config)
    head_state = create_head_state(jax.random.PRNGKey(1), config)
    logger = RunLogger(tmp_path)
    _fit_models(
        wm_state,
        head_state,
        jax.random.PRNGKey(2),
        data,
        obs_tokenizer,
        None,
        config,
        steps=2,
        batch_size=4,
        rng=np.random.default_rng(1),
        log_every=1,
        quiet=True,
        label="run 0 fit",
        metrics_logger=logger,
    )
    records = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert [record["step"] for record in records] == [1, 2]
    assert all(record["phase"] == "run 0 fit" for record in records)
    assert all(record["total"] == 2 for record in records)
    assert all(np.isfinite(record["wm_loss"]) for record in records)
    assert all("head_total_loss" in record for record in records)
