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
from world_marl.scripts.train_single_genwm import (
    _WandbPhaseMetrics,
    _configure_wandb_metrics,
    _fit_models,
    _wandb_phase_namespace,
)

OBS_DIM = 4


class _FakeWandbRun:
    def __init__(self):
        self.logged = []
        self.defined = []

    def log(self, row):
        self.logged.append(row)

    def define_metric(self, name, **kwargs):
        self.defined.append((name, kwargs))


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


def test_wandb_phase_namespace_covers_label_grammar():
    assert _wandb_phase_namespace("run 0 genie") == "genie"
    assert _wandb_phase_namespace("run 0 fit") == "fit"
    assert _wandb_phase_namespace("run 0 policy") == "policy"
    assert _wandb_phase_namespace("run 0 model-free") == "model_free"
    assert _wandb_phase_namespace("run 1 online 2 genie") == "genie"
    assert _wandb_phase_namespace("run 1 online 2 fit") == "fit"
    assert _wandb_phase_namespace("run 1 online 2 policy") == "policy"
    assert _wandb_phase_namespace("run 1 model-free online 2") == "model_free"


def test_wandb_phase_metrics_namespaced_cumulative_rows():
    run = _FakeWandbRun()
    phase_metrics = _WandbPhaseMetrics(run)
    phase_metrics.log("run 0 fit", 1, 4, {"wm_loss": 10.0})
    phase_metrics.log("run 0 fit", 4, 4, {"wm_loss": 8.0})
    phase_metrics.log("run 0 online 0 fit", 2, 2, {"wm_loss": 6.0})
    phase_metrics.log("run 0 policy", 1, 3, {"total_loss": 2.0}, samples_per_step=12)
    phase_metrics.log(
        "run 0 online 0 policy", 2, 3, {"total_loss": 0.5}, samples_per_step=12
    )
    phase_metrics.log(
        "run 0 model-free", 6, 54, {"total_loss": 77.4}, samples_per_step=16
    )
    assert run.logged == [
        {"fit/wm_loss": 10.0, "fit/step": 1},
        {"fit/wm_loss": 8.0, "fit/step": 4},
        {"fit/wm_loss": 6.0, "fit/step": 6},
        {"policy/total_loss": 2.0, "policy/samples": 12},
        {"policy/total_loss": 0.5, "policy/samples": 60},
        {"model_free/total_loss": 77.4, "model_free/samples": 96},
    ]


def test_configure_wandb_metrics_declares_phase_axes():
    run = _FakeWandbRun()
    _configure_wandb_metrics(run)
    defined = dict(run.defined)
    assert defined["eval/real_env_steps"] == {"hidden": True}
    assert defined["eval/return"] == {"step_metric": "eval/real_env_steps"}
    for namespace, axis in (
        ("genie", "step"),
        ("fit", "step"),
        ("policy", "samples"),
        ("model_free", "samples"),
    ):
        assert defined[f"{namespace}/{axis}"] == {"hidden": True}
        assert defined[f"{namespace}/*"] == {"step_metric": f"{namespace}/{axis}"}
    assert defined["fit/wm_loss"] == {"summary": "min"}
    assert defined["policy/total_loss"] == {"summary": "min"}
    assert defined["model_free/total_loss"] == {"summary": "min"}


def test_fit_models_mirrors_fit_namespace_to_wandb(tmp_path):
    config = _tiny_config()
    data = _tiny_replay(16)
    obs_tokenizer = fit_quantile_tokenizer(
        np.concatenate([data["observations"], data["next_observations"]]),
        config.obs_bins,
    )
    wm_state = create_genwm_state(jax.random.PRNGKey(0), config)
    head_state = create_head_state(jax.random.PRNGKey(1), config)
    run = _FakeWandbRun()
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
        metrics_logger=RunLogger(tmp_path),
        wandb_metrics=_WandbPhaseMetrics(run),
    )
    assert [row["fit/step"] for row in run.logged] == [1, 2]
    assert all("fit/wm_loss" in row for row in run.logged)
    assert all("fit/head_total_loss" in row for row in run.logged)
