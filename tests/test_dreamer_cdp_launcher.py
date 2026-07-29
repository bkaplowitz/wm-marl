from __future__ import annotations

import json
import sys
from pathlib import Path

from world_marl.baselines.dreamer_cdp.config import (
    OFFICIAL_DREAMER_CDP_COMMIT,
    DreamerCDPRunSpec,
    default_upstream_root,
)
from world_marl.baselines.dreamer_cdp.launcher import run_training, verify_upstream
from world_marl.scripts.train_dmc_dreamer_cdp import main as train_main


def test_pinned_official_dreamer_cdp_checkout_is_present():
    assert verify_upstream(default_upstream_root()) == OFFICIAL_DREAMER_CDP_COMMIT


def test_m2_command_is_visual_and_uses_250k_budget(tmp_path):
    spec = DreamerCDPRunSpec(
        experiment_dir=tmp_path / "run",
        python=Path(sys.executable),
        platform="cpu",
    )
    command = spec.command
    assert command[1].endswith("external/dreamer-cdp/dreamerv3/main.py")
    assert command[command.index("--configs") + 1] == "dmc_vision"
    assert command[command.index("--run.steps") + 1] == "250000"
    assert command[command.index("--logger.outputs") + 1 :][:2] == [
        "jsonl",
        "scope",
    ]
    metric_filter = command[command.index("--logger.filter") + 1]
    assert "train/dyn_ent" in metric_filter
    assert "train/rep_ent" in metric_filter


def test_debug_config_can_only_follow_visual_profile(tmp_path):
    spec = DreamerCDPRunSpec(
        experiment_dir=tmp_path / "run",
        python=Path(sys.executable),
        platform="cpu",
        configs=("dmc_vision", "debug"),
    )
    start = spec.command.index("--configs") + 1
    assert spec.command[start : start + 2] == ["dmc_vision", "debug"]


def test_debug_launcher_repairs_only_upstream_encoder_width(tmp_path):
    experiment = tmp_path / "debug"
    assert train_main(
        [
            "--experiment-dir",
            str(experiment),
            "--platform",
            "cpu",
            "--python",
            sys.executable,
            "--debug",
            "--dry-run",
        ]
    ) == 0
    launch = json.loads((experiment / "launch.json").read_text())
    command = launch["command"]
    index = command.index("--agent.enc.simple.depth")
    assert command[index + 1] == "64"


def test_dry_run_records_exact_source_and_representation_contract(tmp_path):
    spec = DreamerCDPRunSpec(
        experiment_dir=tmp_path / "run",
        python=Path(sys.executable),
        platform="cpu",
    )
    assert run_training(spec, dry_run=True) == 0
    launch = json.loads((spec.experiment_dir / "launch.json").read_text())
    assert launch["verified_upstream_commit"] == OFFICIAL_DREAMER_CDP_COMMIT
    assert launch["implementation"] == "fmi-basel/Dreamer-CDP"
    assert "detached" in launch["representation_contract"]


def test_official_delta_has_exact_predictive_gradient_contract():
    root = default_upstream_root()
    rssm = (root / "dreamerv3" / "rssm.py").read_text(encoding="utf-8")
    agent = (root / "dreamerv3" / "agent.py").read_text(encoding="utf-8")
    config = (root / "dreamerv3" / "configs.yaml").read_text(encoding="utf-8")
    assert "optax.losses.cosine_distance(sg(slow_tokens), pred_enc" in rssm
    assert "pred_enc = self.predictor(feat['deter'])" in rssm
    assert "sg(x, skip=self.config.dec_grad)" in agent
    assert "dyn_deter: 500" in config
    assert "enc_lr: 6e-6" in config
    assert "dyn_lr: 4e-4" in config
    assert "dec_grad: False" in config
