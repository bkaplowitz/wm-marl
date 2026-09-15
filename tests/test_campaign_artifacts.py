import json
from types import SimpleNamespace

import pytest
import wandb
import elements

from majepa import main


def test_logger_records_resolved_config(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.delenv("MAJEPA_CAMPAIGN_MANIFEST", raising=False)
    monkeypatch.setattr(wandb, "init", lambda **kwargs: captured.update(kwargs))
    config = main._resolve_config_profiles(
        main._load_configs(), ["defaults", "smac_vector", "ma_jepa"]
    ).update(
        {
            "logdir": str(tmp_path),
            "logger.outputs": ["wandb"],
            "agent.imag_action_samples": 2,
            "seed": 7,
        }
    )
    main.make_logger(config)
    assert captured["config"]["agent"]["imag_action_samples"] == 2
    assert captured["config"]["seed"] == 7
    assert captured["config"]["run"] == dict(config.run)


@pytest.mark.parametrize("failure", [None, "upload", "digest", "missing_file"])
def test_artifact_remote_verification(monkeypatch, tmp_path, failure):
    monkeypatch.setenv("WANDB_DATA_DIR", str(tmp_path / "wandb-data"))
    monkeypatch.setenv("MAJEPA_CAMPAIGN_MANIFEST", str(tmp_path / "campaign.json"))
    (tmp_path / "result.json").write_text('{"wins": 2}')
    run = SimpleNamespace(id="unique", entity="team", project="project", summary={})
    remote = None

    def log_artifact(artifact):
        nonlocal remote
        manifest = SimpleNamespace(entries=dict(artifact.manifest.entries))
        remote = SimpleNamespace(digest=artifact.digest, manifest=manifest)
        uploaded = SimpleNamespace(name="unique-evaluation:v0", digest=artifact.digest)

        def wait(timeout):
            if failure == "upload":
                raise RuntimeError("upload failed")
            if failure == "digest":
                remote.digest = "wrong"
            if failure == "missing_file":
                remote.manifest.entries.clear()
            return uploaded

        return SimpleNamespace(wait=wait)

    run.log_artifact = log_artifact
    monkeypatch.setattr(wandb, "run", run)
    monkeypatch.setattr(
        wandb, "Api", lambda: SimpleNamespace(artifact=lambda name: remote)
    )
    if failure:
        with pytest.raises(RuntimeError):
            main.record_campaign_artifact(
                tmp_path, "evaluation", [tmp_path / "result.json"]
            )
        assert run.summary["artifacts/evaluation_verified"] is False
        assert not (tmp_path / "artifact_verification.json").exists()
    else:
        main.record_campaign_artifact(
            tmp_path, "evaluation", [tmp_path / "result.json"]
        )
        receipt = json.loads((tmp_path / "artifact_verification.json").read_text())
        assert receipt["evaluation"]["artifact"] == "team/project/unique-evaluation:v0"
        assert receipt["evaluation"]["files"] == ["result.json"]
        assert run.summary["artifacts/evaluation_verified"] is True


def test_artifacts_are_opt_in(monkeypatch, tmp_path):
    monkeypatch.delenv("MAJEPA_CAMPAIGN_MANIFEST", raising=False)
    monkeypatch.setattr(wandb, "run", None)
    main.record_campaign_artifact(tmp_path, "evaluation", [tmp_path / "absent"])
    assert not list(tmp_path.iterdir())


def test_campaign_source_records_provenance(monkeypatch, tmp_path):
    manifest = tmp_path / "campaign.json"
    manifest.write_text('{"commit": "frozen-revision", "campaign": "unique"}')
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"source")
    monkeypatch.setenv("MAJEPA_CAMPAIGN_MANIFEST", str(manifest))
    monkeypatch.setenv("MAJEPA_SOURCE_ARCHIVE", str(archive))
    monkeypatch.setenv("WANDB_MODE", "online")
    captured = {}
    artifacts = []
    monkeypatch.setattr(wandb, "init", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        main, "record_campaign_artifact", lambda *args: artifacts.append(args)
    )
    config = main._resolve_config_profiles(
        main._load_configs(), ["defaults", "smac_vector", "ma_jepa"]
    ).update({"logdir": str(tmp_path), "logger.outputs": ["wandb"]})
    main.make_logger(config)
    assert captured["config"]["campaign_provenance"]["commit"] == "frozen-revision"
    assert artifacts == [
        (str(tmp_path), "source", [tmp_path / "config.yaml", manifest, archive])
    ]


@pytest.mark.parametrize(
    "updates,mode",
    [
        ({"logger.outputs": ["jsonl"]}, "online"),
        ({"run.final_save": False}, "online"),
        ({}, "offline"),
    ],
)
def test_campaign_rejects_unverifiable_recording(monkeypatch, tmp_path, updates, mode):
    monkeypatch.setenv("MAJEPA_CAMPAIGN_MANIFEST", str(tmp_path / "campaign.json"))
    monkeypatch.setenv("WANDB_MODE", mode)
    config = main._resolve_config_profiles(main._load_configs(), ["defaults"])
    config = config.update(
        {"logdir": str(tmp_path), "logger.outputs": ["wandb"], **updates}
    )
    with pytest.raises(ValueError, match="campaign recording requires"):
        main.make_logger(config)


@pytest.mark.parametrize("failure", [False, True])
def test_main_finishes_campaign_with_failure_status(monkeypatch, tmp_path, failure):
    from majepa import train

    monkeypatch.setenv("MAJEPA_CAMPAIGN_MANIFEST", str(tmp_path / "campaign.json"))
    monkeypatch.setattr(main.portal, "setup", lambda **kwargs: None)
    finished = []
    monkeypatch.setattr(
        wandb, "run", SimpleNamespace(finish=lambda **kwargs: finished.append(kwargs))
    )

    def run_training(*args):
        if failure:
            raise RuntimeError("artifact verification failed")

    monkeypatch.setattr(train, "train", run_training)
    if failure:
        with pytest.raises(RuntimeError, match="artifact verification failed"):
            main.main(["--logdir", str(tmp_path)])
    else:
        main.main(["--logdir", str(tmp_path)])
    assert finished == [{"exit_code": int(failure)}]


def test_training_uploads_completed_final_checkpoint(monkeypatch, tmp_path):
    from majepa import train

    monkeypatch.setenv("MAJEPA_CAMPAIGN_MANIFEST", str(tmp_path / "campaign.json"))
    config = main._resolve_config_profiles(main._load_configs(), ["defaults"])
    args = elements.Config(
        **config.run,
        logdir=str(tmp_path),
        batch_size=1,
        batch_length=1,
        num_agents=5,
    ).update(steps=50_000, final_save=True)
    logger = elements.Logger(elements.Counter(50_000), [lambda metrics: None])
    agent = SimpleNamespace(
        spaces={},
        stream=iter,
        init_train=lambda batch: None,
        init_report=lambda batch: None,
        init_policy=lambda batch: None,
        save=lambda: {"parameters": 42},
        load=lambda state: None,
    )
    replay = SimpleNamespace(
        add=lambda *args: None, save=lambda: {}, load=lambda state: None
    )
    monkeypatch.setattr(
        train.embodied,
        "Driver",
        lambda *args, **kwargs: SimpleNamespace(
            on_step=lambda callback: None,
            reset=lambda callback: None,
        ),
    )
    artifacts = []

    def record(logdir, kind, paths):
        artifacts.append(kind)
        (checkpoint,) = paths
        assert (checkpoint / "done").exists()
        assert (checkpoint / "agent.pkl").exists()

    monkeypatch.setattr(main, "record_campaign_artifact", record)
    train.train(
        lambda: agent,
        lambda: replay,
        lambda index: None,
        lambda replay, mode: [],
        lambda: logger,
        args,
    )
    assert artifacts == ["checkpoint"]
