import hashlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from majepa import main


def test_normal_logger_records_complete_config(monkeypatch, tmp_path):
    config = main._resolve_config_profiles(main._load_configs(), ["defaults"])
    config = config.update(logdir=str(tmp_path), **{"logger.outputs": ["wandb"]})
    monkeypatch.delenv("MAJEPA_CAMPAIGN_MANIFEST", raising=False)
    output = MagicMock()
    monkeypatch.setattr(main.elements.logger, "WandBOutput", output)
    main.make_logger(config)
    assert output.call_args.kwargs["config"] == dict(config)


def fake_wandb(monkeypatch, *, state="COMMITTED", mismatch=False):
    sdk = MagicMock()
    sdk.run.id, sdk.run.entity, sdk.run.project = "run", "entity", "project"
    sdk.run.summary = {}
    artifact = sdk.Artifact.return_value
    entries = {"weights": SimpleNamespace(digest="file-digest")}
    artifact.manifest.entries = entries
    uploaded = sdk.run.log_artifact.return_value.wait.return_value
    uploaded.name, uploaded.digest = "run-checkpoint:v0", "digest"
    remote = sdk.Api.return_value.artifact.return_value
    remote.state, remote.digest = state, "wrong" if mismatch else "digest"
    remote.manifest.entries = entries
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    return sdk


@pytest.mark.parametrize("state,mismatch", [("PENDING", False), ("COMMITTED", True)])
def test_artifact_verification_fails_closed(monkeypatch, tmp_path, state, mismatch):
    from majepa.tracking import upload_verified_artifact

    sdk = fake_wandb(monkeypatch, state=state, mismatch=mismatch)
    path = tmp_path / "weights"
    path.write_bytes(b"weights")
    with pytest.raises(RuntimeError, match="verification failed"):
        upload_verified_artifact("checkpoint", [path])
    assert sdk.run.summary["artifacts/checkpoint_verified"] is False
    sdk.run.finish.assert_not_called()


def test_artifact_upload_preserves_active_training_run(monkeypatch, tmp_path):
    from majepa.tracking import upload_verified_artifact

    sdk = fake_wandb(monkeypatch)
    path = tmp_path / "weights"
    path.write_bytes(b"weights")
    receipt = upload_verified_artifact("checkpoint", [path])
    assert receipt == {
        "artifact": "entity/project/run-checkpoint:v0",
        "digest": "digest",
        "files": ["weights"],
        "verified": True,
    }
    assert sdk.run.summary["artifacts/checkpoint_verified"] is True
    sdk.run.finish.assert_not_called()


def test_manifest_mismatch_and_empty_artifacts_are_rejected(monkeypatch, tmp_path):
    from majepa.tracking import upload_verified_artifact

    sdk = fake_wandb(monkeypatch)
    sdk.Api.return_value.artifact.return_value.manifest.entries = {
        "weights": SimpleNamespace(digest="corrupted")
    }
    with pytest.raises(RuntimeError, match="verification failed"):
        upload_verified_artifact("checkpoint", [tmp_path])
    sdk.Artifact.return_value.manifest.entries = {}
    with pytest.raises(ValueError, match="empty artifact"):
        upload_verified_artifact("checkpoint", [tmp_path])
    assert sdk.run.summary["artifacts/checkpoint_verified"] is False


def test_publisher_resumes_and_finishes_only_its_run(monkeypatch, tmp_path):
    from majepa.tracking import publish_run_artifact

    sdk = fake_wandb(monkeypatch)
    run = sdk.run
    with pytest.raises(RuntimeError, match="active"):
        publish_run_artifact("entity", "project", "run", "checkpoint", [tmp_path])
    run.finish.assert_not_called()
    sdk.run = None

    def init(**kwargs):
        sdk.run = run
        return run

    sdk.init.side_effect = init
    receipt = publish_run_artifact("entity", "project", "run", "checkpoint", [tmp_path])
    assert receipt["verified"] is True
    assert sdk.init.call_args.kwargs["resume"] == "must"
    run.finish.assert_called_once_with(exit_code=0)


def test_publisher_finishes_with_failure_on_verification_error(monkeypatch, tmp_path):
    from majepa.tracking import publish_run_artifact

    sdk = fake_wandb(monkeypatch, mismatch=True)
    run = sdk.run
    sdk.run = None

    def init(**kwargs):
        sdk.run = run
        return run

    sdk.init.side_effect = init
    with pytest.raises(RuntimeError, match="verification failed"):
        publish_run_artifact("entity", "project", "run", "checkpoint", [tmp_path])
    run.finish.assert_called_once_with(exit_code=1)


@pytest.mark.parametrize(
    "campaign,failed", [(False, False), (True, False), (True, True)]
)
def test_main_finishes_campaign_only_after_training(
    monkeypatch, tmp_path, campaign, failed
):
    import majepa

    sdk = fake_wandb(monkeypatch)
    monkeypatch.setitem(
        sys.modules,
        "majepa.marl.core",
        SimpleNamespace(MARLCore=SimpleNamespace(banner=[])),
    )
    monkeypatch.setattr(main.portal, "setup", lambda **kwargs: None)
    if campaign:
        monkeypatch.setenv("MAJEPA_CAMPAIGN_MANIFEST", str(tmp_path / "job.json"))
    else:
        monkeypatch.delenv("MAJEPA_CAMPAIGN_MANIFEST", raising=False)

    def train(*args):
        sdk.run.finish.assert_not_called()
        if failed:
            raise RuntimeError("training failed")

    monkeypatch.setattr(majepa, "train", SimpleNamespace(train=train), raising=False)
    args = [
        "--configs",
        "defaults",
        "--logdir",
        str(tmp_path),
        "--agent.num_agents",
        "2",
    ]
    if failed:
        with pytest.raises(RuntimeError, match="training failed"):
            main.main(args)
    else:
        main.main(args)
    if campaign:
        sdk.run.finish.assert_called_once_with(exit_code=int(failed))
    else:
        sdk.run.finish.assert_not_called()


def test_campaign_source_hash_and_start_artifacts(monkeypatch, tmp_path):
    from majepa import tracking

    config = main._resolve_config_profiles(main._load_configs(), ["defaults"])
    config = config.update(
        logdir=str(tmp_path),
        **{
            "logger.outputs": ["wandb"],
            "run.final_save": True,
        },
    )
    config.save(main.elements.Path(tmp_path / "config.yaml"))
    source = tmp_path / "source.tar.gz"
    source.write_bytes(b"frozen source")
    manifest = tmp_path / "job.json"
    provenance = {
        "campaign": "test",
        "source_sha256": "wrong",
        "base_commit": "a" * 40,
        "dreamerv3_revision": "b" * 40,
        "changed_source_files": ["src/majepa/main.py"],
        "deleted_source_paths": ["old.py"],
        "job": {"pod_id": "pod123", "name": "pod0"},
    }
    manifest_data = {
        **provenance,
        "job": {**provenance["job"], "run_names": ["run1", "run2"]},
        "runs": [{"name": "run2", "config": {"other_config": True}}],
        "jobs": [{"name": "other-pod"}],
    }
    manifest.write_text(json.dumps(manifest_data))
    monkeypatch.setenv("MAJEPA_CAMPAIGN_MANIFEST", str(manifest))
    monkeypatch.setenv("MAJEPA_SOURCE_ARCHIVE", str(source))
    monkeypatch.setenv("WANDB_MODE", "online")
    output = MagicMock()
    monkeypatch.setattr(main.elements.logger, "WandBOutput", output)
    with pytest.raises(ValueError, match="source.*hash"):
        main.make_logger(config)
    output.assert_not_called()
    provenance["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_data["source_sha256"] = provenance["source_sha256"]
    manifest.write_text(json.dumps(manifest_data))
    upload = MagicMock(return_value={"verified": True})
    monkeypatch.setattr(tracking, "upload_verified_artifact", upload)
    main.make_logger(config)
    assert output.call_args.kwargs["config"]["campaign_provenance"] == provenance
    for call in upload.call_args_list:
        assert call.kwargs["metadata"] == provenance
    assert {call.args[0] for call in upload.call_args_list} == {"source", "config"}
    receipts = json.loads((tmp_path / "artifact_verification.json").read_text())
    assert receipts == {"source": {"verified": True}, "config": {"verified": True}}
