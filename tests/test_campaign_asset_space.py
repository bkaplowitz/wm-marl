from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from unittest.mock import patch

from majepa import campaign_assets


def test_staging_checks_space_before_downloading_on_the_asset_disk():
    with TemporaryDirectory() as directory:
        root = Path(directory) / "assets"
        with (
            patch("shutil.disk_usage", return_value=shutil._ntuple_diskusage(50e9, 11e9, 39e9)),
            patch.object(campaign_assets.subprocess, "run") as download,
        ):
            try:
                campaign_assets.stage(root)
            except OSError as error:
                assert "free" in str(error) and "40" in str(error)
            else:
                raise AssertionError("insufficient staging space was accepted")
            download.assert_not_called()
            assert not root.exists()

        with (
            patch("shutil.disk_usage", return_value=shutil._ntuple_diskusage(50e9, 10e9, 40e9)),
            patch.object(campaign_assets.subprocess, "run", side_effect=InterruptedError("stop before download")) as download,
        ):
            try:
                campaign_assets.stage(root)
            except InterruptedError:
                pass
            else:
                raise AssertionError("sufficient staging space did not reach download")
            command = download.call_args.args[0]
            assert command[command.index("--output") + 1] == str(root / "SC2.4.10.zip")


if __name__ == "__main__":
    test_staging_checks_space_before_downloading_on_the_asset_disk()
    print("Asset disk-space check passed (no downloads or cloud operations).")
