"""Build a reproducible Dreamer-CDP runtime containing the M3 delta."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

from world_marl.baselines.dreamer_cdp.config import default_upstream_root
from world_marl.baselines.dreamer_cdp.launcher import verify_upstream
from world_marl.jepa_transformer.foundation import repository_root


RSSM_DEFAULT = (
    "      rssm: {deter: 8192, hidden: 1024, stoch: 32, classes: 64, "
    "act: silu, norm: rms, unimix: 0.01, outscale: 1.0, "
    "winit: trunc_normal_in, imglayers: 2, obslayers: 1, dynlayers: 1, absolute: False, "
    "blocks: 8, free_nats: 1.0}"
)

M3_DYNAMICS_DEFAULT = (
    "      jepa_transformer: {deter: 8192, hidden: 1024, stoch: 32, "
    "classes: 64, act: silu, norm: rms, unroll: False, unimix: 0.01, "
    "outscale: 1.0, winit: trunc_normal_in, imglayers: 2, obslayers: 1, dynlayers: 1, "
    "absolute: False, blocks: 8, free_nats: 1.0, model: 512, layers: 4, "
    "heads: 8, context: 64, ffup: 4}"
)

M3_PROFILE = """

jepa_transformer:
  replay_context: 64
  jax.profiler: False
  agent:
    dyn:
      typ: jepa_transformer
"""


def overlay_path() -> Path:
    return Path(__file__).with_name("upstream") / "m3_rssm.py"


def runtime_fingerprint() -> str:
    payload = (
        overlay_path().read_bytes()
        + M3_DYNAMICS_DEFAULT.encode()
        + M3_PROFILE.encode()
    )
    return hashlib.sha256(payload).hexdigest()[:12]


def default_runtime_root() -> Path:
    return (
        repository_root()
        / ".runtime"
        / f"dreamer-cdp-jepa-transformer-{runtime_fingerprint()}"
    )


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one {label} integration point, found {count}")
    return text.replace(old, new)


def prepare_runtime(
    destination: str | Path | None = None,
    *,
    recreate: bool = False,
) -> Path:
    """Create a clean official snapshot and apply only the registered M3 delta."""
    source = default_upstream_root().resolve()
    revision = verify_upstream(source)
    destination = Path(destination or default_runtime_root()).resolve()
    marker = destination / ".jepa-transformer-runtime"
    expected = f"{revision}\n{runtime_fingerprint()}\n"
    if destination.exists():
        if not recreate and marker.is_file() and marker.read_text() == expected:
            return destination
        if not recreate:
            raise FileExistsError(f"runtime already exists: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    archive = subprocess.Popen(
        ["git", "archive", "HEAD"], cwd=source, stdout=subprocess.PIPE
    )
    assert archive.stdout is not None
    extract = subprocess.run(
        ["tar", "-x", "-C", str(destination)],
        stdin=archive.stdout,
        check=True,
    )
    del extract
    archive.stdout.close()
    if archive.wait() != 0:
        raise RuntimeError("failed to archive pinned Dreamer-CDP source")

    shutil.copy2(overlay_path(), destination / "dreamerv3" / "m3_rssm.py")
    agent_path = destination / "dreamerv3" / "agent.py"
    agent = agent_path.read_text(encoding="utf-8")
    agent = _replace_once(
        agent,
        "from . import rssm\n",
        "from . import m3_rssm, rssm\n",
        label="dynamics import",
    )
    agent = _replace_once(
        agent,
        "        'rssm': rssm.RSSM,\n",
        (
            "        'rssm': rssm.RSSM,\n"
            "        'jepa_transformer': m3_rssm.TransformerRSSM,\n"
        ),
        label="dynamics registry",
    )
    agent = _replace_once(
        agent,
        "          stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))",
        (
            "          stepid=stepid, enc=entries[0],\n"
            "          dyn={key: entries[1][key] for key in self.dyn.entry_space},\n"
            "          dec=entries[2]))"
        ),
        label="replay entry filter",
    )
    agent_path.write_text(agent, encoding="utf-8")
    config_path = destination / "dreamerv3" / "configs.yaml"
    config = config_path.read_text(encoding="utf-8")
    config = _replace_once(
        config,
        RSSM_DEFAULT,
        f"{RSSM_DEFAULT}\n{M3_DYNAMICS_DEFAULT}",
        label="dynamics configuration schema",
    )
    config = _replace_once(
        config,
        "    enable_policy: True\n",
        "    enable_policy: True\n    profiler: True\n",
        label="profiler configuration schema",
    )
    config_path.write_text(
        config + M3_PROFILE,
        encoding="utf-8",
    )
    marker.write_text(expected, encoding="utf-8")
    return destination
