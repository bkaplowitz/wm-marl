import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path[:0] = [
    str(Path(__file__).parents[1] / "src"),
    str(Path(__file__).parents[1] / "external" / "dreamerv3"),
]
