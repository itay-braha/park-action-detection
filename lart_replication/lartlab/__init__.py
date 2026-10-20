"""lartlab - helpers for replicating and studying LART (Rajasegaran et al., CVPR 2023).

Importing it configures the environment *before* PHALP/LART are imported:
  * HOME -> lart_replication/cache, so PHALP's CACHE_DIR ($HOME/.cache) points at our local, git-ignored
    asset folder (built by tools/setup_lart.py) on every OS;
  * the pinned upstream code in third_party/ goes on sys.path.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_HOME = ROOT / "cache"
THIRD = ROOT / "third_party"

os.environ["HOME"] = str(CACHE_HOME)
os.environ.setdefault("PROJECT_ROOT", str(THIRD / "LART"))
for p in (THIRD / "PHALP", THIRD / "LART", THIRD / "4D-Humans"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
if not (THIRD / "LART" / "lart").exists():
    raise RuntimeError("LART code missing - run: python tools/setup_lart.py")
