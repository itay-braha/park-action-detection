"""Make the official TubeR code importable.

Upstream uses top-level package names (`models`, `utils`, `datasets`, `evaluates`, `pipelines`),
so we put its folder at the front of sys.path. That is also why *our* package is called
`parktuber` and not `models`/`utils` - the names would collide.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = ROOT / "third_party" / "tubelet-transformer"
ASSETS_DIR = ROOT / "third_party" / "assets"


def ensure_upstream_on_path() -> Path:
    if not (UPSTREAM_DIR / "models" / "tuber_ava.py").exists():
        raise RuntimeError(
            f"TubeR code not found at {UPSTREAM_DIR}.\n"
            "Run:  python tools/setup_tuber.py"
        )
    if not (UPSTREAM_DIR / "PATCHED_BY_PARKTUBER.txt").exists():
        raise RuntimeError("Upstream TubeR is not patched. Run:  python tools/setup_tuber.py")
    p = str(UPSTREAM_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    return UPSTREAM_DIR


ensure_upstream_on_path()
