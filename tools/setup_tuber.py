"""Fetch the official TubeR code, patch it for modern PyTorch/NumPy/Windows, and download weights.

Usage:
    python tools/setup_tuber.py                         # code + assets only
    python tools/setup_tuber.py --weights csn152-ava22  # + recommended fine-tuning checkpoint
    python tools/setup_tuber.py --weights all           # everything (~3.7 GB)

The upstream repo is pinned to a known commit and downloaded as a zip (no git needed).
It lands in third_party/tubelet-transformer and is imported by `parktuber.upstream`.
We never edit upstream by hand: every change is a patch listed in PATCHES below, so the
whole setup is reproducible and you can read exactly what we changed and why.
"""
import argparse
import io
import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party"
UPSTREAM_DIR = THIRD_PARTY / "tubelet-transformer"
ASSETS_DIR = THIRD_PARTY / "assets"
WEIGHTS_DIR = ROOT / "weights"

UPSTREAM_COMMIT = "f610c97251e5539256095508570563ca2dc8c7a1"  # 2022-08-25, latest on main
UPSTREAM_ZIP = f"https://github.com/amazon-science/tubelet-transformer/archive/{UPSTREAM_COMMIT}.zip"

S3 = "https://yzaws-data-log.s3.amazonaws.com/shared/TubeR_cvpr22/"
WEIGHTS = {
    # full TubeR checkpoints (backbone + transformer + heads) trained on AVA
    "csn152-ava22": ("TubeR_CSN152_AVA22.pth", "TubeR CSN-152, AVA v2.2, 31.1 mAP (recommended start point)"),
    "csn152-ava21": ("TubeR_CSN152_AVA21.pth", "TubeR CSN-152, AVA v2.1, 29.7 mAP"),
    "csn50-ava21": ("TubeR_CSN50_AVA21.pth", "TubeR CSN-50, AVA v2.1, 27.2 mAP (lighter backbone)"),
    "csn152-jhmdb": ("TubeR_CSN152_JHMDB.pth", "TubeR CSN-152, JHMDB (single-label, not used by our pipeline)"),
    # pieces for training from scratch (backbone from Kinetics/IG65M, transformer from DETR)
    "backbone-csn152": ("irCSN_152_ft_kinetics_from_ig65m_f126851907.mat", "CSN-152 backbone only (Kinetics+IG65M)"),
    "backbone-csn50": ("irCSN_50_ft_kinetics_from_ig65m_f233743920.mat", "CSN-50 backbone only (Kinetics+IG65M)"),
    "detr": ("detr.pth", "DETR transformer weights (COCO)"),
}

# (file, regex, replacement, reason)
PATCHES = [
    ("datasets/ava_frame.py", r"np\.int\(", "int(", "np.int was removed in NumPy 1.24"),
    ("datasets/jhmdb_frame.py", r"np\.int\(", "int(", "np.int was removed in NumPy 1.24"),
    ("datasets/ava_frame.py", r"video_frame_path = self\.frame_path\n",
     "video_frame_path = self.frame_path.format(vid)\n",
     "upstream bug (GitHub issue #21): DATA_PATH '{}' placeholder was never filled with the video id"),
    ("evaluates/utils/metrics.py", r"np\.bool\b(?!_)", "np.bool_", "np.bool was removed in NumPy 1.24"),
    ("evaluates/utils/metrics.py", r"np\.float\b(?![_0-9])", "np.float64", "np.float was removed in NumPy 1.24"),
    ("utils/misc.py", r"float\(torchvision\.__version__\[:3\]\) < 0\.7", "False",
     "'0.2x'[:3] parses as 0.2 and would take a dead code path on modern torchvision"),
    ("models/detr/util/misc.py", r"float\(torchvision\.__version__\[:3\]\) < 0\.7", "False", "same as above"),
    ("models/transformer/util/misc.py", r"float\(torchvision\.__version__\[:3\]\) < 0\.7", "False", "same as above"),
    ("utils/model_utils.py", r"torch\.load\(((?:(?!weights_only)[^)])*)\)",r"torch.load(\1, weights_only=False)",
     "PyTorch >= 2.6 defaults to weights_only=True; TubeR checkpoints pickle a yacs config"),
    ("evaluates/evaluate_ava.py", r'open\("/xxx/datasets/ava_val_excluded_timestamps_v2\.1\.csv"\)',
     'open(__import__("os").path.join(__import__("os").path.dirname(self.label_path), '
     '"ava_val_excluded_timestamps_v2.1.csv"))',
     "hard-coded placeholder path; look next to the label map instead (only used for 80-class AVA eval)"),
    ("evaluates/utils/object_detection_evaluation.py", r"\n(\s*)import matplotlib\.pyplot as plt\n",
     r"\n\1plt = None  # (patched) matplotlib import only served disabled plotting code\n",
     "avoid a hard matplotlib dependency for code inside `if 0:` blocks"),
]


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name} already exists")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("content-length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def fetch_upstream(force: bool) -> None:
    if UPSTREAM_DIR.exists() and not force:
        print(f"[upstream] already present at {UPSTREAM_DIR} (use --force to re-download)")
        return
    if UPSTREAM_DIR.exists():
        shutil.rmtree(UPSTREAM_DIR)
    print(f"[upstream] downloading commit {UPSTREAM_COMMIT[:8]}")
    with urllib.request.urlopen(UPSTREAM_ZIP) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(THIRD_PARTY)
    (THIRD_PARTY / f"tubelet-transformer-{UPSTREAM_COMMIT}").rename(UPSTREAM_DIR)


def apply_patches() -> None:
    print("[patch] applying compatibility patches")
    for rel, pattern, repl, reason in PATCHES:
        path = UPSTREAM_DIR / rel
        text = path.read_text(encoding="utf-8")
        new, n = re.subn(pattern, repl, text)
        if n:
            path.write_text(new, encoding="utf-8")
            print(f"  {rel}: {n} change(s) - {reason}")
        else:
            print(f"  {rel}: nothing to change (already patched?)")
    (UPSTREAM_DIR / "PATCHED_BY_PARKTUBER.txt").write_text(
        "This copy was patched by tools/setup_tuber.py. See PATCHES in that file.\n", encoding="utf-8")


def fetch_assets() -> None:
    """AVA label maps + annotation JSONs (27 MB). We need the label maps for zero-shot AVA inference."""
    if (ASSETS_DIR / "ava_action_list_v2.1.pbtxt").exists():
        print("[assets] already present")
        return
    print("[assets] downloading AVA label maps / annotation files")
    with urllib.request.urlopen(S3 + "assets.zip") as r:
        data = r.read()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            if info.filename.startswith("assets/") and not info.is_dir():
                (ASSETS_DIR / Path(info.filename).name).write_bytes(z.read(info))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", nargs="*", default=[],
                    help=f"weights to download: {', '.join(WEIGHTS)} or 'all'")
    ap.add_argument("--force", action="store_true", help="re-download and re-patch the upstream code")
    ap.add_argument("--list", action="store_true", help="list available weights and exit")
    args = ap.parse_args()

    if args.list:
        for k, (fname, desc) in WEIGHTS.items():
            print(f"  {k:16s} {fname:52s} {desc}")
        return

    fetch_upstream(args.force)
    apply_patches()
    fetch_assets()

    names = list(WEIGHTS) if "all" in args.weights else args.weights
    for name in names:
        if name not in WEIGHTS:
            sys.exit(f"unknown weights '{name}'. Choose from: {', '.join(WEIGHTS)}")
        fname, desc = WEIGHTS[name]
        print(f"[weights] {name}: {desc}")
        download(S3 + fname, WEIGHTS_DIR / fname)
    print("done.")


if __name__ == "__main__":
    main()
