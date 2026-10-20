"""Video helpers for run.py: probing, non-destructive excerpts and chunking (the source file is only ever READ)."""
import shutil
import subprocess
from pathlib import Path

import cv2


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg            # pip wheel that ships an ffmpeg binary (no system install needed)
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("ffmpeg not found: install it (apt/conda) or `pip install imageio-ffmpeg`")


def probe(path):
    """fps, frame count, duration (s), width, height - via OpenCV, which needs no extra tools."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "frames": n, "duration": n / fps if fps else 0.0, "width": w, "height": h}


def parse_time(t):
    """'01:30:00' / '90:00' / '5400' -> seconds."""
    if t is None or t == "":
        return None
    parts = [float(p) for p in str(t).split(":")]
    s = 0.0
    for p in parts:
        s = s * 60 + p
    return s


def _ffmpeg(args):
    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def make_excerpt(src, dst, start=None, duration=None):
    """Copy a time range of `src` into `dst` without re-encoding. The source is never modified."""
    dst = Path(dst)
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    args = []
    if start:
        args += ["-ss", str(start)]
    args += ["-i", str(src)]
    if duration:
        args += ["-t", str(duration)]
    _ffmpeg(args + ["-map", "0:v:0", "-c", "copy", str(dst)])
    return dst


def split(src, chunk_dir, chunk_sec):
    """Cut `src` into ~chunk_sec pieces (stream copy: fast, cuts on key frames). Returns [(path, start_sec)].

    Start times are measured from the pieces themselves, since key-frame cuts make chunk lengths uneven.
    """
    chunk_dir = Path(chunk_dir)
    ext = Path(src).suffix.lower() if Path(src).suffix.lower() in {".mp4", ".mkv", ".avi", ".mov"} else ".mkv"
    chunks = sorted(chunk_dir.glob(f"chunk_*{ext}"))
    if not chunks:
        chunk_dir.mkdir(parents=True, exist_ok=True)
        _ffmpeg(["-i", str(src), "-map", "0:v:0", "-c", "copy", "-f", "segment", "-segment_time", str(chunk_sec),
                 "-reset_timestamps", "1", str(chunk_dir / f"chunk_%04d{ext}")])
        chunks = sorted(chunk_dir.glob(f"chunk_*{ext}"))
    out, t = [], 0.0
    for c in chunks:
        out.append((c, t))
        t += probe(c)["duration"]
    return out
