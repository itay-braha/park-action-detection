"""Small helpers shared by the tools (no torch / upstream imports here, so tools start fast)."""
import json
import re
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".ts", ".mts", ".mpg", ".mpeg", ".wmv", ".asf", ".dav", ".h264", ".264"}


def video_id_from_path(path, root=None):
    """Stable, filesystem- and evaluator-safe id: relative path without extension, '/' -> '__'.

    The AVA evaluator splits lines on ' [' and keys on ',', so ids may not contain spaces or commas.
    """
    p = Path(path)
    rel = p.relative_to(root) if root else Path(p.name)
    stem = str(rel.with_suffix("")).replace("\\", "/")
    stem = stem.replace("/", "__")
    return re.sub(r"[^0-9A-Za-z_\-.]+", "_", stem).strip("_")


def read_meta(frames_dir, vid):
    with open(Path(frames_dir) / vid / "meta.json", "r", encoding="utf-8") as f:
        return json.load(f)


def read_pbtxt(path):
    """Minimal .pbtxt label-map reader -> {id: name}. Accepts `id:` or `label_id:`."""
    text = Path(path).read_text(encoding="utf-8")
    items = re.findall(r'name:\s*"([^"]+)"\s*\n\s*(?:label_)?id:\s*(\d+)', text)
    return {int(i): n for n, i in items}
