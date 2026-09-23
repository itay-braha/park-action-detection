"""Decode videos into JPEG frames at a fixed frame rate (default 30 fps, like AVA).

    python tools/extract_frames.py --videos data/raw_videos --out data/frames
    python tools/extract_frames.py --videos cam3.mp4 --out data/frames --start 600 --end 900

Output per video:
    data/frames/<video_id>/000000.jpg, 000001.jpg, ...
    data/frames/<video_id>/meta.json   (source path, source fps, output fps, sizes, start offset)

Why re-sample to a fixed fps?  TubeR was trained on AVA clips of 32 frames taken every 2nd frame of
30 fps video (~2.1 s). Park cameras record at 25, 15, 12 fps or variable rate. Re-sampling on the
time axis keeps "2 seconds of motion" meaning the same thing for the pretrained backbone.
Frames are duplicated/dropped as needed - no interpolation.

Why shrink?  The model only sees ~256 px on the short side; storing 1080p JPEGs wastes disk and
makes data loading the bottleneck. Keep --short-side >= the IMG_SIZE you train with (a bit larger
is fine; bigger helps if you later raise IMG_SIZE for small/distant people).

Uses OpenCV only (pip wheel ships its own FFmpeg) - no system ffmpeg needed. Proprietary NVR
formats (e.g. Dahua .dav) may need converting to .mp4 first with the vendor player or ffmpeg.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parktuber.common import VIDEO_EXTS, video_id_from_path  # noqa: E402


def extract(video_path, out_root, fps, short_side, quality, start, end, root, overwrite):
    vid = video_id_from_path(video_path, root)
    out_dir = Path(out_root) / vid
    if (out_dir / "meta.json").exists() and not overwrite:
        return f"[skip] {vid} (already extracted)"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return f"[FAIL] cannot open {video_path}"
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if src_fps <= 1 or src_fps > 240:
        cap.release()
        return f"[FAIL] {video_path}: unreliable fps ({src_fps}); convert with ffmpeg -r first"
    ow, oh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if short_side and min(ow, oh) > short_side:
        s = short_side / min(ow, oh)
        nw, nh = int(round(ow * s / 2) * 2), int(round(oh * s / 2) * 2)
    else:
        nw, nh = ow, oh
    out_dir.mkdir(parents=True, exist_ok=True)
    i_src = 0                                # index of the next source frame
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start * src_fps)))
        i_src = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # where the seek actually landed
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    n_out = 0
    next_t = start                           # time of the next output frame
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_frame_end = (i_src + 1) / src_fps  # this source frame covers [i/fps, (i+1)/fps)
        if end is not None and i_src / src_fps >= end:
            break
        if next_t < t_frame_end:
            if (nw, nh) != (ow, oh):
                frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            while next_t < t_frame_end and (end is None or next_t < end):
                cv2.imwrite(str(out_dir / f"{n_out:06d}.jpg"), frame, params)
                n_out += 1
                next_t = start + n_out / fps
        i_src += 1
    cap.release()
    meta = {"video_id": vid, "source": str(Path(video_path).resolve()), "src_fps": src_fps, "fps": fps,
            "start_sec": start, "n_frames": n_out, "width": nw, "height": nh,
            "orig_width": ow, "orig_height": oh, "duration_sec": n_out / fps}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return f"[ok] {vid}: {n_out} frames ({n_out / fps:.1f}s) {ow}x{oh}@{src_fps:.2f} -> {nw}x{nh}@{fps}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, help="a video file or a folder (searched recursively)")
    ap.add_argument("--out", default="data/frames")
    ap.add_argument("--fps", type=float, default=30.0, help="output frame rate (must match DATA.CLIP_FPS)")
    ap.add_argument("--short-side", type=int, default=360, help="resize so the short side is this (0 = keep)")
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality")
    ap.add_argument("--start", type=float, default=0.0, help="seconds to skip at the start of each video")
    ap.add_argument("--end", type=float, default=None, help="stop at this second")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = Path(args.videos)
    if src.is_dir():
        videos = sorted(p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
        root = src
    else:
        videos, root = [src], None
    if not videos:
        sys.exit(f"no videos found under {src}")
    print(f"{len(videos)} video(s) -> {args.out} @ {args.fps} fps")
    jobs = [(v, args.out, args.fps, args.short_side, args.quality, args.start, args.end, root, args.overwrite)
            for v in videos]
    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            for msg in ex.map(extract, *zip(*jobs)):
                print(msg, flush=True)
    else:
        for j in jobs:
            print(extract(*j), flush=True)


if __name__ == "__main__":
    main()
