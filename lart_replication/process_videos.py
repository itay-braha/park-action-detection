"""Run LART on hours of footage: chunk -> both GPUs in parallel -> resumable -> one CSV per video.   (Linux)

    python process_videos.py /data/park_videos --out outputs/park --gpus 0,1
    python process_videos.py cam03.mp4 --out outputs/test --chunk-min 1 --max-chunks 1      # 1-minute timing test
    python process_videos.py /data/park_videos --out outputs/park --detector maskrcnn       # faster, fewer small people
    python process_videos.py --export-only --out outputs/park                               # just rebuild the CSVs

Why chunks:
  * PHALP stops after frame 1300 (~43 s) by default (VideoConfig.end_frame) - we pass video.end_frame=-1 per chunk;
  * PHALP first writes EVERY frame as a JPEG and keeps a whole run's results in RAM - chunks bound disk and memory
    (frames of finished chunks are deleted unless --keep-frames);
  * a crash or reboot costs one chunk, not the whole night: finished chunks have a DONE file and are skipped.

Per chunk this runs run_demo.py (ViTDet-H or Mask R-CNN detection -> HMR 2.0 3D bodies -> PHALP tracking ->
MViT appearance -> LART), one chunk per GPU at a time. Afterwards export_csv() writes <out>/<video>_actions.csv:
    video, chunk, time_sec, frame, track_id, x1, y1, x2, y2, action_1, p_1, action_2, p_2, action_3, p_3
Track ids are per chunk (PHALP restarts at every chunk).
"""
import argparse
import csv
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".ts", ".m4v", ".mpg", ".mpeg", ".dav", ".h264"}


def probe_fps(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()
    num, den = (out.split("/") + ["1"])[:2]
    return float(num) / float(den or 1) if num else 30.0


def split_video(src, chunk_dir, chunk_min):
    """Cut without re-encoding (fast; cuts land on key frames, so chunk lengths vary by a few seconds)."""
    chunk_dir.mkdir(parents=True, exist_ok=True)
    if not any(chunk_dir.glob("*.mp4")):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(src), "-map", "0:v:0", "-c", "copy", "-f", "segment",
                        "-segment_time", str(int(chunk_min * 60)), "-reset_timestamps", "1",
                        str(chunk_dir / "chunk_%04d.mp4")], check=True)
    return sorted(chunk_dir.glob("chunk_*.mp4"))


def chunk_start_times(chunks):
    """Real start time of every chunk (key-frame cuts make them uneven)."""
    t, starts = 0.0, []
    for c in chunks:
        starts.append(t)
        d = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(c)],
                           capture_output=True, text=True).stdout.strip()
        t += float(d or 0)
    return starts


def run_chunk(chunk, out_dir, gpu, args):
    done = out_dir / "DONE.json"
    if done.exists():
        return json.loads(done.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYOPENGL_PLATFORM=os.environ.get("PYOPENGL_PLATFORM", "egl"),
               LART_RENDER="1" if args.render else "0")
    cmd = [sys.executable, str(ROOT / "run_demo.py"), f"video.source={chunk}", f"video.output_dir={out_dir}",
           "video.end_frame=-1", f"phalp.detector={args.detector}", f"hydra.run.dir={out_dir / 'hydra'}"]
    t0 = time.time()
    with open(out_dir / "log.txt", "w") as log:
        rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    secs = time.time() - t0
    if rc != 0:
        return {"chunk": str(chunk), "error": f"exit code {rc}, see {out_dir / 'log.txt'}"}
    frames_dir = out_dir / "_DEMO"
    n_frames = sum(len(list(p.glob("*.jpg"))) for p in frames_dir.glob("*/img")) if frames_dir.exists() else 0
    if not args.keep_frames and frames_dir.exists():
        shutil.rmtree(frames_dir, ignore_errors=True)
    info = {"chunk": str(chunk), "gpu": gpu, "seconds": round(secs, 1), "frames": n_frames,
            "fps_processed": round(n_frames / secs, 2) if secs else None}
    done.write_text(json.dumps(info))
    return info


def export_csv(out_root, video_name, fps, starts):
    """Flatten results_temporal/*.pkl (per frame: boxes + LART's 60 action logits per track) into one CSV."""
    import joblib
    import numpy as np
    sys.path.insert(0, str(ROOT))
    import lartlab  # noqa: F401  (cache paths)
    from phalp.configs.base import CACHE_DIR
    names = joblib.load(f"{CACHE_DIR}/phalp/ava/ava_labels.pkl")[0]          # {ava_id: name}
    model_to_ava = joblib.load(f"{CACHE_DIR}/phalp/ava/ava_class_mapping.pkl")  # {1..60: ava_id}
    rows = []
    vdir = out_root / video_name
    for ci, cdir in enumerate(sorted(p for p in vdir.glob("chunk_*") if p.is_dir())):
        for pkl in sorted((cdir / "results_temporal").glob("*.pkl")):
            d = joblib.load(pkl)
            for fi, fkey in enumerate(sorted(d)):
                fr = d[fkey]
                acts = fr.get("ava_action", {})
                for k, tid in enumerate(fr.get("tid", [])):
                    if tid not in acts:
                        continue
                    p = 1 / (1 + np.exp(-np.asarray(acts[tid], dtype=float).reshape(-1)))     # 60 sigmoid scores
                    top = np.argsort(-p)[:3]
                    x, y, w, h = [float(v) for v in fr["bbox"][k]]
                    row = [video_name, cdir.name, round(starts[ci] + fi / fps, 3), fi, tid,
                           round(x), round(y), round(x + w), round(y + h)]
                    for j in top:
                        row += [names.get(model_to_ava[int(j) + 1], str(j)), round(float(p[j]), 3)]
                    rows.append(row)
    out = out_root / f"{video_name}_actions.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "chunk", "time_sec", "frame", "track_id", "x1", "y1", "x2", "y2",
                    "action_1", "p_1", "action_2", "p_2", "action_3", "p_3"])
        w.writerows(rows)
    return out, len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="video files or folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--chunk-min", type=float, default=5.0, help="chunk length in minutes")
    ap.add_argument("--detector", default="vitdet", choices=["vitdet", "maskrcnn"],
                    help="vitdet = ViTDet-H as in the paper (accurate, slow); maskrcnn = faster, misses small people")
    ap.add_argument("--render", action="store_true", help="also render the 3D-mesh video per chunk (slow)")
    ap.add_argument("--keep-frames", action="store_true", help="keep PHALP's extracted JPEG frames")
    ap.add_argument("--max-chunks", type=int, default=0, help="process only the first N chunks per video (tests)")
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args()
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found (sudo apt install ffmpeg, or conda install -c conda-forge ffmpeg)")

    out_root = Path(args.out)
    videos = []
    for s in args.inputs:
        p = Path(s)
        videos += sorted(x for x in p.rglob("*") if x.suffix.lower() in VIDEO_EXTS) if p.is_dir() else [p]

    plan = []   # (video, chunk, chunk_out_dir)
    meta = {}
    for v in videos:
        name = v.stem.replace(" ", "_")
        chunks = split_video(v, out_root / name / "_chunks", args.chunk_min)
        if args.max_chunks:
            chunks = chunks[: args.max_chunks]
        meta[name] = {"fps": probe_fps(v), "starts": chunk_start_times(chunks), "source": str(v)}
        plan += [(name, c, out_root / name / c.stem) for c in chunks]
        print(f"{name}: {len(chunks)} chunk(s) of ~{args.chunk_min} min, {meta[name]['fps']:.2f} fps")
    out_root.mkdir(parents=True, exist_ok=True)
    if meta:
        (out_root / "videos.json").write_text(json.dumps(meta, indent=2))
    else:
        meta = json.loads((out_root / "videos.json").read_text())

    if not args.export_only and plan:
        jobs = queue.Queue()
        for item in plan:
            jobs.put(item)
        results, lock = [], threading.Lock()

        def worker(gpu):
            while True:
                try:
                    name, chunk, cdir = jobs.get_nowait()
                except queue.Empty:
                    return
                info = run_chunk(chunk, cdir, gpu, args)
                with lock:
                    results.append(info)
                    msg = info.get("error") or f"{info['frames']} frames in {info['seconds']:.0f}s = {info['fps_processed']} fps"
                    print(f"[gpu {gpu}] {name}/{chunk.stem}: {msg}   ({len(results)}/{len(plan)})", flush=True)

        threads = [threading.Thread(target=worker, args=(g.strip(),)) for g in args.gpus.split(",")]
        t0 = time.time()
        [t.start() for t in threads]
        [t.join() for t in threads]
        ok = [r for r in results if "error" not in r]
        if ok:
            fps = sum(r["frames"] for r in ok) / max(1e-6, sum(r["seconds"] for r in ok))
            print(f"\n{len(ok)}/{len(results)} chunks OK in {(time.time() - t0) / 60:.1f} min; "
                  f"per-GPU speed {fps:.2f} frames/s -> 1 h of 30-fps video ≈ {30 * 3600 / max(fps, 1e-6) / 3600:.1f} GPU-hours")

    for name, m in meta.items():
        path, n = export_csv(out_root, name, m["fps"], m["starts"])
        print(f"{path}: {n} rows")


if __name__ == "__main__":
    main()
