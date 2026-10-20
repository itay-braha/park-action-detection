"""Run TubeR or LART on a (very long) video and get per-person action detections + a per-minute timeline.

    python run.py --model tuber --video /data/park/cam03_4h.mp4 --out results/cam03_tuber --gpus 0,1
    python run.py --model lart  --video /data/park/cam03_4h.mp4 --out results/cam03_lart  --gpus 0,1
    # test on a piece first (the original file is never modified - the excerpt is copied into --out):
    python run.py --model lart --video /data/park/cam03_4h.mp4 --out results/test --start 01:30:00 --duration 120

What it does
  1. (optional) copies the [--start, --start+--duration] range of the video into <out>/_input/ (stream copy).
  2. cuts the video into chunks of --chunk-min minutes (stream copy, into <out>/_chunks/).
  3. processes the chunks in parallel, one per GPU (--gpus 0,1). Every finished chunk writes DONE.json, so if the run
     is interrupted (reboot, ssh drop, Ctrl-C) just run the same command again: finished chunks are skipped.
  4. merges everything into
       <out>/detections.csv  one row per person per sampled moment (every --sample-sec), with box and top-k actions
       <out>/timeline.csv    per minute: average number of people, and of people doing each action (prob >= --threshold)
       <out>/run_info.json   settings, per-chunk timing, speed

Models (both currently predict the 60 evaluated AVA actions: walk, sit, stand, ride, push an object, talk to ...)
  tuber  one network on 32-frame clips (fast; Windows or Linux). Default: the released AVA 2.2 model;
         use --tuber-config/--tuber-checkpoint for a fine-tuned park model.
  lart   detection -> 3D bodies -> tracking -> appearance -> LART transformer (slow; Linux; see lart_replication/).
"""
import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from runner import backends as B, outputs as O, video as V  # noqa: E402


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(B.BACKENDS))
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0", help="comma-separated GPU ids; one chunk runs on each at a time")
    ap.add_argument("--start", default=None, help="only process from this time (HH:MM:SS or seconds)")
    ap.add_argument("--duration", default=None, help="only process this many seconds (or HH:MM:SS)")
    ap.add_argument("--chunk-min", type=float, default=None, help="chunk length in minutes (tuber 10, lart 5)")
    ap.add_argument("--sample-sec", type=float, default=0.5, help="one prediction per person every N seconds")
    ap.add_argument("--top-k", type=int, default=3, help="actions kept per person in detections.csv")
    ap.add_argument("--threshold", type=float, default=0.5, help="action probability counted in timeline.csv")
    ap.add_argument("--render", action="store_true", help="also write annotated videos per chunk (slower)")
    ap.add_argument("--tuber-config", default=str(ROOT / "configs" / "ava_zero_shot.yaml"))
    ap.add_argument("--tuber-checkpoint", default="", help='"" = released AVA model (FINETUNE.INIT_FROM)')
    ap.add_argument("--lart-detector", default="vitdet", choices=["vitdet", "maskrcnn"],
                    help="vitdet = as in the paper (accurate, slow); maskrcnn = faster, misses small people")
    ap.add_argument("--keep-frames", action="store_true", help="LART: keep the JPEG frames PHALP extracts")
    ap.add_argument("--python", default=sys.executable, help="interpreter for the model subprocesses")
    ap.add_argument("--merge-only", action="store_true", help="only rebuild the CSVs from finished chunks")
    ap.add_argument("--cleanup", action="store_true",
                    help="delete <out>/_chunks and <out>/_input when every chunk is done (they are copies of the video)")
    opts = ap.parse_args()

    backend = B.BACKENDS[opts.model](opts)
    out = Path(opts.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    src = Path(opts.video).resolve()
    if out in src.parents or src.parent == out:
        sys.exit("--out must not contain the source video")

    # 1) optional excerpt (never touches the source)
    start, dur = V.parse_time(opts.start), V.parse_time(opts.duration)
    work_video = src
    if start or dur:
        work_video = V.make_excerpt(src, out / "_input" / f"excerpt_{int(start or 0)}s_{int(dur or 0)}s{src.suffix}",
                                    start, dur)
    info = V.probe(work_video)
    print(f"video: {work_video.name}  {info['duration'] / 60:.1f} min, {info['fps']:.2f} fps, "
          f"{info['width']}x{info['height']}" + (f"  (excerpt of {src.name} from {opts.start})" if start else ""))

    # 2) chunks
    chunk_min = opts.chunk_min or backend.default_chunk_min
    chunks = V.split(work_video, out / "_chunks", chunk_min * 60)
    print(f"{len(chunks)} chunk(s) of ~{chunk_min:g} min -> model '{opts.model}' on GPU(s) {opts.gpus}")

    # 3) process, one worker thread per GPU
    results = []
    if not opts.merge_only:
        jobs = queue.Queue()
        for c in chunks:
            jobs.put(c)
        lock = threading.Lock()
        t0 = time.time()

        def worker(gpu):
            while True:
                try:
                    chunk, cstart = jobs.get_nowait()
                except queue.Empty:
                    return
                r = B.process_chunk(backend, chunk, cstart, out / "chunks" / chunk.stem, gpu)
                with lock:
                    results.append(r)
                    done = len(results)
                    if "error" in r:
                        msg = r["error"]
                    else:
                        msg = ("already done" if r.get("cached") else
                               f"{r['duration_sec'] / 60:.1f} min of video in {r['seconds'] / 60:.1f} min "
                               f"({r['speed_x_realtime']}x real time)")
                    eta = (time.time() - t0) / done * (len(chunks) - done) / 60
                    print(f"[gpu {gpu}] {chunk.stem} @ {O.hms(cstart)}: {msg}   [{done}/{len(chunks)}, "
                          f"~{eta:.0f} min left]", flush=True)

        threads = [threading.Thread(target=worker, args=(g.strip(),)) for g in opts.gpus.split(",")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        failed = [r for r in results if "error" in r]
        if failed:
            print(f"\n{len(failed)} chunk(s) failed - fix the cause and re-run the same command to retry them:")
            for r in failed:
                print("  ", r["chunk"], r["error"])

    # 4) merge whatever is finished
    # times in the outputs are on the ORIGINAL video's clock (excerpts are cut on key frames: +- a few seconds)
    offset = start or 0.0
    rows, timing = [], []
    for chunk, cstart in chunks:
        cdir = out / "chunks" / chunk.stem
        done = B.chunk_done(cdir)
        if not done:
            continue
        timing.append(done)
        rows += [(chunk.stem, offset + cstart, p) for p in backend.read_chunk(cdir, V.probe(chunk))]
    O.write_detections(out / "detections.csv", rows, src.name, opts.model, opts.top_k)
    actions = O.write_timeline(out / "timeline.csv", rows, opts.threshold)
    secs = sum(t["seconds"] for t in timing)
    vid = sum(t["duration_sec"] for t in timing)
    run_info = {"model": opts.model, "video": str(src), "excerpt": {"start": opts.start, "duration": opts.duration},
                "chunks_done": len(timing), "chunks_total": len(chunks), "video_minutes_done": round(vid / 60, 1),
                "gpu_minutes": round(secs / 60, 1), "speed_x_realtime_per_gpu": round(vid / secs, 3) if secs else None,
                "settings": {k: v for k, v in vars(opts).items() if k not in ("python",)}, "git_commit": git_commit(),
                "chunks": timing}
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print(f"\n{len(timing)}/{len(chunks)} chunks merged: {len(rows)} person-moments -> {out / 'detections.csv'}")
    print(f"timeline ({len(actions)} actions above {opts.threshold}) -> {out / 'timeline.csv'}")
    if opts.cleanup and len(timing) == len(chunks):
        import shutil
        for d in ("_chunks", "_input"):
            shutil.rmtree(out / d, ignore_errors=True)
        print("removed the temporary video copies (_chunks, _input)")
    if secs:
        print(f"speed: {vid / secs:.2f}x real time per GPU  (4 h of video ~ {4 * 60 / (vid / secs) / 60:.1f} GPU-hours)")


if __name__ == "__main__":
    main()
