"""Draw run.py's detections.csv onto the video: a box per person with its top actions (no GPU needed).

    python tools/render_detections.py --video real_data/out.mp4 --runs results/out_tuber --out results/out_tuber.mp4
    python tools/render_detections.py --video real_data/out.mp4 --runs results/out_tuber results/out_lart \
        --out results/out_both.mp4        # both models on the same frames, one color each

Predictions exist every 0.5 s; each one is shown until the next, so boxes move in 0.5 s steps.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2

COLORS = [(10, 113, 232), (232, 115, 26), (56, 128, 24), (244, 66, 161)]   # BGR: orange, blue, green, purple


def load(run_dir, min_score):
    by_t = defaultdict(list)
    with open(Path(run_dir) / "detections.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            acts = [(r[f"action_{i}"], float(r[f"score_{i}"])) for i in range(1, 10) if r.get(f"action_{i}")]
            acts = [a for a in acts if a[1] >= min_score] or acts[:1]
            by_t[round(float(r["time_sec"]) * 2) / 2].append(
                ([float(r[k]) for k in ("x1", "y1", "x2", "y2")], acts, r["model"]))
    return by_t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1280, help="output width (keeps the aspect ratio)")
    ap.add_argument("--min-score", type=float, default=0.3, help="actions shown under a box (top one always shown)")
    ap.add_argument("--start", type=float, default=0, help="seconds")
    ap.add_argument("--duration", type=float, default=0, help="seconds, 0 = to the end")
    opts = ap.parse_args()

    runs = [load(d, opts.min_score) for d in opts.runs]
    names = [next(iter(r.values()))[0][2] if r else Path(d).name for r, d in zip(runs, opts.runs)]
    cap = cv2.VideoCapture(opts.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W0, H0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W, H = opts.width, int(H0 * opts.width / W0) // 2 * 2
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = int(opts.start * fps)
    last = min(n_total, first + int(opts.duration * fps)) if opts.duration else n_total
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    Path(opts.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(opts.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    for fi in range(first, last):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (W, H))
        t = fi / fps
        key = int(t * 2) / 2                              # latest prediction at or before this frame
        for i, r in enumerate(runs):
            col = COLORS[i % len(COLORS)]
            for box, acts, _ in r.get(key, []):
                x1, y1, x2, y2 = [int(v * d) for v, d in zip(box, (W, H, W, H))]
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                lines = [f"{a} {s:.2f}" for a, s in acts]
                # first model labels above the box, second below, so they don't overlap
                ys = [y1 - 6 - 15 * k for k in range(len(lines))] if i == 0 else [y2 + 15 + 15 * k for k in range(len(lines))]
                for text, y in zip(lines, ys):
                    cv2.putText(frame, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        # legend + clock
        cv2.rectangle(frame, (0, H - 26), (W, H), (0, 0, 0), -1)
        x = 8
        for i, n in enumerate(names):
            cv2.putText(frame, n, (x, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[i % len(COLORS)], 2, cv2.LINE_AA)
            x += 15 * len(n) + 20
        clock = f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"
        cv2.putText(frame, clock, (W - 130, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
        if (fi - first) % int(fps * 60) == 0:
            print(f"  {clock[:8]} / {(last - 1) / fps / 60:.1f} min", flush=True)
    writer.release()
    cap.release()
    print("video ->", opts.out)


if __name__ == "__main__":
    main()
