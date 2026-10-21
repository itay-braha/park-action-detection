"""Draw LART's AVA-validation predictions (from our replication run) next to the human ground truth on the real movie.

    python tools/render_ava_demo.py --video-id z-fsLpGHq6o --start 1219 --duration 46 \
        --clip outputs/demo/z-fsLpGHq6o_1219.mp4 --out outputs/demo/lart_ava_demo.mp4

Inputs: outputs/eval_all_full_avg.6/ava_val.csv (LART's scored detections, written by eval_ava.py),
cache/.cache/lart/ava_val_v2.2.csv (AVA ground truth), and a clip of the AVA movie starting exactly at --start seconds
(e.g. cut from https://s3.amazonaws.com/ava-dataset/trainval/<video>.mkv with ffmpeg -ss <start> -t <duration>).
AVA labels exist once per second (key frames); each second's boxes are shown for +-0.5 s around it.
"""
import argparse
import csv
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GREEN, ORANGE, BLUE, GREY, WHITE = (80, 200, 80), (0, 150, 255), (230, 160, 40), (170, 170, 170), (255, 255, 255)


def short(name):
    return re.sub(r"\s*\(.*?\)", "", name).strip()     # "talk to (e.g., self, a person)" -> "talk to"


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def load(path, vid, s0, s1, key_cols):
    out = defaultdict(lambda: defaultdict(dict))         # sec -> key -> {"box":..., "acts": {id: score}}
    with open(path, newline="") as f:
        for r in csv.reader(f):
            if r[0] != vid or not s0 <= int(r[1]) <= s1:
                continue
            box = tuple(round(float(x), 3) for x in r[2:6])
            key = box if key_cols == "box" else r[7]
            e = out[int(r[1])].setdefault(key, {"box": box, "acts": {}})
            e["acts"][int(r[6])] = float(r[7]) if key_cols == "box" else 1.0
    return out


def put(img, text, org, color, scale=0.5, bg=(0, 0, 0)):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x, y - h - 4), (x + w + 4, y + 3), bg, -1)
    cv2.putText(img, text, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return h + 7


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--start", type=float, required=True, help="movie time (s) of the clip's first frame")
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pred", default=str(ROOT / "outputs/eval_all_full_avg.6/ava_val.csv"))
    ap.add_argument("--gt", default=str(ROOT / "cache/.cache/lart/ava_val_v2.2.csv"))
    ap.add_argument("--thresh", type=float, default=0.3, help="show actions with probability >= this")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    names = {int(i): n for n, i in re.findall(r'name: "([^"]+)"\s*\n\s*label_id: (\d+)',
                                              (ROOT / "cache/.cache/lart/ava_action_list_v2.2.pbtxt").read_text())}
    s0, s1 = int(args.start), int(args.start + args.duration) + 1
    print("reading predictions (large file)...")
    pred = load(args.pred, args.video_id, s0, s1, "box")
    gt = load(args.gt, args.video_id, s0, s1, "person")
    print(f"{sum(len(v) for v in pred.values())} predicted boxes, {sum(len(v) for v in gt.values())} ground-truth boxes")

    cap = cv2.VideoCapture(args.clip)
    fps, W, H = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
    tmp = Path(args.out).with_suffix(".tmp.mp4")
    wr = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    stats = {"shown": 0, "correct_top1": 0, "missed_gt": 0}
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        t = args.start + i / fps
        sec = int(round(t))
        boxes_p = [p for p in pred.get(sec, {}).values() if max(p["acts"].values()) >= args.thresh]
        boxes_g = list(gt.get(sec, {}).values())
        count = abs(t - sec) < 0.5 / fps          # count statistics once per key frame
        for g in boxes_g:                           # ground-truth people the detector missed
            if not any(iou(g["box"], p["box"]) >= 0.5 for p in boxes_p):
                x1, y1, x2, y2 = (int(g["box"][0] * W), int(g["box"][1] * H), int(g["box"][2] * W), int(g["box"][3] * H))
                for k in range(x1, x2, 12):
                    cv2.line(img, (k, y1), (min(k + 6, x2), y1), GREY, 2); cv2.line(img, (k, y2), (min(k + 6, x2), y2), GREY, 2)
                put(img, "missed by detector: " + ", ".join(short(names[a]) for a in g["acts"]), (x1, max(15, y1 - 4)), GREY, 0.45)
                stats["missed_gt"] += count
        for p in boxes_p:
            x1, y1, x2, y2 = (int(p["box"][0] * W), int(p["box"][1] * H), int(p["box"][2] * W), int(p["box"][3] * H))
            match = max(boxes_g, key=lambda g: iou(g["box"], p["box"]), default=None)
            truth = set(match["acts"]) if match and iou(match["box"], p["box"]) >= 0.5 else set()
            cv2.rectangle(img, (x1, y1), (x2, y2), BLUE, 2)
            top = sorted(p["acts"].items(), key=lambda a: -a[1])[: args.topk]
            y = max(18, y1 - 6 - 18 * len([a for a in top if a[1] >= args.thresh]))
            for a, s in top:
                if s < args.thresh:
                    continue
                y += put(img, f"{short(names[a])} {s:.2f}", (x1, y), GREEN if a in truth else ORANGE)
            if truth:
                put(img, "truth: " + ", ".join(short(names[a]) for a in sorted(truth)), (x1, min(H - 4, y2 + 16)), WHITE, 0.45,
                    bg=(40, 90, 40))
            if count:
                stats["shown"] += 1
                stats["correct_top1"] += bool(truth) and top[0][0] in truth
        # header + legend
        cv2.rectangle(img, (0, 0), (W, 46), (0, 0, 0), -1)
        put(img, f"LART (CVPR 2023) on AVA v2.2 val - movie {args.video_id}  t={int(t // 60)}:{t % 60:04.1f}  "
                 f"(our replication: 45.11 mAP)", (8, 18), WHITE, 0.55)
        x = 8
        for txt, col in (("predicted action = also in human labels", GREEN), ("predicted, not in human labels", ORANGE),
                         ("truth: human labels", WHITE), ("person the detector missed", GREY)):
            put(img, txt, (x, 40), col, 0.45)
            x += cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0] + 24
        wr.write(img)
        if i % int(fps * 10) == int(fps * 3):        # a few stills for slides
            cv2.imwrite(str(Path(args.out).with_name(f"{Path(args.out).stem}_t{int(t)}.jpg")), img)
        i += 1
    cap.release(); wr.release()

    # re-encode to H.264 so it plays everywhere (PowerPoint, browsers, phones)
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ff = "ffmpeg"
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", "-crf", "20", str(args.out)], check=True)
    tmp.unlink()
    print(f"wrote {args.out} ({i} frames)")
    print(f"key frames: {stats['shown']} predicted people, top-1 action in human labels for "
          f"{stats['correct_top1']} ({stats['correct_top1'] / max(1, stats['shown']):.0%}); "
          f"{stats['missed_gt']} labelled people missed by the detector")


if __name__ == "__main__":
    main()
