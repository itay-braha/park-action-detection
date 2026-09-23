"""Run TubeR on a raw video file: draws people + actions into a new video and writes a CSV.

    # your fine-tuned model
    python infer_video.py --config configs/park_csn152_8gb.yaml --checkpoint runs/park_tuber/best.pth \
        --video data/raw_videos/cam03.mp4 --out outputs/cam03

    # the original AVA model, zero-shot on your footage (80 AVA actions) - good first experiment
    python infer_video.py --config configs/ava_zero_shot.yaml --checkpoint "" --video my.mp4 --out outputs/zs

How it works: frames are re-sampled to DATA.CLIP_FPS and kept in a sliding window. Every
--stride-sec seconds, the clip of TEMP_LEN frames (step FRAME_RATE) centred on the current frame
goes through the model, which returns up to QUERY_NUM people with boxes on the centre frame and a
probability for every action. Output is delayed by half a clip (~1 s) because the model looks
into the "future" half of the window as well.
"""
import argparse
import csv
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

from parktuber.config import load_cfg
from parktuber.checkpoint import read_label_names
from parktuber.engine import autocast_ctx
from parktuber.model import build, postprocess

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PALETTE = [(230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66), (0, 114, 178),
           (213, 94, 0), (204, 121, 167), (120, 120, 255), (255, 120, 120), (160, 255, 160)]


def resampled_frames(path, fps):
    """Yield BGR frames re-sampled on the time axis to `fps` (duplicates / drops frames)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    i, n_out = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        while n_out / fps < (i + 1) / src_fps:
            yield frame
            n_out += 1
        i += 1
    cap.release()


def to_input(frame, size):
    h, w = frame.shape[:2]
    nh, nw = (size, int(round(size * w / h))) if h <= w else (int(round(size * h / w)), size)
    rgb = cv2.cvtColor(cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0 - MEAN) / STD  # H, W, 3


def draw(frame, dets, names):
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        acts = d["actions"][:3]
        color = PALETTE[(acts[0][0] if acts else 0) % len(PALETTE)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        lines = [f"{names.get(a + 1, a)} {s:.2f}" for a, s in acts] or [f"person {d['person']:.2f}"]
        for k, text in enumerate(lines):
            y = y1 - 6 - 16 * (len(lines) - 1 - k)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, max(0, y - th - 3)), (x1 + tw + 4, max(th + 3, y + 3)), color, -1)
            cv2.putText(frame, text, (x1 + 2, max(th, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True, help='fine-tuned .pth, or "" to use FINETUNE.INIT_FROM')
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="output prefix: writes <out>.mp4 and <out>.csv")
    ap.add_argument("--stride-sec", type=float, default=0.5, help="run the model every N seconds")
    ap.add_argument("--max-sec", type=float, default=None, help="stop after N seconds of video")
    ap.add_argument("--no-video", action="store_true", help="only write the CSV")
    ap.add_argument("opts", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.opts)
    D, F = cfg.CONFIG.DATA, cfg.CONFIG.FINETUNE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = build(cfg, checkpoint=args.checkpoint)
    model.to(device).eval()
    names = read_label_names(D.LABEL_PATH)

    fps, T, rate = D.CLIP_FPS, D.TEMP_LEN, D.FRAME_RATE
    half = (T // 2) * rate
    offsets = (np.arange(T) - T // 2) * rate
    stride = max(1, int(round(args.stride_sec * fps)))
    buf = deque()          # (index, original_frame, model_input)
    dets, writer = [], None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fcsv = open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8")
    wcsv = csv.writer(fcsv)
    wcsv.writerow(["time_sec", "person_id_in_frame", "person_score", "x1", "y1", "x2", "y2", "action", "score"])

    @torch.no_grad()
    def run_model(center):
        idx = {i: x for i, _, x in buf}
        lo, hi = min(idx), max(idx)
        clip = np.stack([idx[int(np.clip(center + o, lo, hi))] for o in offsets])  # T,H,W,3
        x = torch.from_numpy(clip).permute(3, 0, 1, 2).contiguous().to(device)    # 3,T,H,W
        with autocast_ctx(cfg):
            outputs = model([x])
        H, W = x.shape[-2:]
        scores, boxes, person = postprocess(outputs, torch.tensor([[H, W]]), F.VIS_BINARY_THRESH)
        keep = np.where(person[0] > F.VIS_BINARY_THRESH)[0]
        if len(keep):
            k = torchvision.ops.nms(torch.from_numpy(boxes[0][keep]), torch.from_numpy(person[0][keep]), 0.6).numpy()
            keep = keep[k]
        result = []
        for q in keep:
            nb = boxes[0][q] / np.array([W, H, W, H])
            acts = [(int(a), float(scores[0][q][a])) for a in np.argsort(-scores[0][q])
                    if scores[0][q][a] >= F.VIS_ACTION_THRESH and (int(a) + 1) in names]
            result.append({"nbox": np.clip(nb, 0, 1), "person": float(person[0][q]), "actions": acts})
        return result

    def emit(center):
        nonlocal dets, writer
        if center % stride == 0:
            dets = run_model(center)
            for pid, d in enumerate(dets):
                for a, s in d["actions"] or [(-1, 0.0)]:
                    wcsv.writerow([f"{center / fps:.2f}", pid, f"{d['person']:.3f}", *[f"{v:.4f}" for v in d["nbox"]],
                                   names.get(a + 1, "") if a >= 0 else "", f"{s:.3f}"])
        if args.no_video:
            return
        frame = next(f for i, f, _ in buf if i == center).copy()
        h, w = frame.shape[:2]
        for d in dets:
            d["box"] = tuple(int(v) for v in d["nbox"] * np.array([w, h, w, h]))
        if writer is None:
            writer = cv2.VideoWriter(str(out.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(draw(frame, dets, names))

    n = 0
    for n, frame in enumerate(resampled_frames(args.video, fps)):
        if args.max_sec and n >= args.max_sec * fps:
            n -= 1
            break
        buf.append((n, frame, to_input(frame, D.IMG_SIZE)))
        while buf[0][0] < n - 2 * half:
            buf.popleft()
        if n >= half:
            emit(n - half)
        if n % (fps * 10) == 0:
            print(f"  {n / fps:.0f}s processed", flush=True)
    for c in range(max(0, n - half + 1), n + 1):  # flush the tail
        emit(c)
    fcsv.close()
    if writer:
        writer.release()
    print(f"wrote {out.with_suffix('.csv')}" + ("" if args.no_video else f" and {out.with_suffix('.mp4')}"))


if __name__ == "__main__":
    main()
