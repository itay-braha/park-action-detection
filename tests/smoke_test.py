"""End-to-end smoke test on synthetic videos (no real data or downloads needed; ~1-2 min on a GPU).

    python tests/smoke_test.py            # tiny random-init CSN-50 model
    python tests/smoke_test.py --real     # also loads weights/TubeR_CSN152_AVA22.pth (tests real init + memory)

It runs every step of the real workflow: make videos -> extract_frames -> CSV -> build_annotations
-> preview -> train (2 epochs) -> evaluate -> infer_video. A pass means the plumbing works;
the numbers mean nothing (the "people" are coloured rectangles).
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "runs" / "smoke"
PY = sys.executable


def run(*cmd):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=ROOT, check=True)


def make_videos():
    """Three 8 s clips, 25 fps: one rectangle 'stands', one 'runs' across the frame."""
    vids = WORK / "videos"
    vids.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(0)
    for v in range(3):
        name = f"cam{v}_synthetic"
        fps, W, H, secs = 25, 320, 240, 8
        out = cv2.VideoWriter(str(vids / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        sx, sy = rng.integers(20, 120), rng.integers(60, 140)
        for i in range(fps * secs):
            t = i / fps
            img = np.full((H, W, 3), (60, 120, 60), np.uint8)
            stand = (int(sx), int(sy), int(sx) + 24, int(sy) + 60)
            rx = int(10 + (t / secs) * (W - 40))
            run_box = (rx, 150, rx + 24, 210)
            for (x1, y1, x2, y2), c in ((stand, (200, 50, 50)), (run_box, (50, 50, 220))):
                cv2.rectangle(img, (x1, y1), (x2, y2), c, -1)
            out.write(img)
            if i % fps == 0 and 1 <= t <= secs - 1:
                for box, act, pid in ((stand, 1, "a"), (run_box, 2, "b")):
                    x1, y1, x2, y2 = box
                    rows.append([name, f"{t:.2f}", x1 / W, y1 / H, x2 / W, y2 / H, act, pid])
        out.release()
    csv_path = WORK / "annotations.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "person_id"])
        w.writerows(rows)
    (WORK / "labels.pbtxt").write_text('item {\n  name: "stand"\n  id: 1\n}\nitem {\n  name: "run"\n  id: 2\n}\n')
    return vids, csv_path


def write_config(real):
    rel = lambda p: p.relative_to(ROOT).as_posix()  # noqa: E731
    model = ("  MODEL:\n    BACKBONE_NAME: \"CSN-152\"\n" if real else "  MODEL:\n    BACKBONE_NAME: \"CSN-50\"\n")
    init = '"weights/TubeR_CSN152_AVA22.pth"' if real else '""'
    cfg = f"""CONFIG:
  DATA:
    FRAMES_DIR: "{rel(WORK / 'frames')}"
    TRAIN_ANNO: "{rel(WORK / 'anno' / 'train.json')}"
    VAL_ANNO: "{rel(WORK / 'anno' / 'val.json')}"
    LABEL_PATH: "{rel(WORK / 'labels.pbtxt')}"
    TEMP_LEN: {32 if real else 8}
    IMG_SIZE: {256 if real else 128}
    IMG_RESHAPE_SIZE: {288 if real else 144}
    NUM_WORKERS: 0
{model}  TRAIN:
    EPOCH_NUM: 2
    BATCH_SIZE: 1
    LR_MILESTONE: [100]
  FINETUNE:
    INIT_FROM: {init}
    HEAD_INIT_MAPPING: ""
    FREEZE_BACKBONE: "{'layer3' if real else 'none'}"
    GRAD_ACCUM: 2
  LOG:
    DISPLAY_FREQ: 5
    BASE_PATH: "{rel(WORK / 'runs')}"
    EXP_NAME: "smoke"
"""
    path = WORK / "smoke.yaml"
    path.write_text(cfg)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="use the real CSN-152 AVA checkpoint")
    args = ap.parse_args()
    vids, csv_path = make_videos()
    run(PY, "tools/extract_frames.py", "--videos", vids, "--out", WORK / "frames", "--fps", 30, "--short-side", 0,
        "--workers", 1, "--overwrite")
    run(PY, "tools/build_annotations.py", "--csv", csv_path, "--frames", WORK / "frames",
        "--labels", WORK / "labels.pbtxt", "--out", WORK / "anno", "--group-regex", r"^(cam\d+)",
        "--val-groups", "cam2", "--add-empty-keyframes", "1.0")
    run(PY, "tools/preview_annotations.py", "--anno", WORK / "anno" / "train.json", "--frames", WORK / "frames",
        "--labels", WORK / "labels.pbtxt", "--n", 3, "--out", WORK / "preview")
    cfg = write_config(args.real)
    run(PY, "train.py", "--config", cfg, "--eval-first")
    ckpt = WORK / "runs" / "smoke" / "best.pth"
    run(PY, "evaluate.py", "--config", cfg, "--checkpoint", ckpt)
    run(PY, "infer_video.py", "--config", cfg, "--checkpoint", ckpt, "--video", vids / "cam2_synthetic.mp4",
        "--out", WORK / "infer" / "cam2", "CONFIG.FINETUNE.VIS_BINARY_THRESH", "0.0",
        "CONFIG.FINETUNE.VIS_ACTION_THRESH", "0.0")
    assert (WORK / "infer" / "cam2.mp4").exists() and (WORK / "infer" / "cam2.csv").exists()
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
