"""Synthetic "motion-only" dataset used by TubeR_walkthrough.ipynb (Part E).

Every video shows identical grey rectangles at random places. Some stand still ("stand"), some move in a
random direction ("run"). Colour, size and position carry NO information about the class - only motion does.
(The smoke-test data has a shortcut: the runner is always blue and always at the bottom. See the notebook.)
"""
import csv
from pathlib import Path

import cv2
import numpy as np

W, H, FPS, SECS = 320, 240, 25, 8
BW, BH = 22, 56                     # "person" size in pixels


def make_video(path, rng, n_objects=3):
    kinds = ["stand", "run"] + list(rng.choice(["stand", "run"], n_objects - 2))
    objs = []
    for kind in kinds:
        x, y = rng.uniform(10, W - BW - 10), rng.uniform(10, H - BH - 10)
        ang = rng.uniform(0, 2 * np.pi)
        speed = rng.uniform(25, 45) if kind == "run" else 0.0
        objs.append({"kind": kind, "x": x, "y": y, "vx": speed * np.cos(ang), "vy": speed * np.sin(ang)})
    bg = (np.full((H, W, 3), (70, 120, 70), np.float32) + rng.normal(0, 6, (H, W, 3))).clip(0, 255).astype(np.uint8)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    rows = []
    for i in range(FPS * SECS):
        t = i / FPS
        img = bg.copy()
        for pid, o in enumerate(objs):
            # bounce off the borders so movers stay in view
            o["x"] += o["vx"] / FPS; o["y"] += o["vy"] / FPS
            if not 0 <= o["x"] <= W - BW: o["vx"] *= -1; o["x"] = np.clip(o["x"], 0, W - BW)
            if not 0 <= o["y"] <= H - BH: o["vy"] *= -1; o["y"] = np.clip(o["y"], 0, H - BH)
            x1, y1 = int(o["x"]), int(o["y"])
            cv2.rectangle(img, (x1, y1), (x1 + BW, y1 + BH), (60, 60, 60), -1)
            cv2.circle(img, (x1 + BW // 2, y1 - 6), 7, (60, 60, 60), -1)          # a "head"
            if i % FPS == 0 and 1 <= t <= SECS - 1:
                rows.append([path.stem, f"{t:.2f}", x1 / W, (y1 - 13) / H, (x1 + BW) / W, (y1 + BH) / H,
                             1 if o["kind"] == "stand" else 2, pid])
        out.write(img)
    out.release()
    return rows


def make_dataset(root, n_train=6, n_val=2, seed=0):
    root = Path(root); (root / "videos").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(n_train + n_val):
        split = "train" if v < n_train else "val"
        rows += make_video(root / "videos" / f"{split}{v:02d}_toy.mp4", rng)
    with open(root / "annotations.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "person_id"]); w.writerows(rows)
    (root / "labels.pbtxt").write_text('item {\n  name: "stand"\n  id: 1\n}\nitem {\n  name: "run"\n  id: 2\n}\n')
    (root / "ava_to_toy.yaml").write_text("stand: [12]   # AVA 'stand'\nrun: [10, 14] # AVA 'run/jog' + 'walk'\n")
    return root
