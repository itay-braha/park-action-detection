"""Draw annotated key-frames to JPEGs so you can eyeball that boxes and labels line up.

    python tools/preview_annotations.py --anno data/annotations/train.json --n 40

Do this before every training run on new data: most "the loss does not go down" problems are
annotation problems (boxes shifted by a wrong fps, x/y swapped, pixel vs normalised coordinates).
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parktuber.common import read_pbtxt  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anno", default="data/annotations/train.json")
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--labels", default="configs/park_actions.pbtxt")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="data/annotations/preview")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = read_pbtxt(args.labels)
    d = json.load(open(args.anno, encoding="utf-8"))
    keys = [k for k in d["frame_keys_list"] if d["video_frame_bbox"][k]["bboxes"]]
    random.Random(args.seed).shuffle(keys)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for key in keys[: args.n]:
        vid, f = key.rsplit(",", 1)
        img = cv2.imread(str(Path(args.frames) / vid / f"{int(f):06d}.jpg"))
        if img is None:
            print(f"missing frame for {key}")
            continue
        h, w = img.shape[:2]
        e = d["video_frame_bbox"][key]
        for (x1, y1, x2, y2), acts in zip(e["bboxes"], e["acts"]):
            p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
            cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
            text = ", ".join(names.get(a + 1, str(a)) for a in acts) or "person"
            cv2.putText(img, text, (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.imwrite(str(out / f"{key.replace(',', '_')}.jpg"), img)
    print(f"wrote {min(args.n, len(keys))} previews to {out}")


if __name__ == "__main__":
    main()
