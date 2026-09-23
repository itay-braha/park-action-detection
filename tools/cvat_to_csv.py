"""Convert a CVAT "CVAT for video 1.1" XML export into the project's AVA-style CSV.

    python tools/cvat_to_csv.py --xml exports/cam03_task.xml --video-id cam03_2024-05-01 \
        --labels configs/park_actions.pbtxt --out data/annotations/cam03.csv

Recommended CVAT project setup (see docs/03_data_and_annotation.md):
  * one label `person`, drawn as a *track* (rectangle) on every person you can see;
  * one checkbox attribute per action, named exactly like the classes in the .pbtxt
    (e.g. walk, run, sit, ...), marked "mutable" so it can change along the track.
    Multi-label is natural: a person can `walk` and `talk_to_person` at the same time.
  Also accepted: a single select/radio attribute (default name `action`) whose value is a class name,
  or tracks whose *label* is itself a class name.

We sample one key-frame every --every-sec seconds (AVA uses 1 s) and linearly interpolate boxes
between the frames CVAT exported. Output columns (one row per person x action; a person with no
action gets action_id 0 = "person, no action" which still trains the person detector):

    video_id,timestamp,x1,y1,x2,y2,action_id,person_id      (boxes normalised to [0,1])
"""
import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parktuber.common import read_meta, read_pbtxt, video_id_from_path  # noqa: E402

TRUE = {"true", "1", "yes"}


def parse_track(track, name_to_id, action_attr):
    """-> {frame: (xtl, ytl, xbr, ybr, set(action_ids))} for frames where the person is visible."""
    out = {}
    track_label_action = name_to_id.get(track.get("label"))
    for box in track.findall("box"):
        if box.get("outside") == "1":
            continue
        f = int(box.get("frame"))
        acts = set()
        if track_label_action:
            acts.add(track_label_action)
        for att in box.findall("attribute"):
            name, val = att.get("name"), (att.text or "").strip()
            if name == action_attr and val in name_to_id:
                acts.add(name_to_id[val])
            elif name in name_to_id and val.lower() in TRUE:
                acts.add(name_to_id[name])
        out[f] = (float(box.get("xtl")), float(box.get("ytl")), float(box.get("xbr")), float(box.get("ybr")), acts)
    return out


def box_at(frames_sorted, boxes, f):
    """Box at frame f: exact, or linearly interpolated between neighbours (actions from the nearer one)."""
    if f in boxes:
        return boxes[f]
    import bisect
    i = bisect.bisect_left(frames_sorted, f)
    if i == 0 or i == len(frames_sorted):
        return None
    f0, f1 = frames_sorted[i - 1], frames_sorted[i]
    if f1 - f0 > 60:  # a long gap means the person was not visible
        return None
    a = (f - f0) / (f1 - f0)
    b0, b1 = boxes[f0], boxes[f1]
    xy = tuple(b0[k] * (1 - a) + b1[k] * a for k in range(4))
    return xy + ((b0 if a < 0.5 else b1)[4],)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--labels", default="configs/park_actions.pbtxt")
    ap.add_argument("--video-id", help="defaults to the source file name recorded in the XML")
    ap.add_argument("--frames", default="data/frames", help="to read src fps/size from meta.json if present")
    ap.add_argument("--fps", type=float, help="source video fps (overrides meta.json)")
    ap.add_argument("--every-sec", type=float, default=1.0, help="key-frame spacing in seconds")
    ap.add_argument("--action-attr", default="action", help="name of a select-type action attribute")
    ap.add_argument("--frame-step", type=int, default=1,
                    help="CVAT task 'frame step' (if the task was created with every n-th frame)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    id_to_name = read_pbtxt(args.labels)
    name_to_id = {n: i for i, n in id_to_name.items()}
    root = ET.parse(args.xml).getroot()
    source = root.findtext(".//meta//source") or root.findtext(".//meta//task/name") or "video"
    vid = args.video_id or video_id_from_path(source)
    width = float(root.findtext(".//original_size/width") or 0)
    height = float(root.findtext(".//original_size/height") or 0)

    fps = args.fps
    meta_path = Path(args.frames) / vid / "meta.json"
    if meta_path.exists():
        meta = read_meta(args.frames, vid)
        fps = fps or meta["src_fps"]
        width = width or meta["orig_width"]
        height = height or meta["orig_height"]
    if not fps:
        sys.exit("unknown source fps: pass --fps or extract frames first (meta.json)")
    if not (width and height):
        sys.exit("unknown frame size: not in XML <original_size> and no meta.json")

    tracks = root.findall("track")
    rows, unknown = [], set()
    for tr in tracks:
        boxes = parse_track(tr, name_to_id, args.action_attr)
        if not boxes:
            continue
        for att in tr.iter("attribute"):
            n = att.get("name")
            if n != args.action_attr and n not in name_to_id and n not in {"occluded"}:
                unknown.add(n)
        frames_sorted = sorted(boxes)
        t0 = frames_sorted[0] * args.frame_step / fps
        t1 = frames_sorted[-1] * args.frame_step / fps
        k = int(t0 // args.every_sec)
        while k * args.every_sec <= t1:
            t = k * args.every_sec
            k += 1
            if t < t0:
                continue
            b = box_at(frames_sorted, boxes, int(round(t * fps / args.frame_step)))
            if b is None:
                continue
            x1, y1, x2, y2 = b[0] / width, b[1] / height, b[2] / width, b[3] / height
            for a in sorted(b[4]) or [0]:
                rows.append([vid, f"{t:.3f}", f"{x1:.4f}", f"{y1:.4f}", f"{x2:.4f}", f"{y2:.4f}", a, tr.get("id")])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "person_id"])
        w.writerows(rows)
    print(f"{vid}: {len(tracks)} tracks -> {len(rows)} rows at {args.every_sec}s key-frames -> {args.out}")
    if unknown:
        print(f"warning: attributes not matching any class in {args.labels}: {sorted(unknown)}")


if __name__ == "__main__":
    main()
