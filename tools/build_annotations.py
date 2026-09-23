"""Turn AVA-style CSV annotations into the train/val JSON files the dataset loader reads.

    python tools/build_annotations.py --csv data/annotations/*.csv --frames data/frames \
        --labels configs/park_actions.pbtxt --out data/annotations \
        --group-regex "^(cam\\d+)" --val-ratio 0.2 --add-empty-keyframes 1.0

Input CSV columns (header optional; same as AVA but timestamp is seconds from the file start):
    video_id,timestamp,x1,y1,x2,y2,action_id,person_id
      boxes normalised to [0,1]; action_id 1..N from the label map, 0 = person with no action.
      Several rows with the same (video, timestamp, person) are merged into one multi-label box.

Splitting.  Clips from the *same camera* look almost identical (same background, same angle), so a
random split by clip or even by video file leaks information and gives inflated val scores.
Use --group-regex to extract a group (e.g. the camera id) from the video id; whole groups go to
either train or val. Or name the val groups/videos explicitly with --val-groups.

Output JSON (compatible with upstream TubeR's ava_{train,val}_v22.json structure):
    {"video_frame_bbox": {"<vid>,<frame>": {"bboxes": [[x1,y1,x2,y2],...], "acts": [[0-based ids],...]}},
     "frame_keys_list": [...], "meta": {...}}
"""
import argparse
import csv
import glob
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parktuber.common import read_meta, read_pbtxt  # noqa: E402


def read_rows(paths):
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if not row or row[0].strip() in ("video_id", "") or row[0].startswith("#"):
                    continue
                vid, ts, x1, y1, x2, y2, act = row[:7]
                pid = row[7] if len(row) > 7 else ""
                yield vid.strip(), float(ts), [float(x1), float(y1), float(x2), float(y2)], int(float(act)), pid.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", required=True, help="CSV files (globs allowed)")
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--labels", default="configs/park_actions.pbtxt")
    ap.add_argument("--out", default="data/annotations")
    ap.add_argument("--group-regex", default=None,
                    help="regex with one group applied to video_id -> split unit (e.g. camera). Default: video")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--val-groups", nargs="*", default=None, help="explicit list of groups for validation")
    ap.add_argument("--add-empty-keyframes", type=float, default=0.0,
                    help="if annotation is exhaustive, add key-frames every N sec with nobody in them")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = sorted({p for pat in args.csv for p in glob.glob(pat)})
    if not paths:
        sys.exit("no CSV files matched")
    id_to_name = read_pbtxt(args.labels)
    num_classes = len(id_to_name)
    assert sorted(id_to_name) == list(range(1, num_classes + 1)), "label ids must be exactly 1..N"

    metas, missing = {}, set()
    samples = defaultdict(dict)  # key -> person -> {"bbox", "acts"}
    bad_ids = Counter()
    for vid, ts, bbox, act, pid in read_rows(paths):
        if vid not in metas:
            try:
                metas[vid] = read_meta(args.frames, vid)
            except FileNotFoundError:
                missing.add(vid)
                continue
        m = metas[vid]
        fidx = int(round((ts - m.get("start_sec", 0.0)) * m["fps"]))
        if not 0 <= fidx < m["n_frames"]:
            bad_ids["timestamp outside extracted frames"] += 1
            continue
        if act < 0 or act > num_classes:
            bad_ids[f"unknown action id {act}"] += 1
            continue
        bbox = [min(max(v, 0.0), 1.0) for v in bbox]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            bad_ids["degenerate box"] += 1
            continue
        key = f"{vid},{fidx:06d}"
        person = pid or ",".join(f"{v:.3f}" for v in bbox)
        entry = samples[key].setdefault(person, {"bbox": bbox, "acts": set()})
        if act > 0:
            entry["acts"].add(act - 1)  # model outputs are 0-based

    if missing:
        print(f"warning: {len(missing)} video(s) in CSV have no extracted frames (skipped): {sorted(missing)[:5]}...")
    for k, v in bad_ids.items():
        print(f"warning: skipped {v} rows: {k}")

    if args.add_empty_keyframes > 0:
        by_vid = defaultdict(list)
        for key in samples:
            vid, f = key.rsplit(",", 1)
            by_vid[vid].append(int(f))
        added = 0
        for vid, frames in by_vid.items():
            fps = metas[vid]["fps"]
            step = max(1, int(round(args.add_empty_keyframes * fps)))
            for f in range(min(frames), max(frames) + 1, step):
                key = f"{vid},{f:06d}"
                if key not in samples and all(abs(f - g) > step // 2 for g in frames):
                    samples[key] = {}
                    added += 1
        print(f"added {added} empty key-frames (nobody annotated)")

    # ---- split by group ----
    def group_of(vid):
        if args.group_regex:
            m = re.search(args.group_regex, vid)
            return m.group(1) if m else vid
        return vid

    groups = sorted({group_of(k.rsplit(",", 1)[0]) for k in samples})
    if args.val_groups:
        val_groups = set(args.val_groups)
    else:
        rng = random.Random(args.seed)
        shuffled = groups[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(groups) * args.val_ratio))) if len(groups) > 1 else 0
        val_groups = set(shuffled[:n_val])
    if not val_groups:
        print("warning: only one group -> no validation split possible; val = train (numbers will be optimistic)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split in ("train", "val"):
        in_split = (lambda g: g in val_groups) if split == "val" else (lambda g: g not in val_groups)
        if split == "val" and not val_groups:
            in_split = lambda g: True  # noqa: E731
        vfb, keys, counts = {}, [], Counter()
        for key in sorted(samples):
            if not in_split(group_of(key.rsplit(",", 1)[0])):
                continue
            persons = list(samples[key].values())
            vfb[key] = {"bboxes": [p["bbox"] for p in persons], "acts": [sorted(p["acts"]) for p in persons]}
            keys.append(key)
            counts["_keyframes"] += 1
            counts["_boxes"] += len(persons)
            for p in persons:
                for a in p["acts"]:
                    counts[id_to_name[a + 1]] += 1
        meta = {"num_classes": num_classes, "labels": id_to_name,
                "groups": sorted({group_of(k.rsplit(',', 1)[0]) for k in keys})}
        with open(out / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump({"video_frame_bbox": vfb, "frame_keys_list": keys, "meta": meta}, f)
        stats[split] = dict(counts)
        print(f"{split}: {counts['_keyframes']} key-frames, {counts['_boxes']} boxes, groups={meta['groups']}")

    print(f"\n{'class':32s} {'train':>7s} {'val':>7s}")
    for i, name in id_to_name.items():
        tr, va = stats["train"].get(name, 0), stats["val"].get(name, 0)
        flag = "   <-- too few examples" if tr < 30 else ""
        print(f"{name:32s} {tr:7d} {va:7d}{flag}")
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
