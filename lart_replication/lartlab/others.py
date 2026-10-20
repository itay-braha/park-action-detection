"""Which "other people" does LART see?

PHALP_action_dataset groups tracks by file-name prefix `ava-val_<video>_<keyframe>`, i.e. by AVA *key frame*:
the "others" of a person-of-interest are the other people tracked around the same annotated moment (so they are
on screen at the same time). Up to max_people-1 = 4 of them are used; if there are more, 4 are drawn at random
(np.random.shuffle - upstream evaluates unseeded, so crowded scenes make the score slightly random).
"""
import json
import os
import re
from pathlib import Path

import joblib
import numpy as np

from . import ROOT

_FRAME = re.compile(r"_(\d{6})\.jpg$")


def track_interval(path):
    """(first, last) absolute frame number of a tracklet, read from its frame names."""
    t = joblib.load(path)
    names = t["frame_name"]
    return int(_FRAME.search(names[0]).group(1)), int(_FRAME.search(names[-1]).group(1))


def intervals_for(paths, cache=ROOT / "data" / "_TMP" / "track_intervals.json"):
    """Frame intervals for many tracks, cached on disk."""
    cache = Path(cache)
    known = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [p for p in map(str, paths) if os.path.basename(p) not in known]
    if todo:
        res = joblib.Parallel(n_jobs=8)(joblib.delayed(track_interval)(p) for p in todo)
        known.update({os.path.basename(p): r for p, r in zip(todo, res)})
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(known))
    return {os.path.basename(str(p)): tuple(known[os.path.basename(str(p))]) for p in paths}


def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def context_stats(ds, intervals):
    """Per track: number of other tracks at the same key frame, and how many of them overlap it in time."""
    n_same, n_overlap = [], []
    for path, others in zip(ds.data, ds.track2video):
        me = intervals[os.path.basename(str(path))]
        o = [intervals[os.path.basename(str(x))] for x in others if os.path.basename(str(x)) in intervals]
        n_same.append(len(others))
        n_overlap.append(sum(overlap(me, x) > 0 for x in o))
    return np.array(n_same), np.array(n_overlap)
