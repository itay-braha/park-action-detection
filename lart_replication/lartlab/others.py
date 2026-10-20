"""Which "other people" does LART see?  Measuring and fixing the context sampling.

PHALP_action_dataset gives the person-of-interest up to 4 other tracks, drawn uniformly at random from ALL tracks
of the same 15-minute AVA video (`self.track2video[idx]`, then np.random.shuffle). A track lasts a few seconds,
so most of those draws do not overlap it in time - their tokens are then empty ([MASK]) and the model gets no
interaction signal. Here we compute each track's absolute frame interval and can restrict the candidates to tracks
that are actually on screen at the same time ("co-occurring" sampling).
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
    """Frame intervals for many tracks, cached on disk (reading a track takes ~5 ms)."""
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
    """For each track: how many tracks of the same video exist, and how many overlap it in time."""
    n_same, n_overlap = [], []
    for path, others in zip(ds.data, ds.track2video):
        me = intervals[os.path.basename(str(path))]
        o = [intervals[os.path.basename(str(x))] for x in others if os.path.basename(str(x)) in intervals]
        n_same.append(len(others))
        n_overlap.append(sum(overlap(me, x) > 0 for x in o))
    return np.array(n_same), np.array(n_overlap)


def use_cooccurring_others(ds, intervals):
    """Make the dataset draw 'other people' only among tracks that overlap the person-of-interest in time."""
    new = []
    for path, others in zip(ds.data, ds.track2video):
        me = intervals[os.path.basename(str(path))]
        keep = [x for x in others
                if os.path.basename(str(x)) in intervals and overlap(me, intervals[os.path.basename(str(x))]) > 0]
        new.append(keep)
    ds.track2video = np.asarray(new + [[]], dtype=object)[:-1]   # keep a 1-D object array
    return ds
