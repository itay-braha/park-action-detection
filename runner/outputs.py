"""Write run.py's results: detections.csv (one row per person per sampled moment) and timeline.csv (per minute)."""
import csv
from collections import defaultdict


def write_detections(path, rows, video, model, top_k):
    """rows: [(chunk_name, chunk_start_sec, PersonPrediction)]"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "model", "chunk", "time_sec", "time_hms", "person_id", "person_score",
                    "x1", "y1", "x2", "y2"] + [c for i in range(1, top_k + 1) for c in (f"action_{i}", f"score_{i}")])
        for chunk, start, p in rows:
            t = start + p.time_sec
            acts = p.actions[:top_k] + [("", "")] * (top_k - len(p.actions[:top_k]))
            w.writerow([video, model, chunk, f"{t:.2f}", hms(t), f"{chunk}:{p.person_id}",
                        f"{p.person_score:.3f}" if p.person_score == p.person_score else "",
                        *[f"{v:.4f}" for v in p.box]] + [x if isinstance(x, str) else f"{x:.3f}"
                                                          for a in acts for x in a])


def write_timeline(path, rows, threshold):
    """Per minute: how many people were seen on average, and how many were doing each action (prob >= threshold).

    Averaged over the sampled moments in that minute, so TubeR (every 0.5 s) and LART (sampled at the same rate)
    produce comparable numbers: "on average 3.2 people were walking during minute 47".
    """
    moments = defaultdict(set)                         # minute -> sampled timestamps
    people = defaultdict(int)                          # minute -> person detections
    acting = defaultdict(lambda: defaultdict(int))     # minute -> action -> detections above threshold
    actions = set()
    for _, start, p in rows:
        t = start + p.time_sec
        m = int(t // 60)
        moments[m].add(round(t, 2))
        people[m] += 1
        for a, s in p.actions:
            if s >= threshold:
                acting[m][a] += 1
                actions.add(a)
    actions = sorted(actions, key=lambda a: -sum(acting[m][a] for m in acting))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["minute", "time_hms", "sampled_moments", "avg_people"] + actions)
        for m in sorted(moments):
            n = len(moments[m])
            w.writerow([m, hms(m * 60), n, f"{people[m] / n:.2f}"] + [f"{acting[m][a] / n:.2f}" for a in actions])
    return actions


def hms(t):
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"
