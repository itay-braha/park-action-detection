"""One self-contained HTML report from run.py results: charts over time, top actions, model agreement, snapshots.

    python tools/compare_report.py --video real_data/out.mp4 --runs results/out_tuber results/out_lart \
        --out results/out_report.html

Works with one run or several (each --runs dir needs detections.csv; run_info.json is used if present).
Agreement between two models: at the same moment (0.5 s grid) boxes are matched by IoU >= --iou, then their
top-1 actions are compared.
"""
import argparse
import base64
import csv
import html
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

COLORS = ["#e8710a", "#1a73e8", "#188038", "#a142f4"]   # per model, BGR conversion below for OpenCV


def load_run(d):
    d = Path(d)
    rows = []
    with open(d / "detections.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            acts = [(r[f"action_{i}"], float(r[f"score_{i}"])) for i in range(1, 10)
                    if r.get(f"action_{i}")]
            rows.append({"t": float(r["time_sec"]), "box": [float(r[k]) for k in ("x1", "y1", "x2", "y2")],
                         "acts": acts, "top": acts[0][0] if acts else "", "model": r["model"]})
    info = json.loads((d / "run_info.json").read_text()) if (d / "run_info.json").exists() else {}
    name = rows[0]["model"] if rows else d.name
    return {"name": name, "dir": d, "rows": rows, "info": info}


def grid(t):
    return round(t * 2) / 2


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def chart_people(runs, duration):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    for i, r in enumerate(runs):
        per = defaultdict(int)
        moments = defaultdict(set)
        for x in r["rows"]:
            b = int(x["t"] // 10)
            per[b] += 1
            moments[b].add(grid(x["t"]))
        bins = np.arange(0, int(duration // 10) + 1)
        n = [len(moments[b]) or 1 for b in bins]
        ax.plot(bins * 10 / 60, [per[b] / k for b, k in zip(bins, n)], color=COLORS[i], label=r["name"], lw=1.6)
    ax.set_xlabel("minute in video")
    ax.set_ylabel("people in view (avg per 10 s)")
    ax.grid(alpha=.3)
    ax.legend()
    return fig_to_img(fig)


def chart_actions(runs, top=12):
    total = Counter()
    shares = []
    for r in runs:
        c = Counter(x["top"] for x in r["rows"] if x["top"])
        n = sum(c.values()) or 1
        shares.append({a: v / n for a, v in c.items()})
        total.update({a: v / n for a, v in c.items()})
    acts = [a for a, _ in total.most_common(top)][::-1]
    fig, ax = plt.subplots(figsize=(11, 0.35 * len(acts) + 1))
    h = 0.8 / len(runs)
    y = np.arange(len(acts))
    for i, (r, s) in enumerate(zip(runs, shares)):
        ax.barh(y + i * h - 0.4 + h / 2, [100 * s.get(a, 0) for a in acts], height=h, color=COLORS[i],
                label=r["name"])
    ax.set_yticks(y, acts)
    ax.set_xlabel("% of person detections where this is the most likely action")
    ax.grid(axis="x", alpha=.3)
    ax.legend(loc="lower right")
    return fig_to_img(fig)


def chart_minutes(run, color_idx, top=6):
    c = Counter(x["top"] for x in run["rows"] if x["top"])
    acts = [a for a, _ in c.most_common(top)]
    per = defaultdict(Counter)
    moments = defaultdict(set)
    for x in run["rows"]:
        m = int(x["t"] // 60)
        moments[m].add(grid(x["t"]))
        if x["top"] in acts:
            per[m][x["top"]] += 1
    mins = sorted(moments)
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.stackplot(mins, [[per[m][a] / len(moments[m]) for m in mins] for a in acts], labels=acts, alpha=.85,
                 colors=plt.cm.tab10.colors[:len(acts)])
    ax.set_xlabel("minute in video")
    ax.set_ylabel("avg people doing it")
    ax.set_title(run["name"], loc="left", color=COLORS[color_idx], fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
    ax.grid(alpha=.3)
    return fig_to_img(fig)


def agreement(a, b, thr):
    A, B = defaultdict(list), defaultdict(list)
    for x in a["rows"]:
        A[grid(x["t"])].append(x)
    for x in b["rows"]:
        B[grid(x["t"])].append(x)
    common = sorted(set(A) & set(B))
    na = nb = matched = same = 0
    pairs = Counter()
    for t in common:
        pa, pb = A[t], B[t]
        na, nb = na + len(pa), nb + len(pb)
        if not pa or not pb:
            continue
        cost = np.array([[1 - iou(p["box"], q["box"]) for q in pb] for p in pa])
        for i, j in zip(*linear_sum_assignment(cost)):
            if 1 - cost[i, j] >= thr:
                matched += 1
                same += pa[i]["top"] == pb[j]["top"]
                if pa[i]["top"] != pb[j]["top"]:
                    pairs[(pa[i]["top"], pb[j]["top"])] += 1
    return {"moments": len(common), "na": na, "nb": nb, "matched": matched, "same": same,
            "pairs": pairs.most_common(8)}


def snapshots(video, runs, times, width=960):
    cap = cv2.VideoCapture(str(video))
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        H, W = frame.shape[:2]
        s = width / W
        frame = cv2.resize(frame, (width, int(H * s)))
        H, W = frame.shape[:2]
        for i, r in enumerate(runs):
            col = tuple(int(COLORS[i][k:k + 2], 16) for k in (5, 3, 1))   # hex -> BGR
            for x in r["rows"]:
                if abs(x["t"] - t) > 0.26:
                    continue
                x1, y1, x2, y2 = [int(v * d) for v, d in zip(x["box"], (W, H, W, H))]
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                label = f'{x["acts"][0][0]} {x["acts"][0][1]:.2f}' if x["acts"] else ""
                ty = y1 - 4 if i == 0 else y2 + 13
                cv2.putText(frame, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        out.append((t, base64.b64encode(buf.tobytes()).decode()))
    cap.release()
    return out


def hms(t):
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--snapshots", type=int, default=8)
    opts = ap.parse_args()

    runs = [load_run(d) for d in opts.runs]
    cap = cv2.VideoCapture(opts.video)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 25)
    cap.release()

    # summary table
    trs = []
    for i, r in enumerate(runs):
        moments = {grid(x["t"]) for x in r["rows"]}
        c = Counter(x["top"] for x in r["rows"] if x["top"])
        info = r["info"]
        speed = info.get("speed_x_realtime_per_gpu")
        trs.append(f'<tr><td style="color:{COLORS[i]}"><b>{html.escape(r["name"])}</b></td>'
                   f'<td>{len(r["rows"]):,}</td><td>{len(r["rows"]) / max(1, len(moments)):.1f}</td>'
                   f'<td>{", ".join(f"{html.escape(a)} ({100 * v / sum(c.values()):.0f}%)" for a, v in c.most_common(3))}</td>'
                   f'<td>{info.get("gpu_minutes", "?")} min ({speed}x real time)</td></tr>')
    summary = ("<table><tr><th>model</th><th>person detections<br>(every 0.5 s)</th><th>avg people<br>in view</th>"
               "<th>most common actions (top-1)</th><th>compute time</th></tr>" + "".join(trs) + "</table>")

    agree_html = ""
    if len(runs) >= 2:
        a, b = runs[0], runs[1]
        g = agreement(a, b, opts.iou)
        pairs = "".join(f"<tr><td>{html.escape(p)}</td><td>{html.escape(q)}</td><td>{n}</td></tr>"
                        for (p, q), n in g["pairs"])
        agree_html = f"""<h2>Do the models agree?</h2>
<p>At {g['moments']:,} shared moments, boxes from the two models were paired when they overlap (IoU &ge; {opts.iou}).</p>
<ul>
<li><b>{100 * g['matched'] / max(1, g['na']):.0f}%</b> of {a['name']}'s people were also found by {b['name']}, and
<b>{100 * g['matched'] / max(1, g['nb']):.0f}%</b> of {b['name']}'s people were also found by {a['name']}.</li>
<li>For people found by both, the most likely action is the same <b>{100 * g['same'] / max(1, g['matched']):.0f}%</b>
of the time ({g['same']:,} of {g['matched']:,}).</li></ul>
<p>Most common disagreements on the same person:</p>
<table><tr><th>{a['name']} says</th><th>{b['name']} says</th><th>times</th></tr>{pairs}</table>"""

    # snapshots: evenly spaced, nudged to the busiest moment in each slice
    counts = Counter(grid(x["t"]) for x in runs[0]["rows"])
    times = []
    edges = np.linspace(0, duration, opts.snapshots + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        cand = [(n, t) for t, n in counts.items() if lo <= t < hi]
        times.append(max(cand)[1] if cand else (lo + hi) / 2)
    snaps = snapshots(opts.video, runs, times)
    legend = " ".join(f'<span style="color:{COLORS[i]}">&#9632; {html.escape(r["name"])}</span>'
                      for i, r in enumerate(runs))
    snap_html = "".join(f'<figure><img src="data:image/jpeg;base64,{b}"><figcaption>{hms(t)}</figcaption></figure>'
                        for t, b in snaps)

    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Action detection report</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#202124;background:#fff}}
table{{border-collapse:collapse;margin:8px 0}}td,th{{border:1px solid #ddd;padding:6px 10px;text-align:left;font-size:14px}}
th{{background:#f4f4f4}}img{{max-width:100%}}figure{{margin:0 0 18px}}figcaption{{font-size:13px;color:#555}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(480px,1fr));gap:12px}}</style></head><body>
<h1>Action detection report: {html.escape(Path(opts.video).name)}</h1>
<p>{duration / 60:.1f} minutes of video. Models: {legend}. Actions are the 60 AVA classes the released models
predict (not yet fine-tuned on park footage).</p>
<h2>Summary</h2>{summary}
<h2>People in view over time</h2>{chart_people(runs, duration)}
<h2>What are people doing?</h2>{chart_actions(runs)}
<h2>Actions minute by minute</h2>{"".join(chart_minutes(r, i) for i, r in enumerate(runs))}
{agree_html}
<h2>Snapshots</h2><p>Busiest moment in each 1/{opts.snapshots} of the video. Boxes: {legend}
(label = most likely action and its probability).</p><div class="grid">{snap_html}</div>
</body></html>"""
    Path(opts.out).write_text(page, encoding="utf-8")
    print("report ->", opts.out)


if __name__ == "__main__":
    main()
