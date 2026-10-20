"""Model backends for run.py. Each backend processes ONE chunk on ONE GPU in a subprocess, then reads that chunk's
predictions back as a list of PersonPrediction rows in a common format, so TubeR and LART results are comparable:

    time_sec (within the chunk), person_id, person_score, box (x1, y1, x2, y2 normalised to 0..1),
    actions: [(official AVA action name, probability), ...] sorted by probability
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LART_DIR = ROOT / "lart_replication"


@dataclass
class PersonPrediction:
    time_sec: float
    person_id: str
    person_score: float
    box: tuple
    actions: list = field(default_factory=list)


def _run(cmd, env, log_path, cwd):
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(" ".join(map(str, cmd)) + "\n\n")
        log.flush()
        return subprocess.run(list(map(str, cmd)), env=env, stdout=log, stderr=subprocess.STDOUT, cwd=cwd).returncode


# ------------------------------------------------------------------------------------------------------- TubeR
class TubeRBackend:
    name = "tuber"
    default_chunk_min = 10

    def __init__(self, opts):
        self.opts = opts

    def run_chunk(self, chunk, out_dir, gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        cmd = [self.opts.python, ROOT / "infer_video.py", "--config", self.opts.tuber_config,
               "--checkpoint", self.opts.tuber_checkpoint, "--video", chunk, "--out", out_dir / "pred",
               "--stride-sec", self.opts.sample_sec,
               # keep low-probability actions too; run.py keeps the top-k per person
               "CONFIG.FINETUNE.VIS_ACTION_THRESH", "0.02"]
        if not self.opts.render:
            cmd.insert(cmd.index("--stride-sec"), "--no-video")
        rc = _run(cmd, env, out_dir / "log.txt", ROOT)
        return rc == 0 and (out_dir / "pred.csv").exists()

    def read_chunk(self, out_dir, video_info):
        people = {}
        with open(out_dir / "pred.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = (float(r["time_sec"]), r["person_id_in_frame"])
                p = people.setdefault(key, PersonPrediction(
                    key[0], f"t{key[0]:.1f}_p{key[1]}", float(r["person_score"]),
                    tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2"))))
                if r["action"]:
                    p.actions.append((r["action"], float(r["score"])))
        for p in people.values():
            p.actions.sort(key=lambda a: -a[1])
        return list(people.values())


# -------------------------------------------------------------------------------------------------------- LART
class LARTBackend:
    name = "lart"
    default_chunk_min = 5

    def __init__(self, opts):
        self.opts = opts
        if sys.platform.startswith("win"):
            raise SystemExit("The LART video pipeline (detectron2, PHALP, HMR 2.0) runs on Linux only - see "
                             "lart_replication/setup_linux.sh. (The AVA replication/notebook do run on Windows.)")

    def run_chunk(self, chunk, out_dir, gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), LART_RENDER="1" if self.opts.render else "0",
                   PYOPENGL_PLATFORM=os.environ.get("PYOPENGL_PLATFORM", "egl"))
        cmd = [self.opts.python, LART_DIR / "run_demo.py", f"video.source={chunk}", f"video.output_dir={out_dir}",
               "video.end_frame=-1",                      # PHALP's default stops at frame 1300 (~43 s)!
               f"phalp.detector={self.opts.lart_detector}", f"hydra.run.dir={out_dir / 'hydra'}"]
        rc = _run(cmd, env, out_dir / "log.txt", LART_DIR)
        demo = out_dir / "_DEMO"
        if demo.exists() and not self.opts.keep_frames:
            shutil.rmtree(demo, ignore_errors=True)       # PHALP extracts every frame as JPEG - free the disk
        return rc == 0 and any((out_dir / "results_temporal").glob("*.pkl"))

    def read_chunk(self, out_dir, video_info):
        import joblib
        import numpy as np
        sys.path.insert(0, str(LART_DIR))
        import lartlab  # noqa: F401  (cache paths)
        from phalp.configs.base import CACHE_DIR
        names = joblib.load(f"{CACHE_DIR}/phalp/ava/ava_labels.pkl")[0]          # {AVA id: official name}
        model_to_ava = joblib.load(f"{CACHE_DIR}/phalp/ava/ava_class_mapping.pkl")  # {1..60: AVA id}
        fps, W, H = video_info["fps"], video_info["width"], video_info["height"]
        step = max(1, int(round(self.opts.sample_sec * fps)))   # same temporal sampling as TubeR
        rows = []
        for pkl in sorted((out_dir / "results_temporal").glob("*.pkl")):
            d = joblib.load(pkl)
            for fi, fkey in enumerate(sorted(d)):
                if fi % step:
                    continue
                fr = d[fkey]
                acts = fr.get("ava_action", {})
                confs = fr.get("conf", [])
                for k, tid in enumerate(fr.get("tid", [])):
                    if tid not in acts:
                        continue
                    p = 1 / (1 + np.exp(-np.asarray(acts[tid], dtype=float).reshape(-1)))
                    x, y, w, h = [float(v) for v in fr["bbox"][k]]
                    order = np.argsort(-p)
                    rows.append(PersonPrediction(
                        fi / fps, f"track{tid}", float(confs[k]) if k < len(confs) else float("nan"),
                        (x / W, y / H, (x + w) / W, (y + h) / H),
                        [(names.get(model_to_ava[int(j) + 1], str(j)), float(p[j])) for j in order[:10]]))
        return rows


BACKENDS = {"tuber": TubeRBackend, "lart": LARTBackend}


def chunk_done(out_dir):
    f = out_dir / "DONE.json"
    return json.loads(f.read_text()) if f.exists() else None


def process_chunk(backend, chunk, start, out_dir, gpu):
    """Run one chunk unless already done; record timing in DONE.json (this is what makes runs resumable)."""
    info = chunk_done(out_dir)
    if info:
        return {**info, "cached": True}
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ok = backend.run_chunk(chunk, out_dir, gpu)
    secs = time.time() - t0
    if not ok:
        return {"chunk": chunk.name, "error": f"failed - see {out_dir / 'log.txt'}"}
    from .video import probe
    v = probe(chunk)
    info = {"chunk": chunk.name, "start_sec": start, "duration_sec": round(v["duration"], 2), "gpu": gpu,
            "seconds": round(secs, 1), "speed_x_realtime": round(v["duration"] / secs, 3) if secs else None}
    (out_dir / "DONE.json").write_text(json.dumps(info))
    return info
