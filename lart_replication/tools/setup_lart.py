"""Fetch LART + PHALP (+ 4D-Humans), patch them, and rebuild every asset whose download link has died.

    python tools/setup_lart.py                 # code + assets for the AVA replication & notebook (any OS)
    python tools/setup_lart.py --demo          # + 4D-Humans and the MViT demo weights (Linux, ~2.4 GB)
    python tools/setup_lart.py --smpl /path/to/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl

Why this script exists (state of the upstream release as of Oct 2026):
  * LART's demo and training code download from people.eecs.berkeley.edu/~jathushan/..., which now
    returns 403. PHALP moved its files to www.cs.utexas.edu/~pavlakos/ (Feb 2026); LART was never updated.
  * Seven LART metadata files (AVA label maps, AVA val CSV, valid-class list, ...) exist nowhere anymore.
    We rebuild them from the official AVA v2.2 release and PHALP's class mapping (see build_lart_metadata).
  * lart_mvit.config (the demo model's config) is blocked by the new server; the checkpoint stores the
    exact same config in hyper_parameters, so we regenerate it from there.
  * PHALP auto-downloads the licensed SMPL model from an unofficial mirror (now 404, LART issue #34).
    We never do that: you register at https://smpl.is.tue.mpg.de and pass the file with --smpl.

Everything lands in lart_replication/third_party (code) and lart_replication/cache (assets; used as
$HOME/.cache by PHALP/LART via the HOME variable our entry points set). Both are git-ignored.
"""
import argparse
import csv
import io
import pickle
import re
import shutil
import sys
import types
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]           # lart_replication/
THIRD = ROOT / "third_party"
CACHE = ROOT / "cache" / ".cache"                    # == CACHE_DIR inside PHALP/LART (HOME=ROOT/cache)

REPOS = {  # name -> (github repo, pinned commit)
    "LART": ("brjathu/LART", "9d332425e8c28beb5702a2eaf57a635a1a203b1e"),        # 2023-12-29
    "PHALP": ("brjathu/PHALP", "96f7e6c09fb858ec3f597d59246c151ab4394bc3"),      # 2026-02-07 "Update Links"
    "4D-Humans": ("shubham-goel/4D-Humans", "efe18deff163b29dff87ddbd575fa29b716a356c"),  # 2026-02-07
}

UT = "https://www.cs.utexas.edu/~pavlakos/phalp"
PHALP_ASSETS = {  # file -> (url, cache subdir)   (same list PHALP downloads itself, all still alive)
    "head_faces.npy": (f"{UT}/3D/head_faces.npy", "phalp/3D"),
    "mean_std.npy": (f"{UT}/3D/mean_std.npy", "phalp/3D"),
    "smpl_mean_params.npz": (f"{UT}/3D/smpl_mean_params.npz", "phalp/3D"),
    "SMPL_to_J19.pkl": (f"{UT}/3D/SMPL_to_J19.pkl", "phalp/3D"),
    "texture.npz": (f"{UT}/3D/texture.npz", "phalp/3D"),
    "bmap_256.npy": (f"{UT}/bmap_256.npy", "phalp/3D"),
    "fmap_256.npy": (f"{UT}/fmap_256.npy", "phalp/3D"),
    "ava_labels.pkl": (f"{UT}/ava/ava_labels.pkl", "phalp/ava"),
    "ava_class_mapping.pkl": (f"{UT}/ava/ava_class_mappping.pkl", "phalp/ava"),
}
DEMO_ASSETS = {
    "hmar_v2_weights.pth": (f"{UT}/weights/hmar_v2_weights.pth", "phalp/weights"),
    "pose_predictor.pth": (f"{UT}/weights/pose_predictor_40006.ckpt", "phalp/weights"),
    "pose_predictor.yaml": (f"{UT}/weights/config_40006.yaml", "phalp/weights"),
    "lart_mvit.ckpt": (f"{UT}/ava/lart_mvit.ckpt", "phalp/ava"),     # 582 MB, LART with MViT appearance
    "mvit.yaml": (f"{UT}/ava/mvit.yaml", "phalp/ava"),
    "mvit.pyth": (f"{UT}/ava/mvit.pyth", "phalp/ava"),               # 1.7 GB, MViTv2 AVA feature extractor
}
AVA_ZIP = "https://research.google.com/ava/download/ava_v2.2.zip"     # official AVA v2.2 annotations
HIERA_CKPT = "https://www.dropbox.com/s/aja2sfo9ge8hqxf/lart_hiera_001.tar.gz?dl=1"  # 530 MB, the 45.1 mAP model

WEIGHTS_ONLY_FALSE = '__import__("functools").partial(torch.load, weights_only=False)('

# (repo, file, regex, replacement inserted literally, reason)
PATCHES = [
    ("LART", "lart/datamodules/components/phalp_action_datatset.py",
     r"self\.track2video = np\.array\(self\.track2video\)",
     "self.track2video = np.asarray(self.track2video, dtype=object)",
     "NumPy >= 1.24 refuses ragged arrays (LART issues #20, #30)"),
    ("LART", "lart/datamodules/components/phalp_action_datatset.py",
     r'path_npy = "data/_TMP/"\+"fast_"\.join\(root_dir\.split\("/"\)\)\+"\.npy"',
     'path_npy = "data/_TMP/" + "fast_".join(__import__("os").path.normpath(root_dir).replace(":", "").split(__import__("os").sep)[-2:]) + ".npy"',
     "the file-list cache name embedded the absolute path (invalid on Windows: 'C:\\', spaces)"),
    # torch.load(...) -> partial(torch.load, weights_only=False)(...): valid whatever the arguments are
    # (some contain nested parentheses), and the pattern cannot match the patched text again.
    ("LART", "lart/train.py", r"torch\.load\(", WEIGHTS_ONLY_FALSE,
     "PyTorch >= 2.6 defaults to weights_only=True; Lightning checkpoints pickle configs"),
    ("LART", "lart/utils/wrapper_phalp.py", r"torch\.load\(path\)", WEIGHTS_ONLY_FALSE + 'path, map_location="cpu")',
     "same as above (+ load on CPU first)"),
    ("LART", "lart/utils/ema.py", r"torch\.load\(", WEIGHTS_ONLY_FALSE, "same as above"),
    ("LART", "lart/ActivityNet/Evaluation/ava/metrics.py", r"np\.float\b(?![_0-9])", "np.float64",
     "np.float was removed in NumPy 1.24 (AVA evaluator)"),
    ("LART", "lart/ActivityNet/Evaluation/ava/metrics.py", r"np\.bool\b(?!_)", "np.bool_",
     "np.bool was removed in NumPy 1.24 (AVA evaluator)"),
    ("LART", "lart/evaluators/ava.py", r"f = open\(csv_path, 'w'\)", "f = open(csv_path, 'w', newline='')",
     "csv.writer without newline='' writes blank rows on Windows -> AVA evaluator assertion"),
    # MViT appearance features run attention over the whole 40-frame clip: ~15 GB at 312 px in fp32 (fine on the
    # authors' 40 GB GPUs or a 24 GB Titan RTX). LART_MVIT_FP16=1 / LART_MVIT_CROP=256 make it fit 8 GB laptops;
    # unset, the behaviour is unchanged.
    ("LART", "lart/utils/wrapper_phalp.py",
     r'cfg\.TEST\.CHECKPOINT_FILE_PATH=f"\{CACHE_DIR\}/phalp/ava/mvit\.pyth"(?!; )',
     'cfg.TEST.CHECKPOINT_FILE_PATH=f"{CACHE_DIR}/phalp/ava/mvit.pyth"; '
     'cfg.DATA.TEST_CROP_SIZE = int(__import__("os").environ.get("LART_MVIT_CROP", cfg.DATA.TEST_CROP_SIZE))',
     "optional smaller MViT input (memory)"),
    ("LART", "lart/utils/wrapper_phalp.py", r"with torch\.no_grad\(\):(?=\n\s+task_\s+= SlowFastWrapper)",
     'with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, '
     'enabled=__import__("os").environ.get("LART_MVIT_FP16") == "1"):',
     "optional fp16 MViT (memory)"),
    ("LART", "lart/utils/wrapper_phalp.py", r"task_\.action_preds\[0\]\n", "task_.action_preds[0].float()\n",
     "fp16 outputs back to fp32"),
    ("LART", "lart/utils/wrapper_phalp.py", r"task_\.action_preds\[1\]\n", "task_.action_preds[1].float()\n",
     "fp16 outputs back to fp32"),
    ("LART", "lart/models/lart.py", r"https://people\.eecs\.berkeley\.edu/~jathushan/projects/lart/",
     "https://www.cs.utexas.edu/~pavlakos/lart-files-moved/",
     "dead host; the files are rebuilt locally by this script, so the URL is only a fallback that fails loudly"),
]


# ----------------------------------------------------------------------------------------- helpers
def download(url, dest, quiet=False):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if not quiet:
        print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        total, done = int(r.headers.get("content-length", 0)), 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total > 50e6 and not quiet:
                print(f"\r    {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    if total > 50e6 and not quiet:
        print()
    tmp.replace(dest)
    return dest


def fetch_repo(name, force=False):
    repo, commit = REPOS[name]
    dst = THIRD / name
    if dst.exists() and not force:
        print(f"[{name}] present ({commit[:8]})")
        return dst
    shutil.rmtree(dst, ignore_errors=True)
    print(f"[{name}] downloading {repo}@{commit[:8]}")
    data = urllib.request.urlopen(f"https://github.com/{repo}/archive/{commit}.zip").read()
    THIRD.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        top = z.namelist()[0].split("/")[0]
        z.extractall(THIRD)
    (THIRD / top).rename(dst)
    return dst


def apply_patches():
    print("[patch] applying patches")
    for repo, rel, pattern, repl, why in PATCHES:
        p = THIRD / repo / rel
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        new, n = re.subn(pattern, lambda m, r=repl: r, text)    # literal: no backslash/group processing
        if n:
            p.write_text(new, encoding="utf-8")
        print(f"  {repo}/{rel}: {n} change(s) - {why}")
    import py_compile                     # a patch must never leave invalid Python behind
    for repo, rel, *_ in PATCHES:
        if (THIRD / repo / rel).exists():
            py_compile.compile(str(THIRD / repo / rel), doraise=True)
    # LART's train.py finds its root via pyrootutils, which looks for .git or pyproject.toml.
    marker = THIRD / "LART" / "pyproject.toml"
    if not marker.exists():
        marker.write_text("# marker for pyrootutils (added by lart_replication/tools/setup_lart.py)\n")


# ------------------------------------------------------------------------- rebuilt LART metadata
def build_lart_metadata():
    """Recreate the 7 files LART downloaded from the dead Berkeley URL, from official sources."""
    out = CACHE / "lart"
    out.mkdir(parents=True, exist_ok=True)
    zpath = download(AVA_ZIP, ROOT / "cache" / "ava_v2.2.zip")
    z = zipfile.ZipFile(zpath)
    # 1-3: official AVA v2.2 files (label maps, val ground truth)
    (out / "ava_val_v2.2.csv").write_bytes(z.read("ava_val_v2.2.csv"))
    (out / "ava_action_list_v2.2_for_activitynet_2019.pbtxt").write_bytes(z.read("ava_action_list_v2.2_for_activitynet_2019.pbtxt"))
    full = z.read("ava_action_list_v2.2.pbtxt")
    (out / "ava_action_list_v2.2.pbtxt").write_bytes(full)
    (out / "ava_action_list.pbtxt").write_bytes(full)          # LART reads the 80-class map under this name
    # 4: ava_valid_classes.npy = AVA ids of the 60 evaluated classes, in LART's model-output order.
    #    PHALP's (still online) ava_class_mapping.pkl maps model index 1..60 -> AVA id; we invert that.
    import joblib
    mapping = joblib.load(CACHE / "phalp" / "ava" / "ava_class_mapping.pkl")
    valid = np.array([mapping[i] for i in range(1, 61)], dtype=np.int64)
    eval_ids = sorted(int(x) for x in re.findall(rb"id: (\d+)", z.read("ava_action_list_v2.2_for_activitynet_2019.pbtxt")))
    assert sorted(valid.tolist()) == eval_ids, "class mapping does not match the official 60 AVA classes"
    np.save(out / "ava_valid_classes.npy", valid)
    # 5: class_sum.pkl = number of training boxes per class name (only used for plots, utils_plot.py)
    names = dict(re.findall(r'name: "([^"]+)"\s*\n\s*(?:label_)?id: (\d+)', full.decode()))
    id2name = {int(v): k for k, v in names.items()}
    counts = Counter(int(row[6]) for row in csv.reader(io.StringIO(z.read("ava_train_v2.2.csv").decode())))
    joblib.dump({id2name[i]: counts.get(i, 0) for i in id2name}, out / "class_sum.pkl")
    # 6: kinetics_annot_train.pkl = {youtube_id: label} for Kinetics-400 *training* tracks. Only needed to
    #    TRAIN with kinetics_train data; evaluation never reads it. Empty placeholder unless already present.
    kin = out / "kinetics_annot_train.pkl"
    if not kin.exists():
        joblib.dump({}, kin)
    print(f"[metadata] rebuilt LART metadata in {out}")


# --------------------------------------------------------------------------------------- SMPL
class _ChumpyStub:
    """Stands in for chumpy objects while unpickling SMPL files (chumpy is broken on NumPy >= 1.24)."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"x": state})

    def to_numpy(self):
        for key in ("x", "r", "_x", "a"):
            if key in self.__dict__:
                v = self.__dict__[key]
                return v.to_numpy() if isinstance(v, _ChumpyStub) else np.asarray(v)
        raise ValueError(f"cannot convert chumpy object with keys {list(self.__dict__)}")


class _SMPLUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _ChumpyStub
        if module == "scipy.sparse.csc":            # old scipy module path
            module = "scipy.sparse"
        return super().find_class(module, name)


def install_smpl(src):
    """Convert a licensed SMPL neutral model to a plain-numpy pickle and place it where PHALP/4DHumans look."""
    src = Path(src)
    with open(src, "rb") as f:
        data = _SMPLUnpickler(f, encoding="latin1").load()
    clean = {k: (v.to_numpy() if isinstance(v, _ChumpyStub) else v) for k, v in data.items()}
    for k in ("v_template", "shapedirs", "posedirs", "J_regressor", "weights", "f", "kintree_table"):
        assert k in clean, f"{src.name} does not look like an SMPL model (missing {k})"
    print(f"[smpl] {src.name}: shapedirs {np.asarray(clean['shapedirs']).shape}, v_template {np.asarray(clean['v_template']).shape}")
    targets = [CACHE / "phalp/3D/models/smpl/SMPL_NEUTRAL.pkl",          # PHALP / LART
               CACHE / "4DHumans/data/smpl/SMPL_NEUTRAL.pkl"]             # HMR 2.0 (4D-Humans)
    for t in targets:
        t.parent.mkdir(parents=True, exist_ok=True)
        with open(t, "wb") as f:
            pickle.dump(clean, f)
        print(f"  -> {t}")


# ------------------------------------------------------------------------------ demo extras
def rebuild_mvit_config():
    """lart_mvit.config is 403 on the new host; the checkpoint carries the identical config."""
    import torch
    from omegaconf import OmegaConf
    ckpt = CACHE / "phalp/ava/lart_mvit.ckpt"
    cfg = torch.load(ckpt, map_location="cpu", weights_only=False)["hyper_parameters"]["cfg"]
    OmegaConf.save(OmegaConf.create({"configs": OmegaConf.to_container(cfg)}), CACHE / "phalp/ava/lart_mvit.config")
    print("[demo] rebuilt lart_mvit.config from the checkpoint's hyper_parameters")


def fetch_hiera_checkpoint():
    dst = ROOT / "logs" / "LART_Hiera" / "0" / "checkpoints" / "epoch_002-EMA.ckpt"
    if dst.exists():
        print(f"[ckpt] {dst.relative_to(ROOT)} present")
        return
    import tarfile
    tgz = download(HIERA_CKPT, ROOT / "cache" / "lart_hiera_001.tar.gz")
    with tarfile.open(tgz) as t:
        t.extractall(ROOT)          # creates logs/LART_Hiera/0/{checkpoints,results,.hydra}
    print(f"[ckpt] extracted LART-Hiera checkpoint to {dst.parent}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="also fetch 4D-Humans + video-demo weights (~2.4 GB)")
    ap.add_argument("--smpl", help="path to the SMPL neutral .pkl you downloaded after registering")
    ap.add_argument("--no-checkpoint", action="store_true", help="skip the 530 MB LART-Hiera checkpoint")
    ap.add_argument("--force", action="store_true", help="re-download and re-patch the code")
    args = ap.parse_args()

    for name in ("LART", "PHALP") + (("4D-Humans",) if args.demo else ()):
        fetch_repo(name, args.force)
    apply_patches()
    print("[assets] PHALP assets")
    for fname, (url, sub) in PHALP_ASSETS.items():
        download(url, CACHE / sub / fname, quiet=True)
    build_lart_metadata()
    if not args.no_checkpoint:
        fetch_hiera_checkpoint()
    if args.demo:
        print("[demo] demo weights")
        for fname, (url, sub) in DEMO_ASSETS.items():
            download(url, CACHE / sub / fname)
        rebuild_mvit_config()
    if args.smpl:
        install_smpl(args.smpl)
    elif not (CACHE / "phalp/3D/models/smpl/SMPL_NEUTRAL.pkl").exists():
        print("\n[smpl] NOT installed. Register at https://smpl.is.tue.mpg.de, download 'SMPL for Python' "
              "(v1.1.0, female/male/neutral 300 shapes) and re-run with\n"
              "       --smpl <path>/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl")
    print("done.")


if __name__ == "__main__":
    sys.exit(main())
