# 2. Replicating LART: setup, what was broken, and how to run it

## 2.1 Two levels of replication

| Level | What | Where it runs | Needs |
|---|---|---|---|
| **A. The paper's number** | evaluate the released LART-Hiera checkpoint on the released AVA val tracks → **45.1 mAP** | Windows or Linux, any CUDA GPU (8 GB is plenty) | 28 GB download, SMPL file |
| **A'. The paper's claims** | ablations: temporal context, other people, pose vs. appearance | same | same |
| **B. New videos** | detect → 3D-track → MViT features → LART → rendered video | **Linux**, GPU ≥ 12–16 GB, ~32 GB RAM | `setup_linux.sh`, +8 GB of weights |

Level A is what "replicate the paper" means for a course: the same checkpoint, data and evaluation code give the
same number. Level B shows the method working on footage it has never seen, such as your park videos.

## 2.2 Setup

### Windows / Linux, level A (what was tested on this machine)
```bash
# from the CVProject root, in the project's .venv (Python 3.11, torch 2.11 + cu128)
pip install omegaconf hydra-core lightning einops timm torchmetrics rich pyrootutils joblib scikit-learn dill \
            "smplx==0.1.28" fvcore iopath av "git+https://github.com/facebookresearch/pytorchvideo.git"
cd lart_replication
python tools/setup_lart.py --smpl SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
python tools/download_ava_val.py          # 27.4 GB stream-extracted, then marked complete
python eval_ava.py                        # full replication
```

### Linux, levels A + B
```bash
cd lart_replication
bash setup_linux.sh --smpl /path/to/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
conda activate lart
python tools/download_ava_val.py && python eval_ava.py
PYOPENGL_PLATFORM=egl python run_demo.py video.source=third_party/LART/assets/jump.mp4
```
`setup_linux.sh` defaults to the authors' versions (PyTorch 2.0, CUDA 11.7). It compiles detectron2 and the
neural renderer against whatever PyTorch is installed (`--no-build-isolation`), so **nvcc must match torch's
CUDA version**; the script checks this. For very new GPUs (H100, RTX 50xx), pass a newer `TORCH_SPEC` and
`TORCH_INDEX` (see the script header).

### The SMPL body model (licensed)
Register at <https://smpl.is.tue.mpg.de> (free for research), download **SMPL for Python v1.1.0** (the
"female/male/neutral, 300 shapes" archive), and pass the **neutral** `.pkl` with `--smpl`.
`tools/setup_lart.py` converts it to plain NumPy, without needing `chumpy`, which is broken on modern NumPy,
and puts it where PHALP and HMR 2.0 look for it. **Never commit it:** it's git-ignored.

## 2.3 What was broken upstream, and what we did

The authors' code dates from 2023 and its web assets have moved since. Everything we change is in one place,
`tools/setup_lart.py` (`PATCHES` plus the rebuild functions), applied to pinned commits.

| # | Problem | Effect | Fix |
|---|---|---|---|
| 1 | Demo + training download from `people.eecs.berkeley.edu/~jathushan/…` | **403**, nothing can be downloaded | PHALP's files moved to `www.cs.utexas.edu/~pavlakos/` (Feb 2026); we fetch from there |
| 2 | 7 LART metadata files (AVA label maps, val CSV, valid-class list, class counts, Kinetics labels) exist **nowhere** anymore, not even in the Wayback Machine | evaluation can't start | rebuilt from the **official AVA v2.2 release** (Google) + PHALP's surviving `ava_class_mapping.pkl`; the valid-class list is asserted equal to the official 60 classes |
| 3 | `lart_mvit.config` (demo model config) is blocked (403) on the new host | demo can't build the model | regenerated **exactly** from the checkpoint's `hyper_parameters['cfg']` |
| 4 | SMPL auto-downloaded from an unofficial GitHub mirror (now 404, issue #34), and needs `chumpy` | license problem + crash | user-provided SMPL, converted without chumpy |
| 5 | `np.array(ragged list)` in the dataset | `ValueError: inhomogeneous shape` on NumPy ≥ 1.24 (issues #20, #30) | `np.asarray(..., dtype=object)` |
| 6 | `np.float`, `np.bool` in the AVA evaluator | `AttributeError` on NumPy ≥ 1.24 | `np.float64`, `np.bool_` |
| 7 | `torch.load` default changed in PyTorch 2.6 | checkpoints don't load | `weights_only=False` (patched so that nested arguments stay valid) |
| 8 | File-list cache named after the absolute data path | invalid filename on Windows (`C:\…`, spaces) | name from the last two path parts |
| 9 | `csv.writer` on a file opened without `newline=''` | blank rows on Windows → evaluator assertion | `newline=''` |
| 10 | **Stale file-list cache after a partial download** (issue #29) | silently **low mAP** even after re-downloading | `download_ava_val.py` writes a completion marker; `eval_ava.py` refuses incomplete data and deletes stale caches |
| 11 | SLURM/submitit launcher + `pyrootutils` root detection | can't run on a single machine / from a zip | `eval_ava.py` uses the same classes through `Trainer.validate()`; `setup_lart.py` adds the root marker |
| 12 | Windows Smart App Control blocked PyAV's DLL on first load | `ImportError: DLL load failed` | it went away on retry; if it persists, try another `av` version |

## 2.4 Running the evaluation

```bash
python eval_ava.py                                   # full AVA v2.2 val (64 videos) - the paper's 45.1
python eval_ava.py --videos 8                        # quicker: first 8 videos, scored on their own GT
python eval_ava.py --videos 8 --ablation context_pm0     # only the key frame visible (no tracking)
python eval_ava.py --videos 8 --ablation no_others       # only the person of interest (no interactions)
python eval_ava.py --videos 8 --ablation no_appearance   # pose + joints only (out-of-distribution test)
python eval_ava.py --videos 8 --test-type track.fullframe@GT   # perfect labels on PHALP boxes: the box ceiling
```
Outputs land in `outputs/<name>/`: `ava_val.csv` (every scored detection), `summary.json` (mAP and per-class AP),
and `slowfast/` (per-track predictions).

**Our results:** see the README of this folder and the notebook, where they are filled in from real runs.

## 2.5 Things to know when reading the numbers

- **Boxes come from PHALP, not from ground truth.** AVA's protocol scores against GT boxes, so detection
  misses and loose boxes cost mAP. The `@GT` run measures this ceiling.
- **`avg.6`:** predictions are averaged over ±6 frames (0.2 s at 30 fps) around the key frame, as in the paper.
- **Subsets** (`--videos K`) are scored only against the ground truth of those K videos. They're consistent
  for comparing ablations, but individual videos vary a lot; only the full run is comparable to 45.1.
- **The data are features, not pixels:** the released tracks already contain the Hiera appearance vectors, so we
  can't swap the appearance model. That would need re-running Hiera on the AVA videos.
