# LART replication: *On the Benefits of 3D Pose and Tracking for Human Action Recognition* (CVPR 2023)

This folder replicates **LART** (Rajasegaran, Pavlakos, Kanazawa, Feichtenhofer, Malik;
[paper](https://arxiv.org/abs/2304.01199), [code](https://github.com/brjathu/LART)), the second paper of this
project after [TubeR](../README.md). LART recognises actions by **following each person** (3D tracking with PHALP
and HMR 2.0) and reasoning over their **3D pose + appearance tracks** with a transformer. It reaches **45.1 mAP**
on AVA v2.2, against TubeR's 31.1.

| Path | What |
|---|---|
| [notebooks/LART_walkthrough.ipynb](notebooks/LART_walkthrough.ipynb) | **Start here.** The paper explained on the real model and data: tracklets, 3D meshes, tokens, the replication, and tests of the paper's claims |
| [docs/01_lart_explained.md](docs/01_lart_explained.md) | the method, the paper's tables, and LART vs. TubeR |
| [docs/02_replication_guide.md](docs/02_replication_guide.md) | setup (Windows/Linux), **12 things that were broken upstream and how they're fixed**, running the evaluation |
| [docs/03_park_cctv_with_lart.md](docs/03_park_cctv_with_lart.md) | running LART on park footage, fine-tuning on your own classes, a fair comparison with TubeR |
| [docs/04_related_work_and_forks.md](docs/04_related_work_and_forks.md) | upstream repos, forks with real work (hockey, lite), issues, related methods |
| [eval_ava.py](eval_ava.py) | the AVA v2.2 evaluation (authors' code, single GPU) with subsets, ablations and seeds |
| [run_demo.py](run_demo.py) | LART on any video: detect → 3D track → MViT → actions → rendered video (**Linux**) |
| [tools/setup_lart.py](tools/setup_lart.py) | pins + patches LART/PHALP/4D-Humans, rebuilds dead assets, installs your SMPL |
| [tools/download_ava_val.py](tools/download_ava_val.py) | the authors' AVA val tracks (27.4 GB → 97,148 tracklets, 51 GB on disk) |
| [setup_linux.sh](setup_linux.sh) | full Linux environment including detectron2 and the renderer, for the video demo |
| [lartlab/](lartlab/) | our helpers: environment, evaluation and ablations, "other people" analysis, labels |

Git-ignored: `data/` (tracks), `logs/` (checkpoint), `cache/` (assets), `third_party/` (upstream code),
`outputs/`, and your licensed SMPL files.

## Quick start (Windows or Linux, for the AVA replication and the notebook)
```bash
# in the project .venv (see ../requirements.txt), plus LART's pure-Python deps:
pip install omegaconf hydra-core lightning einops timm torchmetrics rich pyrootutils joblib scikit-learn dill \
            "smplx==0.1.28" fvcore iopath av "git+https://github.com/facebookresearch/pytorchvideo.git"
cd lart_replication
python tools/setup_lart.py --smpl <path>/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl   # SMPL: register at smpl.is.tue.mpg.de
python tools/download_ava_val.py
python eval_ava.py                       # ~45.1 mAP
jupyter lab notebooks/LART_walkthrough.ipynb
```
For the video demo on a Linux GPU server: `bash setup_linux.sh --smpl <path>` then `python run_demo.py video.source=<video>`.

## Results

| | paper | ours (this repo, RTX 5060 laptop) |
|---|---|---|
| AVA v2.2 frame-mAP, all 64 val videos (97,148 tracks) | 45.1 | **45.11** (authors' own saved result: 45.105) |
| person movement / object manipulation / person interaction | 63.8 / 37.3 / 45.7 | 63.8 / 37.3 / 45.7 |
| per-class difference vs. the authors' saved result | | 0.04 mean, 0.36 max (60 classes) |

Runtime: 2 h 24 min on one RTX 5060 Laptop GPU.
