# 4. LART: code, forks, the surrounding ecosystem

(Checked October 2026.)

## 4.1 Official code and its building blocks

| Repo | Role | Notes |
|---|---|---|
| [brjathu/LART](https://github.com/brjathu/LART) | the action model, training + eval | ~290 stars, 34 forks; last commit Dec 2023; **no license file**, so strictly all rights reserved: fine for coursework, but don't redistribute the code |
| [brjathu/PHALP](https://github.com/brjathu/PHALP) | 3D tracker (detection, HMAR appearance, association), renderer, LART post-processing | MIT; assets moved to `www.cs.utexas.edu/~pavlakos/phalp/` in Feb 2026 |
| [shubham-goel/4D-Humans](https://github.com/shubham-goel/4D-Humans) | HMR 2.0: ViT-based 3D human mesh recovery, used inside PHALP for the demo | needs SMPL |
| [brjathu/SlowFast](https://github.com/brjathu/SlowFast) | fork of PySlowFast used to extract MViT appearance features in the demo | |
| [facebookresearch/hiera](https://github.com/facebookresearch/hiera) | appearance backbone of the 45.1 model | the Hiera feature extractor used for the released tracks is not part of LART's demo |
| [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2) | ViTDet-H person detector (PHALP) | built from source |

## 4.2 Forks with their own work

| Fork | What they did | Why it's interesting |
|---|---|---|
| [howird/lart](https://github.com/howird/lart) (Jul 2026) | `uv` setup, **hockey action recognition**, an HMM model, balanced sampler, class-weighted loss | the closest analogue to adapting LART to a new domain (your parks) |
| [hoangle209/LART](https://github.com/hoangle209/LART) (2024) | "lart_lite" (a smaller model), 2D-joint config, a cleaner trainer/validator/dataloader for AVA | lighter variants, plus another take on the data pipeline |
| [michaelmunje/LART](https://github.com/michaelmunje/LART) (2025) | custom demo file, changed hyper-parameters, run script | a recent demo setup |
| [HoangfVx/LART](https://github.com/HoangfVx/LART) (2023) | download-script fixes | |

## 4.3 Upstream issues worth reading

#12 (fine-tuning on a custom dataset, with the authors' answer), #16 (missing `apperance_index` in custom tracks),
#29 (stale cache → low mAP), #30/#20 (NumPy), #27/#14/#7 (installation, detectron2, Windows),
#8/#21/#31 (CUDA out of memory in the demo), #32 (request for the LART-pose checkpoint: **not released**),
#33 (missing PHALP assets during evaluation), #34 (dead SMPL link).

## 4.4 Related methods

- **TubeR** (CVPR 2022), the first part of this project: end-to-end, Eulerian. 31.1 mAP.
- **Hiera** (ICML 2023) and **VideoMAE / VideoMAE V2**, **MaskFeat**: strong Eulerian backbones; LART's
  appearance stream comes from these.
- **ACAR-Net** (CVPR 2021): actor–context–actor relations. LART models relations only through track identity.
- **PoTion** (CVPR 2018), **JMRN**: earlier pose-based action recognition (2D keypoints). LART-pose beats them by about 10 mAP.
- **STMixer** (CVPR 2023), **EVAD** (ICCV 2023): one-stage detectors. They're contemporaries of LART on AVA,
  and simpler to deploy.
- **4DHumans / PHALP** (ICCV 2023 / CVPR 2022): the tracking side, a strong person tracker on its own.

## 4.5 Datasets

The released LART data: about 1.5 M PHALP tracklets on **AVA** (train ~93 GB, val ~27 GB) and **Kinetics-400**
(~210 GB, with pseudo-labels). See `tools/download_ava_val.py` and upstream `scripts/download_data.sh`.
For surveillance-like footage see MEVA and VIRAT in [../../docs/05_related_work_and_forks.md](../../docs/05_related_work_and_forks.md).
