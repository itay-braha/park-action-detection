# 5. Related work, forks and resources

(Checked in September 2026. Star counts and dates are from the GitHub API at that time.)

## 5.1 TubeR itself

| What | Link | Notes |
|---|---|---|
| Paper (arXiv) | <https://arxiv.org/abs/2104.00969> | CVPR 2022, Amazon/AWS AI Labs + University of Amsterdam + Rutgers |
| Paper (CVF open access) | [CVPR 2022 PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhao_TubeR_Tubelet_Transformer_for_Video_Action_Detection_CVPR_2022_paper.pdf) | |
| Official code | <https://github.com/amazon-science/tubelet-transformer> | ~96 stars, 22 forks, Apache-2.0; one real commit (Aug 2022), never updated since |
| Poster | [CVPR_TubeR_Poster.pdf](https://jovapo.github.io/publications/2022-6-CVPR-TubeR/CVPR_TubeR_Poster.pdf) | a good one-page summary |
| Upstream issues | <https://github.com/amazon-science/tubelet-transformer/issues> | worth reading, see below |

**Upstream issues that matter for you:**
- #27 "when training custom dataset, both train loss and val loss hardly decreased!" (closed Jan 2026 with
  no fix posted). The symptoms match the hard-coded 0.8 person gate and/or the `avg` vs `decode`
  mismatch; both are handled here.
- #21 "path doesnt exist .../frames/{}/", the unformatted `DATA_PATH` bug (fixed here).
- #17 "Inference ... mAP 0.00001", very likely the `TEMPORAL_DS_STRATEGY` mismatch we found.
- #26 "Code here don't match the paper", #25 "Cannot reproduce training results", #18/#23 on DETR
  pretraining details, #15 on the long-term-memory model (never released).

## 5.2 Forks of TubeR with real changes

Most of the 22 forks are unchanged copies. These are the ones with their own commits:

| Fork | What they did | Useful for you |
|---|---|---|
| [salmank255/tubelet-transformer](https://github.com/salmank255/tubelet-transformer) | Adapted TubeR to the **ROAD / ROAD++ road-event datasets** (autonomous-driving video: agents, actions, locations). Adds `datasets/road_frames.py`, `train_tuber_road.py`, `train_tuber_roadpp.py`, ROAD label maps and eval, and fixes the frame-path bug | **The closest existing example of TubeR on a custom, non-AVA dataset.** Compare their dataset class with ours. The fork's author appears to be from the ROAD++ challenge team |
| [sylee-skhu/tubelet-transformer](https://github.com/sylee-skhu/tubelet-transformer) | "First custom code" (May 2024): custom CSN-50 AVA 2.1/2.2 configs, and AVA annotation assets checked in | Another attempt at running it on custom data |
| [coocoo90/tubelet-transformer](https://github.com/coocoo90/tubelet-transformer) | Sep 2022: fixes JHMDB loading, changes sampling, edits eval paths | If you ever want JHMDB-style full tubelets (non-AVA mode) |
| [yjh0410/tubelet-transformer](https://github.com/yjh0410/tubelet-transformer) | "debug train" commits on the CSN-50 config | The author is Jianhua Yang, who later made **YOWOv2** (below), a real-time alternative |
| [jiaozizhao/tubelet-transformer](https://github.com/jiaozizhao/tubelet-transformer) | README edits by the first author (Jiaojiao Zhao) | nothing code-wise |

## 5.3 Alternative / newer spatio-temporal action detectors (with code)

| Model | Venue | Type | Code | Why you might care |
|---|---|---|---|---|
| **STMixer** | CVPR 2023 | one-stage, query-based (sparse) | [MCG-NJU/STMixer](https://github.com/MCG-NJU/STMixer) | The same family as TubeR (DETR-like, no external detector), stronger on AVA, and with cleaner code. The natural "next step" comparison |
| **EVAD** | ICCV 2023 | end-to-end, ViT with token dropout | [MCG-NJU/EVAD](https://github.com/MCG-NJU/EVAD) | End-to-end like TubeR, but with a VideoMAE ViT backbone; efficient |
| **VideoMAE (v1, v2) for detection** | NeurIPS 2022 / CVPR 2023 | two-stage (person boxes + ViT) | [MCG-NJU/VideoMAE-Action-Detection](https://github.com/MCG-NJU/VideoMAE-Action-Detection), [OpenGVLab/VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2) | Among the top AVA results; self-supervised pretraining is data-efficient, which matters for small custom datasets |
| **Hiera** | ICML 2023 | backbone (hierarchical ViT, MAE-pretrained) | [facebookresearch/hiera](https://github.com/facebookresearch/hiera) | Strong, fast video backbone with AVA results |
| Gritsenko et al., "End-to-End Spatio-Temporal Action Localisation with Video Transformers" | CVPR 2024 | end-to-end transformer (Google) | [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Gritsenko_End-to-End_Spatio-Temporal_Action_Localisation_with_Video_Transformers_CVPR_2024_paper.pdf) (Google's JAX research code lives in [scenic](https://github.com/google-research/scenic)) | Can be trained with sparse, key-frame-only boxes, much like our annotation budget |
| **SlowFast** | ICCV 2019 | two-stage, 3D CNN | [facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast) | The classic baseline; its AVA data format is the one we copy |
| **YOWO / YOWOv2** | 2019 / 2023 | one-stage, real-time CNN | [wei-tim/YOWO](https://github.com/wei-tim/YOWO), [yjh0410/YOWOv2](https://github.com/yjh0410/YOWOv2) | **If you need live CCTV speed**: much lighter than TubeR, trains on AVA/UCF24/JHMDB formats |
| **AlphAction** | ECCV 2020 | two-stage + interaction modelling | [MVIG-SJTU/AlphAction](https://github.com/MVIG-SJTU/AlphAction) | Person–person and person–object interaction (fights, hand-overs); includes a demo pipeline for video |
| **Query matching (QMM)** | arXiv 2024 | improves query-based detectors like TubeR | [arXiv:2409.18408](https://arxiv.org/abs/2409.18408) | Deals with fast motion, where a person's box moves a lot within the clip |

**Toolboxes:**
- **[MMAction2](https://github.com/open-mmlab/mmaction2)** (OpenMMLab, ~5k stars, active). It has
  a model zoo for spatio-temporal action detection (SlowFast, SlowOnly, VideoMAE, ACRN, LFB) and
  [documentation for custom AVA-format datasets](https://mmaction2.readthedocs.io/en/latest/model_zoo/detection.html).
  Our CSV/JSON is close to the AVA format, so running a two-stage baseline there takes little
  conversion work. It's the best choice for a "TubeR vs. two-stage" comparison.
- **[PySlowFast](https://github.com/facebookresearch/SlowFast)**, FAIR's reference AVA code.

**Surveys / lists:**
- [GitHub topic: spatio-temporal-action-localization](https://github.com/topics/spatio-temporal-action-localization)
- [GitHub topic: action-detection](https://github.com/topics/action-detection)
- "Deep Learning-based Action Detection in Untrimmed Videos: A Survey", <https://arxiv.org/abs/2110.00111>

## 5.4 Datasets relevant to park / surveillance action detection

| Dataset | What | Relevance |
|---|---|---|
| **AVA / AVA-Kinetics** ([research.google.com/ava](https://research.google.com/ava/)) | Movies, 80 atomic actions, 1 Hz key-frame boxes | What TubeR was trained on, and the annotation protocol we copy |
| **MEVA** ([mevadata.org](https://mevadata.org), [paper](https://arxiv.org/abs/2012.00914)) | Kitware/IARPA DIVA: 29 cameras including outdoor, EO + thermal IR, ~100 actors; 144 h annotated for 37 activity types with boxes | **The closest public data to "outdoor security cameras"**, with thermal footage too. Good for pre-fine-tuning before your parks |
| **VIRAT Video Dataset** ([viratdata.org](https://viratdata.org)) | Outdoor surveillance (parking lots, streets), person and vehicle activities | Same camera geometry as CCTV: high, far, small people |
| **ROAD / ROAD++** ([gurkirt/road-dataset](https://github.com/gurkirt/road-dataset)) | Road agents, actions and locations from a car camera | The TubeR fork above runs on it |
| **UCF101-24, JHMDB-21** | Classic action-tube benchmarks (single actor, trimmed) | TubeR's non-AVA mode |
| **MultiSports** | Sports, fine-grained, fast motion | For fast actions (falls, running) |
| **Okutama-Action** | Aerial (drone) video of people with action labels and boxes | Top-down views of small people, similar to high-mounted cameras |
| **UCF-Crime**, **RWF-2000**, **CCTV-Fights** | Surveillance anomalies / violence, video-level labels | Not box-level, but useful for finding fight/fall clips to annotate or pre-train on |

## 5.5 Tools

- **Annotation:** [CVAT](https://github.com/cvat-ai/cvat) (recommended, self-hostable, video tracks),
  [VIA (VGG Image Annotator)](https://www.robots.ox.ac.uk/~vgg/software/via/) (single HTML file, runs offline),
  [Label Studio](https://github.com/HumanSignal/label-studio).
- **Face blurring, for sharing figures and demos:** any face detector (e.g. from OpenCV's DNN module or
  `insightface`) plus a Gaussian blur on the output video.
- **Video conversion:** `ffmpeg`; vendor players for proprietary NVR formats.
