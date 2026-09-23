# ParkTubeR — fine-tuning TubeR for action detection in park CCTV footage

This project adapts **[TubeR: Tubelet Transformer for Video Action Detection](https://arxiv.org/abs/2104.00969)**
(Zhao et al., CVPR 2022, [official code](https://github.com/amazon-science/tubelet-transformer)) to
security-camera footage of parks in Israel. It detects every person in a video and says what each one is
doing (walking, sitting, riding a scooter, falling, fighting, and so on).

TubeR is *end-to-end*: one network looks at a ~2 second clip and directly outputs
"person box + action probabilities" for everyone in it. It needs no separate person detector,
no anchors and no proposals.

```
video ──► frames @30fps ──► 32-frame clip ──► TubeR (CSN-152 3D CNN + DETR-style transformer)
                                                   │
                                                   ▼
                          for each of 15 "queries":  P(person), box, P(walk), P(sit), P(run) ...
```

## What is in here

| Path | What it is |
|---|---|
| [notebooks/TubeR_walkthrough.ipynb](notebooks/TubeR_walkthrough.ipynb) | **Hands-on tour of TubeR**: every paper concept demonstrated on the real model, with outputs already saved |
| [docs/](docs/) | Six written guides, listed below |
| [parktuber/](parktuber/) | Our Python package: config, dataset, checkpoint loading, train/eval loops |
| [train.py](train.py), [evaluate.py](evaluate.py), [infer_video.py](infer_video.py) | The three entry points |
| [tools/](tools/) | Data pipeline: set up TubeR, extract frames, convert CVAT, build splits, preview |
| [configs/](configs/) | Training configs, the action label map, and the AVA→park class mapping |
| [tests/smoke_test.py](tests/smoke_test.py) | End-to-end test on synthetic videos (proves the whole pipeline runs) |
| `third_party/tubelet-transformer/` | Official TubeR code, downloaded and patched by `tools/setup_tuber.py` (not committed) |
| `weights/` | Pretrained checkpoints (not committed) |
| `data/` | Your videos, frames and annotations (git-ignored; see [data/README.md](data/README.md)) |

### Learning path

1. **[notebooks/TubeR_walkthrough.ipynb](notebooks/TubeR_walkthrough.ipynb)**, executed and saved, so you can
   read it on GitHub or in VS Code without running anything. It takes the real model apart: clip sampling,
   backbone shapes, the `avg`/`decode` bug side by side, query boxes, attention maps, Hungarian matching,
   losses, AP, and a live fine-tuning run that proves the model uses motion. To re-run it (~5 min on 8 GB):
   `jupyter lab notebooks/TubeR_walkthrough.ipynb` (kernel: the project's `.venv`).
2. The docs below, for depth and reference.

### Documentation

1. [01_tuber_explained.md](docs/01_tuber_explained.md): the task, the paper, and how the architecture works
2. [02_codebase_tour.md](docs/02_codebase_tour.md): upstream code map, the bugs we fixed, and how our code plugs in
3. [03_data_and_annotation.md](docs/03_data_and_annotation.md): turning park footage into a training set (CVAT, labels, splits)
4. [04_finetuning_guide.md](docs/04_finetuning_guide.md): training, evaluation, memory and speed numbers, debugging
5. [05_related_work_and_forks.md](docs/05_related_work_and_forks.md): forks of TubeR, alternative models, related datasets
6. [06_park_cctv_israel.md](docs/06_park_cctv_israel.md): domain challenges of Israeli park CCTV, plus privacy and legal notes

## Quick start

### 1. Install (once)

**Windows (PowerShell)**, tested on Windows 11 with an RTX 5060 Laptop GPU:
```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
.\.venv\Scripts\Activate.ps1
```
**Linux**:
```bash
bash setup.sh            # TORCH_CUDA=cu126 bash setup.sh for another CUDA wheel
source .venv/bin/activate
```
Both scripts create `.venv/`, install PyTorch plus the requirements, download and patch the official TubeR
code, and download the recommended checkpoint (`TubeR_CSN152_AVA22.pth`, 1.1 GB).
RTX 50xx GPUs (Blackwell) need the `cu128` wheels or newer, which is the default.

### 2. Check that everything works
```bash
python tests/smoke_test.py          # synthetic data, tiny model, ~1 min
python tests/smoke_test.py --real   # same, with the real CSN-152 checkpoint
```

### 3. See what TubeR does on footage *before* any training
```bash
python infer_video.py --config configs/ava_zero_shot.yaml --checkpoint "" \
    --video data/samples/vtest.avi --out outputs/vtest_zero_shot
```
This runs the original AVA model (80 movie actions) on a public pedestrian clip, or on any of your videos.
It writes `outputs/vtest_zero_shot.mp4` with boxes and labels drawn in, plus a CSV of detections.

### 4. Fine-tune on your own data
```bash
python tools/extract_frames.py   --videos data/raw_videos --out data/frames
#   ... annotate in CVAT, export "CVAT for video 1.1" (see docs/03) ...
python tools/cvat_to_csv.py      --xml data/annotations/cvat/cam03.xml --out data/annotations/cam03.csv
python tools/build_annotations.py --csv "data/annotations/*.csv" --group-regex "^(cam\d+)" --add-empty-keyframes 1
python tools/preview_annotations.py            # LOOK at the pictures before training
python train.py --config configs/park_csn152_8gb.yaml --eval-first
tensorboard --logdir runs
python infer_video.py --config configs/park_csn152_8gb.yaml \
    --checkpoint runs/park_csn152_8gb/best.pth --video some_clip.mp4 --out outputs/some_clip
```

## Status (what has been verified)

- The official code is patched to run on PyTorch 2.11, NumPy 2.4 and Windows. It no longer needs
  DDP/NCCL or multiple GPUs.
- Zero-shot inference with the released AVA 2.2 checkpoint gives correct boxes and "walk"/"stand"
  labels on a real pedestrian CCTV clip.
- The whole pipeline passes the smoke tests (extract, annotate, build, train, evaluate, resume, infer),
  with both the tiny model and the real CSN-152 checkpoint.
- The AVA evaluator was checked to return mAP = 1.0 when given perfect predictions.
- Measured on an RTX 5060 Laptop (8 GB), CSN-152 at 32×256×455 with bf16: 2.0 GB and 0.16 s/iter
  when training layer4 and the transformer; 4.9 GB and 0.37 s/iter when training almost the whole
  backbone. See [docs/04](docs/04_finetuning_guide.md#memory-and-speed).
- **Not yet done:** training on real park data, because that data needs annotating first. Accuracy
  numbers will come from you.

## Important findings about the upstream code

These are explained in detail in [docs/02](docs/02_codebase_tour.md#what-we-patched-and-why).

1. **The AVA 2.2 checkpoint uses `TEMPORAL_DS_STRATEGY: decode`, but the published config says `avg`.**
   With `avg`, the temporal-pooling decoder weights are silently dropped and the boxes come out as
   garbage (we saw this happen). This is a likely cause of upstream issue #17 ("mAP 0.00001").
   Our loader reads the architecture from the checkpoint itself.
2. `DATA_PATH` `{}` was never formatted with the video id (upstream issue #21).
3. `load_model(load_fc=False)` deletes keys that don't exist (`module.fc.*`), so fine-tuning to a new
   label set crashes. We wrote a shape-aware loader, and it can also initialise the new classes from
   related AVA classes.
4. The evaluation gate `P(person) > 0.8` is hard-coded, which can make a freshly fine-tuned model
   score about 0 mAP (compare issue #27). Here it's configurable.
5. The code relies on removed NumPy aliases, a `torchvision.__version__` float parse, `torch.load`
   defaults, and an unconditional `dist.barrier()`. All are fixed or bypassed.

## License and credits

TubeR is © Amazon.com, Inc., under the Apache-2.0 license. It is downloaded unmodified from GitHub, and
our small, listed patches are applied locally ([tools/setup_tuber.py](tools/setup_tuber.py)). Please cite:
```bibtex
@inproceedings{zhao2022tuber,
  title={TubeR: Tubelet transformer for video action detection},
  author={Zhao, Jiaojiao and Zhang, Yanyi and Li, Xinyu and Chen, Hao and Shuai, Bing and Xu, Mingze and
          Liu, Chunhui and Kundu, Kaustav and Xiong, Yuanjun and Modolo, Davide and Marsic, Ivan and
          Snoek, Cees G.M. and Tighe, Joseph},
  booktitle={CVPR}, pages={13598--13607}, year={2022}
}
```
