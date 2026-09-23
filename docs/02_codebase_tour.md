# 2. Codebase tour

## 2.1 Two layers: upstream TubeR + our `parktuber` package

```
CVProject/
├── third_party/tubelet-transformer/   official code @ commit f610c97 (Aug 2022), patched by tools/setup_tuber.py
│   ├── models/tuber_ava.py            the TubeR network (class DETR) + build_model(cfg)   <- we use as-is
│   ├── models/criterion.py            SetCriterionAVA: losses                               <- as-is
│   ├── models/detr/matcher.py         Hungarian matcher                                     <- as-is
│   ├── models/backbones/ir_CSN_*.py   CSN-50 / CSN-152 backbones                            <- as-is
│   ├── datasets/video_transforms.py   box-aware clip augmentations                          <- as-is
│   ├── evaluates/evaluate_ava.py      frame-mAP evaluator (TF object-detection API port)    <- as-is
│   ├── datasets/ava_frame.py          AVA-only data loader                                  <- replaced
│   ├── utils/video_action_recognition.py  DDP train/val loops                               <- replaced
│   └── train_tuber_ava.py, eval_*.py  multi-node launch scripts                             <- replaced
├── parktuber/
│   ├── upstream.py      puts third_party/tubelet-transformer on sys.path
│   ├── config.py        yacs config: upstream defaults + ours (every key documented)
│   ├── dataset.py       ParkClipDataset: any fps, any video naming, empty key-frames for eval
│   ├── checkpoint.py    shape-aware loading, AVA->park head init, architecture sync
│   ├── model.py         build(), backbone freezing, frozen BatchNorm, post-processing
│   ├── engine.py        single-GPU train loop (AMP + grad accumulation) and evaluation
│   └── common.py        small helpers for the tools (no torch import)
├── train.py / evaluate.py / infer_video.py
└── tools/  setup_tuber.py, extract_frames.py, cvat_to_csv.py, build_annotations.py, preview_annotations.py
```

**Why not fork and edit upstream?** Keeping upstream pristine plus a short list of regex patches
(`PATCHES` in [tools/setup_tuber.py](../tools/setup_tuber.py)) means you can always see exactly what
differs from the paper's code, and re-create it with one command.

**Name clash warning.** Upstream uses the top-level package names `models`, `utils`, `datasets`,
`evaluates` and `pipelines`. Don't create folders with those names in the project root, and don't
`pip install datasets` (HuggingFace) into this venv, or imports will resolve to the wrong package.

## 2.2 What we patched and why

| # | Problem | Symptom | Fix |
|---|---|---|---|
| 1 | **Released AVA 2.2 checkpoint uses `TEMPORAL_DS_STRATEGY: decode`; the shipped yaml says `avg`** | The model builds, loads "fine" (DETR-style loading ignores extra keys), and outputs boxes that are wrong. We reproduced this on a pedestrian clip | `checkpoint.sync_architecture()` copies structural keys from the config stored *inside* the checkpoint; the loader lists unused checkpoint tensors |
| 2 | `DATA_PATH` `'.../frames/{}/'` never `.format(vid)`-ed | "path doesnt exist .../frames/{}/" (issue #21) | patch + our own dataset |
| 3 | `load_model(load_fc=False)` deletes `module.fc.weight`, which doesn't exist (the head is `class_fc`) | `KeyError` when fine-tuning to a new number of classes | `checkpoint.load_pretrained()`: loads by name+shape, partial copy for `query_embed`, head init from AVA rows |
| 4 | Post-processing hard-codes `P(person) > 0.8` | A fine-tuned model still unsure about "person" gets zero detections, so mAP ≈ 0 (issue #27 pattern) | `FINETUNE.EVAL_BINARY_THRESH` (default 0 = rank by score, the correct way to compute AP) |
| 5 | Train/val loops call `torch.distributed.barrier()` unconditionally and assume NCCL | Crash on 1 GPU or Windows | `parktuber/engine.py` has no DDP |
| 6 | `np.int`, `np.float`, `np.bool` | `AttributeError` on NumPy ≥ 1.24 | regex patches |
| 7 | `float(torchvision.__version__[:3]) < 0.7` | "0.26" parses as 0.2 and would select a dead code path | patch to `False` |
| 8 | `torch.load` default `weights_only=True` in PyTorch ≥ 2.6 | Checkpoints pickle a yacs config, so loading fails | `weights_only=False` (only load checkpoints you trust!) |
| 9 | `import matplotlib` inside the evaluator for disabled plotting code | Extra dependency just to compute AP | patched out |
| 10 | AVA-specific data assumptions: 30 fps, timestamps start at 900 s, a global frame folder, `num_workers=9` | Unusable for other data | `parktuber/dataset.py` |
| 11 | BCE loss under autocast is rejected by PyTorch | Crash with AMP | model runs in bf16/fp16, loss in fp32 |
| 12 | BatchNorm with batch size 1–2 | Noisy statistics destroy pretrained features | `FINETUNE.FREEZE_BN` (BN layers stay in eval mode) |

## 2.3 The life of one training sample

```
data/annotations/train.json  key "cam03_morning,004530"  ->  {"bboxes": [[x1,y1,x2,y2] (0..1)], "acts": [[1, 11]]}
                                    │
ParkClipDataset._load:  frames 004530 + (k-16)*2 for k in 0..31   (32 frames, ~2.1 s at 30 fps)
                        resize so the short side = IMG_RESHAPE_SIZE (288)
                        target: boxes [N,5] = (t=16, x1,y1,x2,y2 in px), labels [N,14] multi-hot
                                    │
transforms (upstream):  RandomHorizontalFlip -> RandomSizeCrop_Custom(256) -> ColorJitter (HSV)
                        -> [our] RandomGrayscaleClip(p=0.2) -> ToTensor -> Normalize
                        (the crop drops boxes with area <= 30 px²; Normalize converts boxes to normalised cx,cy,w,h)
                                    │
clip [3, 32, 256, W]  ->  collate_fn pads a batch into a NestedTensor (tensors + padding mask)
                                    │
model(NestedTensor) -> {"pred_logits": [B,15,14], "pred_boxes": [B,15,4], "pred_logits_b": [B,15,3],
                        "aux_outputs": [5 x same, from the earlier decoder layers]}
                                    │
SetCriterionAVA -> Hungarian matching -> loss_ce, loss_ce_b, loss_bbox, loss_giou (+ aux)
weighted sum: 12*ce + 1*ce_b + 5*bbox + 2*giou, per decoder layer
```

## 2.4 Evaluation

`engine.evaluate()` writes two text files per run in the upstream format, and then calls the upstream
evaluator. That evaluator is a port of the official ActivityNet/AVA evaluation, which in turn is based
on the TensorFlow object-detection API:

```
detections.txt:  <key> [x1, y1, x2, y2, score_class1 .. score_classN, P(person)]       (one line per query)
groundtruth.txt: <key> [0, key_pos, x1, y1, x2, y2, label_1 .. label_N]                (one line per GT person)
```

- **frame-mAP@0.5**: per-class AP at IoU 0.5, averaged over classes that appear in the validation set.
  This is the AVA metric.
- **person-AP@0.5**: class-agnostic AP of `P(person)`. It separates "the detector misses people" from
  "the detector finds people but gets the action wrong". If person-AP is high and mAP is low, you need
  more or better action labels. If person-AP is low, work on resolution, crops, and the domain gap.
- Key-frames with nobody annotated are included, so false alarms on empty benches count.
- We checked the evaluator by feeding it the ground truth as predictions: mAP = 1.0.

## 2.5 Config system

yacs, as in upstream: `defaults (parktuber/config.py) ← your yaml ← command-line KEY VALUE overrides`.

- Every project key is defined with a comment in `_add_project_defaults()`. Read it once.
- Relative paths are resolved against the project root, so commands work from any directory.
- `NUM_CLASSES` is taken from the label map automatically.
- **Architecture keys come from the checkpoint you load**, not from the yaml (see 2.2 #1).
- yacs is type-strict: if a default is a float, write `1.0e-4` or `0.0` in yaml, not `0`.

Example override: `python train.py --config configs/park_csn152_8gb.yaml CONFIG.TRAIN.LR 5.0e-5 CONFIG.LOG.EXP_NAME lr5e-5`
