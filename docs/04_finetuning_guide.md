# 4. Fine-tuning guide

## 4.1 The recipe

1. **Baseline first.** Evaluate the AVA model with the mapped head, before any training:
   ```bash
   python evaluate.py --config configs/park_csn152_8gb.yaml --checkpoint ""
   ```
   This is your zero-shot number. Fine-tuning has to beat it, and the per-class table tells you which
   classes AVA already understands on your cameras.
2. **Train** with the default recipe:
   ```bash
   python train.py --config configs/park_csn152_8gb.yaml --eval-first
   tensorboard --logdir runs          # http://localhost:6006
   ```
   Starting point: TubeR CSN-152 AVA 2.2. The new class head is initialised from AVA classes. Backbone
   stem→layer3 is frozen, while layer4, the transformer and the heads train. BatchNorm is frozen. bf16,
   batch 1 × grad-accumulation 8. LR 1e-4 (transformer/heads) and 1e-5 (backbone). 20 epochs with 1 epoch
   of warm-up, and LR ×0.1 at epochs 12 and 17.
3. **Watch** `val/mAP`, `val/person_AP` and `train/loss` in TensorBoard, and `runs/<exp>/metrics.jsonl`.
   `best.pth` is saved whenever val mAP improves.
4. **Look at videos**, not just numbers:
   ```bash
   python infer_video.py --config configs/park_csn152_8gb.yaml --checkpoint runs/park_csn152_8gb/best.pth \
       --video data/raw_videos/some_val_camera.mp4 --out outputs/check
   ```
5. **Iterate on the data** (usually the biggest win) or on the knobs below.

## 4.2 Knobs, roughly in order of importance

| Knob | Default | When to change |
|---|---|---|
| `FINETUNE.FREEZE_BACKBONE` | `layer3` | Small data (< ~2k boxes): keep `layer3` or even `layer4` (backbone fully frozen). More data or a big domain gap (night IR, top-down cameras): `layer2`, `layer1` or `stem` |
| `TRAIN.LR` / `LR_BACKBONE` | 1e-4 / 1e-5 | Loss spiky or diverging: halve both. Plateau early: try 2e-4 for the heads |
| `TRAIN.EPOCH_NUM`, `LR_MILESTONE` | 20, [12, 17] | Scale with dataset size. If val mAP peaks early and then drops, you are overfitting: fewer epochs or more freezing |
| `DATA.IMG_SIZE` / `IMG_RESHAPE_SIZE` | 256 / 288 | **Small or distant people:** 320/360 or 384/432. Memory grows about quadratically. Extract frames with `--short-side` ≥ this |
| `MODEL.QUERY_NUM` | 15 | More than ~12 people in view at once: 20–30 (extra queries start random) |
| `DATA.TEMP_LEN` / `FRAME_RATE` | 32 / 2 | Keep, since that's what the checkpoint learned. Slower actions (lying, sitting) don't need more; for very fast ones (falls) try stride 1 |
| `FINETUNE.HEAD_INIT_MAPPING` | `configs/ava_to_park.yaml` | Set `""` to compare against a random head init. The mapping usually helps a lot early on |
| `DATA.GRAYSCALE_P` | 0.2 | 0.3–0.5 if much of the val set is night IR footage; 0 if none is |
| `DATA.HFLIP` | True | Fine for parks |
| `LOSS_COFS.*`, `MATCHER.*` | as in the paper | Leave them alone unless you know why you're changing them |

## 4.3 Memory and speed

Measured on an **RTX 5060 Laptop GPU (8 GB)**, Windows 11, torch 2.11 with cu128, CSN-152,
TEMP_LEN 32, one training clip per step (batch 1), bf16:

| FREEZE_BACKBONE | Input (h×w) | Peak GPU memory | Time / iteration |
|---|---|---|---|
| layer3 (default) | 256×455 (16:9) | 2.0 GB | 0.16 s |
| layer2 | 256×455 | 3.0 GB | 0.24 s |
| stem (almost everything trains) | 256×455 | 4.9 GB | 0.37 s |
| layer3 | 320×569 | 3.1 GB | 0.26 s |

So on 8 GB you can train almost the whole network, or use a bigger input, or `BATCH_SIZE: 2`. Data
loading (32 JPEG decodes per clip) is often the real bottleneck: raise `DATA.NUM_WORKERS` and keep frames
on an SSD. Inference (`infer_video.py`, IMG_SIZE 256, one model call every 0.5 s) processed the 80 s,
768×576 test clip in about 90 s end-to-end, which includes model loading, decoding and writing the
annotated video. That's roughly real time.

On a big Linux GPU, use [configs/park_csn152_24gb.yaml](../configs/park_csn152_24gb.yaml).
Multi-GPU is not implemented because a single modern GPU is plenty for datasets of this size. If you need
it, wrap the model with `torch.nn.parallel.DistributedDataParallel` in `train.py` and launch with `torchrun`.

## 4.4 Reading the numbers

- **frame-mAP@0.5** is the AVA metric: the mean over classes of AP at IoU ≥ 0.5 on key frames. It's
  harsh. A correct action on a slightly loose box of a small person counts as a miss.
- **person-AP@0.5** is class-agnostic person detection.
- The **per-class AP table** is printed at every evaluation. Classes with little val data give noisy
  numbers, and "(no ground truth)" means the class never appears in val, so it's excluded from the mean.
- The **loss** is summed over 6 decoder layers with weights (12·ce + ce_b + 5·L1 + 2·GIoU), so absolute
  values of 50–200 are normal. Look at the trend, and at the individual terms (`loss_bbox`,
  `loss_giou`, `loss_ce`, `loss_ce_b`) in the log.

## 4.5 Debugging checklist: "the loss doesn't go down" or "mAP is 0"

1. **Preview your annotations** (`tools/preview_annotations.py`). Most failures are data bugs.
2. **Check the zero-shot baseline.** If `evaluate.py --checkpoint ""` gives person-AP near 0 on your data
   while the zero-shot videos look fine, the problem is in your annotations or evaluation data, not in
   training.
3. Look at the training log for `WARNING unused checkpoint tensor` or `shape mismatch`. There should be none,
   except for `class_fc` when your number of classes differs from 80.
4. `EVAL_BINARY_THRESH` must be 0 for AP. A high gate is how upstream issue #27 got 0 mAP with a
   decreasing loss.
5. Can the model overfit 20 clips? Set `TRAIN_ANNO` to a tiny JSON and train 30 epochs. Train mAP on
   those clips should approach 1. If it doesn't, something in the pipeline is broken.
6. `FloatingPointError: loss is nan`: lower the LR. If you use `AMP: "fp16"`, switch to `"bf16"`, which
   overflows far less, or to `"none"`.
7. CUDA out of memory: raise `FREEZE_BACKBONE`, lower `IMG_SIZE`, keep `BATCH_SIZE: 1` and raise `GRAD_ACCUM`.

## 4.6 Experiments worth running (good for a course project report)

1. Zero-shot AVA vs. fine-tuned, per class.
2. Head init: AVA mapping vs. random.
3. Freezing depth: `layer4` / `layer3` / `layer1`, against the amount of training data (learning curves at 25/50/100% of the data).
4. Input resolution 256 vs. 320 vs. 384, and its effect on small-person AP. Bin people by box height.
5. Temporal context: `TEMP_LEN` 16 vs. 32 (the checkpoint was trained with 32).
6. Day vs. night-IR performance, with and without `GRAYSCALE_P`.
7. CSN-50 vs. CSN-152: accuracy vs. speed.
8. Cross-camera generalisation: train on parks A and B, test on park C.
9. Compare with a two-stage baseline (a person detector plus VideoMAE/SlowFast from MMAction2, see [05](05_related_work_and_forks.md)).
