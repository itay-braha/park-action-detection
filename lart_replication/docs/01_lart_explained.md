# 1. LART, explained

**Paper:** Rajasegaran, Pavlakos, Kanazawa, Feichtenhofer, Malik, *"On the Benefits of 3D Pose and Tracking for
Human Action Recognition"*, CVPR 2023 ([arXiv 2304.01199](https://arxiv.org/abs/2304.01199),
[code](https://github.com/brjathu/LART)). UC Berkeley + Meta FAIR. The model is called **LART**:
*Lagrangian Action Recognition with Tracking*.

Same task as TubeR (see [../../docs/01_tuber_explained.md](../../docs/01_tuber_explained.md)): AVA-style
spatio-temporal action detection. For every person at every key frame, output a box and probabilities for 60 atomic
actions, scored by frame-mAP@0.5. **LART reaches 45.1 mAP on AVA v2.2; TubeR reaches 31.1.**

## 1.1 The idea: Lagrangian vs. Eulerian

The paper borrows two terms from fluid mechanics:

- **Eulerian:** sit on the river bank and watch the water flow past a *fixed point*. This is how almost all
  modern video networks work, including TubeR, SlowFast, MViT and VideoMAE. They extract features on a fixed
  space-time grid (patches, cuboids), and for a person-centred prediction they crop that grid at the person's
  box in the **middle frame**.
- **Lagrangian:** sit in a boat and *drift with the water*. Follow each **person** through the video and reason
  about their trajectory.

Why it matters: if someone runs from left to right, a feature crop at the middle-frame box contains only part of
the motion. A *track* contains all of it, and lets you represent the person in ways that don't depend on the
camera (3D pose).

## 1.2 The pipeline

```
video ─► PHALP tracker ──────────────────────────────────────► tracklets (one per person, up to ~128 frames)
           │  person detection (ViTDet / Mask R-CNN)                  │
           │  3D human mesh per detection (HMR 2.0 → SMPL)            │ for every frame t of every person i:
           │  association in 3D (pose + location + appearance)        ▼
           │                                              H_it = { P_it : 3D pose θ (23 joints, rotation matrices),
           │                                                              global orientation ψ, 3D location L
           │                                                       Q_it : "contextualised appearance" U_it }
           ▼
     appearance model (MViT-MaskFeat or Hiera, an AVA-trained video backbone) run on the person's box every
     ~6 frames → a 1152-d (MViT) or 2048-d (Hiera) feature U_it, shared by neighbouring frames
                                                                   │
                         ┌─────────────────────────────────────────┘
                         ▼
   tokens: person of interest + up to 4 other people from the same video, 125 frames each → 5 × 125 = 625 tokens
   token = Linear(pose 229 → 128) ⊕ MLP(3D joints 135 → 128) ⊕ MLP(appearance → 256) = 512-d
           + 2D sinusoidal position embedding (time t, track-id i)  ;  missing detection → learned [MASK] token
                         ▼
   vanilla Transformer, 16 layers, 16 heads, d = 512 (42 M of the model's 52 M parameters)
                         ▼
   per-token heads:  AVA actions (60, sigmoid) | SMPL pose | 3D location | (Kinetics 400, on a class token)
                         ▼
   prediction for the key frame = mean of sigmoid(action logits) over ±6 frames, on the PHALP box
```

What this repo shows you directly (`notebooks/LART_walkthrough.ipynb`):
- a real tracklet from AVA val, with every one of the fields above;
- the person rendered as a 3D SMPL mesh from its 229 pose numbers;
- the model's per-frame action probabilities along the track.

## 1.3 Why each ingredient helps (paper Tables 2, 5, 6)

| Model (AVA v2.2) | OM (object manip.) | PI (person interact.) | PM (person movement) | mAP |
|---|---|---|---|---|
| JMRN (2D pose only, previous SOTA for pose) | 7.1 | 17.2 | 27.6 | 14.1 |
| **LART-pose**, 3D pose only, 1 person | 12.0 | 22.0 | 46.6 | 22.9 |
| **LART-pose**, 3D pose only, 5 people | 13.3 | 25.9 | 48.7 | **24.1** |
| MViT (MaskFeat) on the same boxes | 32.2 | 41.1 | 58.6 | 40.2 |
| MViT + **tracking** (MViT features over the track) | 33.4 | 43.0 | 59.3 | 41.4 (+1.2) |
| MViT + tracking + **3D pose** = LART-MViT | 34.4 | 43.9 | 59.9 | 42.3 (+0.9) |
| Hiera | 34.5 | 43.0 | 62.3 | 42.6 |
| **LART-Hiera** (the checkpoint we evaluate) | 37.3 | 45.7 | 63.8 | **45.1** |

How to read this:
1. **3D pose alone is a strong signal for movement actions.** Without seeing any pixels, LART-pose reaches about
   80% of MViT's score on person-movement classes (stand, sit, walk, run…). It's weak on object manipulation,
   because the pose doesn't say *which* object is held.
2. **Other people matter.** Going from 1 to 5 people adds +3.9 mAP on person-interaction classes (hug, fight,
   dance, hand-shake).
3. **Tracking matters more than pose** once appearance is present: +1.2 vs. +0.9.
4. **The gains carry over to a stronger backbone.** LART adds +2.5 mAP on top of Hiera, and improves 53 of the
   60 classes.

## 1.4 Training (what the released checkpoint went through)

- **Data:** PHALP was run on all of AVA and Kinetics-400, giving about 1 M tracks and 104 M person boxes
  (900 hours of tracks, mean length 3.4 s). The authors released these tracks; we use the AVA validation part.
- **Pre-training on Kinetics** with *pseudo-labels*. An MViT model predicts AVA actions at 1 Hz on every
  Kinetics track, and those predictions supervise LART. Kinetics has no box-level action labels, so this
  "teacher" trick is what makes 1 M tracks usable.
- **Fine-tuning on AVA** with ground-truth labels. Every token is supervised (a label is assumed valid for
  ±0.5 s around the key frame), which gives far more training signal than one label per clip.
- **Masked training:** the paper trains with 40% of the input tokens replaced by the [MASK] token (§4.4). The
  released checkpoint's config shows `mask_ratio: 0.0` for its last fine-tuning stage. Either way, every frame
  without a detection is a [MASK] token at train and test time. That makes the model robust to missed detections
  and occlusions, and it's why our "hide frames / hide people" ablations stay close to what the model has seen.
- Loss: binary cross-entropy (multi-label). Optimiser: AdamW with cosine schedule. 16-layer transformer.

## 1.5 LART vs. TubeR, side by side

| | TubeR (CVPR 2022) | LART (CVPR 2023) |
|---|---|---|
| Unit of reasoning | space-time feature volume of a 2 s clip | **tracklets of people** (up to ~4 s here) |
| Person detection | inside the model (DETR queries) | external (ViTDet / Mask R-CNN, inside PHALP) |
| Temporal extent | fixed 32-frame window | whole track, 125-frame windows, masked gaps |
| Body representation | none (pixels only) | **3D SMPL pose + 3D location**, amodal (sees occluded limbs) |
| Appearance | its own CSN-152 backbone, trained end-to-end | features from a frozen AVA video model (MViT / Hiera) |
| Other people | implicit (shared feature map) | explicit tokens for up to 4 other tracked people |
| Training data | AVA | AVA + Kinetics-400 tracks with pseudo-labels |
| AVA v2.2 mAP | 31.1 | **45.1** |
| Pieces to run on a new video | 1 network | detector + HMR 2.0 + PHALP + MViT + LART (≈8 GB of weights) |
| Practicality | one model, ~real-time on a laptop GPU | heavy, multi-stage, Linux only in practice |

The trade-off in one sentence: **LART is much more accurate, but it's a system of five models, while TubeR is
one.** That matters for CCTV deployment (see [03](03_park_cctv_with_lart.md)).

## 1.6 Limitations (from the paper and our experience)

- Everything rests on **tracking quality**. An ID switch mixes two people's poses into one track.
- **3D pose from far-away, low-resolution people is noisy.** HMR 2.0 was trained mostly on larger people.
- Object interactions are weak in the pose stream (there's no object model). The appearance features cover them.
- **Heavy compute:** the demo needs about 32 GB of RAM and a large GPU, and the per-track MViT feature
  extraction is slow.
- Single-person-of-interest formulation: n people need n forward passes.

## 1.7 Reading list

1. **The paper**: <https://arxiv.org/abs/2304.01199>. Read §3 (method) and Tables 2, 5, 6.
2. **PHALP**, the tracker: Rajasegaran et al., *"Tracking People by Predicting 3D Appearance, Location & Pose"*,
   CVPR 2022, <https://arxiv.org/abs/2112.04477>.
3. **HMR 2.0 / 4DHumans**, the 3D human recovery: Goel et al., ICCV 2023, <https://arxiv.org/abs/2305.20091>.
4. **SMPL**, the body model: Loper et al., SIGGRAPH Asia 2015, <https://smpl.is.tue.mpg.de>.
5. **Hiera**: Ryali et al., ICML 2023, <https://arxiv.org/abs/2306.00989>. **MaskFeat**: Wei et al., CVPR 2022.
6. AVA: Gu et al., CVPR 2018, <https://arxiv.org/abs/1705.08421>.
