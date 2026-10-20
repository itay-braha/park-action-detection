# 3. Using LART on park CCTV (and comparing it with TubeR)

## 3.1 Zero-shot first

The demo (`run_demo.py`, Linux) runs the released **LART-MViT** model (AVA v2.2: 42.3 mAP) on any video and
writes:
- `outputs/PHALP_<video>.mp4`, with tracked people drawn as 3D meshes and their top AVA actions;
- `outputs/results/*.pkl` (3D tracks), and `results_temporal_fast/` with per-frame action probabilities for every
  track. These are the inputs for any analysis or fine-tuning.

AVA's 60 classes already cover most park activity (stand, sit, walk, run, ride, lie, carry, talk, hug, fight,
fall…). Run it on a few minutes of each camera before deciding what to annotate.

## 3.2 What changes vs. the TubeR pipeline in this repo

| Step | TubeR (`../tools/`) | LART |
|---|---|---|
| Frames | `extract_frames.py` at 30 fps | PHALP reads the video itself (or a frame folder) |
| People | learned by TubeR's queries | detected by ViTDet-H, lifted to 3D by HMR 2.0, tracked by PHALP |
| Annotation | boxes + actions at 1 Hz key frames (CVAT) | **the same CVAT annotations** work: you need actions per (track, key frame), and PHALP's tracks are matched to your boxes by IoU |
| Training data unit | a 32-frame clip | a tracklet `.pkl` (pose, joints, location, appearance, labels per frame) |
| Labels | multi-hot over your classes | multi-hot over your classes (replace the AVA head), or keep AVA's 60 |

## 3.3 Fine-tuning on your own classes (how it would work)

The authors describe the recipe in issue #12, and the `howird/lart` fork does it for ice hockey:
1. Run PHALP + LART's feature pass on your videos with `post_process.save_fast_tracks=True`. This writes one
   `.pkl` per track with the same fields as the AVA tracks (see the notebook, Part A).
2. Match your CVAT key-frame boxes to PHALP's tracks (IoU > 0.5 at the key frame). Write your labels into
   `action_label_gt` and set `has_gt = 2` on those frames.
3. Multi-label classes (like AVA) → replace `action_head_ava` with a `Linear(512, N)`. Single-label → use the
   Kinetics head slot instead.
4. Load the released checkpoint with `load_strict=False`, enable layer-wise LR decay
   (`configs.solver.layer_decay`, e.g. 0.75), and fine-tune the whole model on your tracks.
5. Small datasets: the hockey fork adds a **balanced sampler** and a **class-weighted loss**, which are worth
   copying for rare park events (falls, fights).

Known gap: upstream issue #16. With the current PHALP, some fast-track files lack `apperance_index`, and the
fix is in the feature pass. Expect to debug this step on your server.

## 3.4 Where LART should help on park footage, and where it may not

**Helps:**
- **Person movement** (walk / run / sit / lie / ride): the 3D pose and track signal is strongest here.
- **Interactions** (fights, hugs, talking groups): other people are explicit tokens.
- **Occlusions** (trees, benches, other people): SMPL is amodal and missing frames become mask tokens.
- **Long, slow actions**: the context is up to about 4 s instead of 2 s.

**Risks:**
- **Small, distant people.** HMR 2.0's 3D pose degrades on people tens of pixels tall, while TubeR at least
  sees pixels. Measure AP binned by box height.
- **High, top-down viewpoints** are rare in AVA and in HMR training data.
- **Night IR footage** and heavy compression hurt detection, and therefore tracks, and therefore everything else.
- **Compute.** Five networks per video. Offline analysis only, unless you substitute lighter parts.

## 3.5 A fair comparison for your report

Run both on the *same* held-out cameras with the *same* key-frame annotations and the AVA frame-mAP@0.5 protocol:
1. TubeR zero-shot (AVA head) vs. LART zero-shot (AVA head), on the classes you map to AVA.
2. TubeR fine-tuned vs. LART fine-tuned (if the custom-track step is done).
3. Report person-AP too. If LART's detector (ViTDet-H) finds more people than TubeR's queries, part of LART's
   gain is detection, not action reasoning. The `@GT` run in `eval_ava.py` is how to separate the two on AVA.
