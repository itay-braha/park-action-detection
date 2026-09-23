# 3. From park footage to a training set

Most of the work, and most of the final accuracy, comes from this step. The pipeline:

```
raw_videos/*.mp4 ─► extract_frames.py ─► frames/<vid>/000000.jpg + meta.json
        │
        └──► CVAT (annotate people + actions) ─► export XML ─► cvat_to_csv.py ─► <vid>.csv
                                                                                  │
                                               build_annotations.py ◄─────────────┘
                                                        │
                                   train.json / val.json / stats.json ─► preview_annotations.py (look!)
```

## 3.1 Decide your label set first

Edit [configs/park_actions.pbtxt](../configs/park_actions.pbtxt). The 14 starter classes are:

`stand, walk, run, sit, lie_down, bend_crouch, ride_bike_scooter, fall_down, fight_push, throw,
carry_object, talk_to_person, use_phone, hug`

Guidelines:
- **Pick actions that are visible at your camera's resolution.** `use_phone` may be impossible for a
  person 40 px tall. Fewer, clearly visible classes beat many ambiguous ones.
- **Separate poses from interactions**, like AVA does. Every person gets exactly one pose
  (`stand/walk/run/sit/lie_down/bend_crouch`, and optionally `ride_*`) plus zero or more
  object/person interactions. This makes annotation consistent.
- **Rare, important events** (`fall_down`, `fight_push`) will have few examples. Plan to collect or
  stage more of them (see 3.6).
- After changing the file, update [configs/ava_to_park.yaml](../configs/ava_to_park.yaml) so each class
  starts from the most similar AVA classes (ids are listed in that file).
- Names must contain no spaces, because they become CVAT attribute names.

## 3.2 Extract frames

```bash
python tools/extract_frames.py --videos data/raw_videos --out data/frames            # all videos, 30 fps, short side 360
python tools/extract_frames.py --videos data/raw_videos/cam03.mp4 --start 600 --end 1800   # just a segment
```
- Re-sampling to 30 fps keeps TubeR's time scale (32 frames × stride 2 = 2.1 s) identical to AVA.
- Keep `--short-side` ≥ the `IMG_SIZE` you will train with. Use 360 for IMG_SIZE 256–320, or more if
  you plan to experiment with higher resolution for small people.
- Disk cost: about 15–25 KB per frame at 360p, so ~2 GB per hour of video. Extract only the segments
  you annotate.
- The video id is the file path relative to `--videos`, with `/` replaced by `__`, e.g.
  `yarkon__cam03_2024-05-01_0800`. **Put the camera id at the start of the file name**, because the
  train/val split uses it (3.5).
- NVR exports: `.mp4` and `.avi` work directly. Proprietary `.dav` (Dahua) and some `.264` files must
  first be converted with the vendor player or `ffmpeg -i in.dav -c copy out.mp4`.

## 3.3 Annotate in CVAT

[CVAT](https://github.com/cvat-ai/cvat) is free and open source, and it handles video and tracks. For
CCTV of real people, **self-host it** (`docker compose up`, see the CVAT docs) instead of using the
cloud service. See [06](06_park_cctv_israel.md) for why.

**Project setup (do this once):**
1. Create a project with one label, `person`, of type *rectangle*.
2. Add one **checkbox** attribute per action, named exactly like the `.pbtxt` names (`walk`, `sit`, ...),
   and tick **mutable** so the value can change along a track.
3. Create a task per video, or per 5–10 minute segment. **Use the same video file you extracted frames
   from**, or at least one with the same start time.

**Annotating:**
- Draw a **track** for every visible person (press `N` in track mode). CVAT interpolates boxes between the
  key frames you set, so adjust the box every ~1 s or whenever the motion changes.
- Toggle the action checkboxes at the frame where the action starts or stops.
- AVA protocol, which we copy: annotate at **1 Hz key frames**, **every person**, **all actions that
  apply**. If you skip people, the model is trained to call them "background", which hurts.
- If a person is too small or occluded to judge the action, still draw the box (it teaches person
  detection) and leave the checkboxes empty. They become "person, no action".
- Export with **Menu → Export task dataset → "CVAT for video 1.1"**.

**Speed-up: pre-annotation.** Run the zero-shot model first
(`infer_video.py --config configs/ava_zero_shot.yaml`) and watch its output video. That tells you
quickly which segments contain interesting actions, so you annotate those first rather than 20 minutes
of an empty lawn.

## 3.4 Convert CVAT → CSV

```bash
python tools/cvat_to_csv.py --xml data/annotations/cvat/cam03_task.xml \
    --video-id yarkon__cam03_2024-05-01_0800 --out data/annotations/cam03_0800.csv
```
The converter samples every track at 1 s key frames (`--every-sec`). The source fps comes from
`meta.json`, and the video id defaults to the file name stored in the XML. It prints a warning for
attributes that don't match any class name, which usually means a typo.

The CSV format, which you can also produce with any other tool, is AVA-like:
```
video_id,timestamp,x1,y1,x2,y2,action_id,person_id
yarkon__cam03_2024-05-01_0800,12.000,0.4123,0.3310,0.4550,0.5821,2,7        # person 7 walks...
yarkon__cam03_2024-05-01_0800,12.000,0.4123,0.3310,0.4550,0.5821,13,7       # ...and uses a phone
yarkon__cam03_2024-05-01_0800,12.000,0.7010,0.2000,0.7200,0.2600,0,9        # person 9, action unknown
```
Timestamps are seconds from the start of the video file, and boxes are normalised to [0, 1].

## 3.5 Build train/val splits

```bash
python tools/build_annotations.py --csv "data/annotations/*.csv" \
    --group-regex "__(cam\d+)" --val-ratio 0.2 --add-empty-keyframes 1.0
```
- **Split by camera, not by clip.** Two clips from the same camera share background, angle and
  lighting. A random clip split would measure memorisation, not generalisation. `--group-regex`
  extracts the group from the video id; alternatively, list the validation groups with `--val-groups cam07 cam12`.
- Even better, if you have enough data: **hold out whole parks** for a final test set that you evaluate
  exactly once, at the end.
- `--add-empty-keyframes 1.0` adds key frames with nobody in them. Use it only if your annotation is
  **exhaustive**, meaning every person in the annotated segment is boxed. They make evaluation penalise
  false alarms. Training ignores them, since there is nothing to match.
- The script prints a per-class count table and flags classes with fewer than 30 training examples.

## 3.6 Look at your data

```bash
python tools/preview_annotations.py --n 50      # -> data/annotations/preview/*.jpg
```
Always do this. The typical bugs it catches:
- boxes shifted in time, from the wrong fps or a video that doesn't match the CVAT task;
- boxes scaled wrongly, from pixel coordinates treated as normalised;
- labels off by one (ids are 1-based in the CSV and 0-based inside the JSON).

## 3.7 How much data?

There are no hard rules, but rough experience from fine-tuning AVA-pretrained detectors:

| Per class (training boxes) | What to expect |
|---|---|
| < 30 | noise; the class is effectively untrained (unless AVA already knows it well and you mapped it) |
| 100–300 | usable for classes similar to AVA ones (walk, sit, stand, run, ride) |
| 500+ | solid, including for new concepts |

A person on screen for 60 s gives 60 key-frame boxes, but they are highly correlated. **Diversity
(cameras, times of day, days) matters more than raw count.**

For rare events (falls, fights):
- stage them with consenting volunteers (students) in front of the actual cameras;
- borrow data: AVA itself has `fall down` and `fight/hit`; see also the surveillance datasets in [05](05_related_work_and_forks.md);
- during training, over-sample the key frames that contain them (a small change to `build_loader`,
  using a `WeightedRandomSampler`).

## 3.8 Annotation checklist (give this to annotators)

- [ ] Every visible person gets a track, however small.
- [ ] One pose per person per key frame (stand / walk / run / sit / lie_down / bend_crouch).
- [ ] Interactions are ticked only while they happen.
- [ ] A box covers the whole visible body, tightly. When a person is partly occluded, box only the visible part.
- [ ] Groups: box each person separately.
- [ ] People on bikes or scooters: the box covers the person (the vehicle may be included); tick `ride_bike_scooter`.
- [ ] Children count as persons.
- [ ] When unsure of the action, draw the box and leave the actions empty. Never guess.
