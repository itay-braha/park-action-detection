# data/

Everything in this folder except this file is git-ignored — CCTV footage of real people must never
end up in a repository. Expected layout (all paths configurable):

```
data/
  raw_videos/            your original camera exports (.mp4/.avi/.mkv...)
  frames/<video_id>/     000000.jpg ... + meta.json     <- tools/extract_frames.py
  annotations/
    cvat/*.xml           CVAT "CVAT for video 1.1" exports
    *.csv                AVA-style CSV (one per video)   <- tools/cvat_to_csv.py
    train.json, val.json                                 <- tools/build_annotations.py
    preview/             sanity-check images             <- tools/preview_annotations.py
  samples/vtest.avi      public OpenCV pedestrian clip, handy for trying things out
```

See [docs/03_data_and_annotation.md](../docs/03_data_and_annotation.md).
