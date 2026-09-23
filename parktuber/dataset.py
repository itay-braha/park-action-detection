"""Clip dataset for custom footage, producing exactly the targets TubeR's AVA code expects.

Why not reuse upstream `datasets/ava_frame.py`?  It hard-codes AVA specifics: frames at 30 fps,
timestamps that start at second 900, and one global frame folder. Here a sample is identified by
    key = "<video_id>,<keyframe_index>"          (e.g. "cam03_2024-05-01_1200,004530")
where keyframe_index indexes the extracted JPEGs of that video (tools/extract_frames.py).

Target dict per clip (what upstream criterion / matcher / evaluator read):
    boxes      [N, 5]  (t, cx, cy, w, h) normalised to [0,1] after transforms   (t = key position)
    raw_boxes  [N, 6]  (sample_idx, t, x1, y1, x2, y2) in pixels of the transformed clip - used for eval
    labels     [N, C]  multi-hot action vector (a person can "walk" and "talk" at once)
    size       [2]     (h, w) of the transformed clip
    image_id   [key with ',' -> '_', key_pos]
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import upstream  # noqa: F401
import datasets.video_transforms as T  # upstream (patched) - box-aware clip transforms
from utils.misc import collate_fn      # upstream - pads clips into a NestedTensor


def load_anno_json(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d["video_frame_bbox"], d["frame_keys_list"]


class RandomGrayscaleClip:
    """With probability p turn the whole clip grey (3 identical channels).

    Park cameras switch to infrared at night and deliver grey frames; AVA (movies) is all colour.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, imgs, target):
        if random.random() < self.p:
            imgs = [Image.fromarray(np.asarray(im)).convert("L").convert("RGB") for im in imgs]
        return imgs, target


def make_transforms(mode, cfg):
    D = cfg.CONFIG.DATA
    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    if mode == "train":
        ops = []
        if D.HFLIP:
            ops.append(T.RandomHorizontalFlip())
        ops.append(T.RandomSizeCrop_Custom(D.IMG_SIZE))
        if D.COLOR_JITTER:
            ops.append(T.ColorJitter())
        if D.GRAYSCALE_P > 0:
            ops.append(RandomGrayscaleClip(D.GRAYSCALE_P))
        return T.Compose(ops + [normalize])
    return T.Compose([T.Resize_Custom(D.IMG_SIZE), normalize])


class FrameStore:
    """Lists and caches the sorted frame files of each video folder."""

    def __init__(self, frames_dir):
        self.root = Path(frames_dir)
        self._cache = {}

    def frames(self, vid):
        if vid not in self._cache:
            files = sorted((self.root / vid).glob("*.jpg"))
            if not files:
                raise FileNotFoundError(f"no frames for video '{vid}' in {self.root / vid}")
            self._cache[vid] = files
        return self._cache[vid]


class ParkClipDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, anno_json, mode="train"):
        D = cfg.CONFIG.DATA
        self.mode = mode
        self.num_classes = D.NUM_CLASSES
        self.clip_len = D.TEMP_LEN
        self.rate = D.FRAME_RATE
        # train: resize to IMG_RESHAPE_SIZE then random-crop IMG_SIZE; eval: resize straight to IMG_SIZE
        self.resize_size = D.IMG_RESHAPE_SIZE if mode == "train" else D.IMG_SIZE
        self.transforms = make_transforms(mode, cfg)
        self.store = FrameStore(D.FRAMES_DIR)
        self.boxes, keys = load_anno_json(anno_json)
        if mode == "train" or not cfg.CONFIG.FINETUNE.KEEP_EMPTY_VAL:
            keys = [k for k in keys if len(self.boxes[k]["bboxes"]) > 0]
        self.keys = keys
        self._sample_counter = 0

    def __len__(self):
        return len(self.keys)

    # ------------------------------------------------------------------
    def clip_indices(self, key_frame, n_frames):
        offsets = (np.arange(self.clip_len) - self.clip_len // 2) * self.rate
        return np.clip(key_frame + offsets, 0, n_frames - 1)

    def _load(self, key):
        vid, fidx = key.rsplit(",", 1)
        files = self.store.frames(vid)
        idx = self.clip_indices(int(fidx), len(files))
        first = Image.open(files[idx[0]])
        ow, oh = first.size
        if oh <= ow:
            nh, nw = self.resize_size, int(round(self.resize_size * ow / oh))
        else:
            nw, nh = self.resize_size, int(round(self.resize_size * oh / ow))
        imgs = [Image.open(files[i]).convert("RGB").resize((nw, nh), Image.BILINEAR) for i in idx]
        return imgs, self._target(key, nw, nh)

    def _target(self, key, nw, nh):
        key_pos = self.clip_len // 2
        entry = self.boxes[key]
        boxes, labels = [], []
        for bbox, acts in zip(entry["bboxes"], entry["acts"]):
            onehot = np.zeros(self.num_classes, dtype=np.float32)
            for a in acts:
                onehot[a] = 1.0
            x1, y1, x2, y2 = bbox
            boxes.append([key_pos, x1 * nw, y1 * nh, x2 * nw, y2 * nh])
            labels.append(onehot)
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 5)
        boxes[:, 1::2].clamp_(0, nw)  # x1, x2
        boxes[:, 2::2].clamp_(0, nh)  # y1, y2
        raw_boxes = F.pad(boxes, (1, 0), value=float(self._sample_counter))
        self._sample_counter += 1
        return {
            "image_id": [key.replace(",", "_"), key_pos],
            "boxes": boxes,
            "raw_boxes": raw_boxes,
            "labels": torch.as_tensor(np.array(labels), dtype=torch.float32).reshape(-1, self.num_classes),
            "orig_size": torch.as_tensor([nh, nw]),
            "size": torch.as_tensor([nh, nw]),
        }

    def __getitem__(self, index):
        key = self.keys[index]
        for attempt in range(10):
            imgs, target = self._load(key)
            imgs_t, target_t = self.transforms(imgs, target)
            if self.mode != "train" or target_t["boxes"].shape[0] > 0:
                # (evaluation keeps empty key-frames: they measure false positives)
                imgs, target = imgs_t, target_t
                break
            # the random crop cut every person out - try another crop, then another sample
            if attempt >= 4:
                key = self.keys[random.randrange(len(self.keys))]
        else:
            raise RuntimeError("could not produce a clip with at least one box")
        clip = torch.stack(imgs, dim=0).permute(1, 0, 2, 3)  # [3, T, H, W]
        return clip, target


def build_loader(cfg, mode):
    D = cfg.CONFIG.DATA
    anno = D.TRAIN_ANNO if mode == "train" else D.VAL_ANNO
    ds = ParkClipDataset(cfg, anno, mode)
    bs = cfg.CONFIG.TRAIN.BATCH_SIZE if mode == "train" else cfg.CONFIG.VAL.BATCH_SIZE
    return torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=(mode == "train"), drop_last=(mode == "train"),
        num_workers=D.NUM_WORKERS, pin_memory=True, collate_fn=collate_fn,
        persistent_workers=D.NUM_WORKERS > 0)
