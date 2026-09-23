"""Config loading.

We keep TubeR's yacs config format (so the upstream `build_model(cfg)` works unchanged) and add
our own keys with defaults here. Every key used by this project is listed in `_add_project_defaults`
with a comment - read it once, it doubles as documentation.
"""
from pathlib import Path

from . import upstream  # noqa: F401
from pipelines.video_action_recognition_config import get_cfg_defaults  # upstream
from yacs.config import CfgNode as CN

ROOT = upstream.ROOT

# Keys that hold filesystem paths; relative paths are resolved against the project root.
_PATH_KEYS = [
    ("DATA", "FRAMES_DIR"), ("DATA", "TRAIN_ANNO"), ("DATA", "VAL_ANNO"), ("DATA", "LABEL_PATH"),
    ("MODEL", "PRETRAIN_BACKBONE_DIR"), ("MODEL", "PRETRAIN_TRANSFORMER_DIR"),
    ("FINETUNE", "INIT_FROM"), ("FINETUNE", "HEAD_INIT_MAPPING"), ("LOG", "BASE_PATH"),
]


def _add_project_defaults(cfg):
    C = cfg.CONFIG
    C.EVAL_ONLY = False
    C.TWO_STREAM = False
    C.USE_LFB = False
    C.USE_LOCATION = False

    # ---------------- data ----------------
    C.DATA.DATASET_NAME = "ava"        # keep 'ava': selects TubeR's multi-label (sigmoid) head + AVA matcher
    C.DATA.FRAMES_DIR = "data/frames"  # frames/<video_id>/000000.jpg ... (tools/extract_frames.py)
    C.DATA.TRAIN_ANNO = "data/annotations/train.json"  # built by tools/build_annotations.py
    C.DATA.VAL_ANNO = "data/annotations/val.json"
    C.DATA.LABEL_PATH = "configs/park_actions.pbtxt"  # class names; ids must be 1..NUM_CLASSES
    C.DATA.NUM_CLASSES = 14             # (auto-set from LABEL_PATH)
    C.DATA.CLIP_FPS = 30               # fps the frames were extracted at (AVA frames are 30 fps)
    C.DATA.TEMP_LEN = 32               # frames per input clip
    C.DATA.FRAME_RATE = 2              # temporal stride inside the clip -> 32*2/30 = ~2.1 s of context
    C.DATA.IMG_SIZE = 256              # short side at eval / train crop size
    C.DATA.IMG_RESHAPE_SIZE = 288      # short side before the random crop during training
    C.DATA.HFLIP = True                # horizontal flip augmentation (OK for parks; turn off for text/signs)
    C.DATA.COLOR_JITTER = True
    C.DATA.GRAYSCALE_P = 0.2           # prob. of a grey clip in training (night IR footage); 0 = off
    C.DATA.NUM_WORKERS = 4
    C.DATA.MULTIGRID = False

    # ---------------- model (same meaning as upstream) ----------------
    M = C.MODEL
    M.SINGLE_FRAME = True              # AVA-style: predict boxes for the key (centre) frame
    M.BACKBONE_NAME = "CSN-152"        # or "CSN-50"
    M.TEMPORAL_DS_STRATEGY = "avg"     # architecture keys are overwritten from the checkpoint you load
    M.LAST_STRIDE = False
    M.GENERATE_LFB = False
    M.NAME = "tuber_park"
    M.ENC_LAYERS = 6
    M.DEC_LAYERS = 6
    M.D_MODEL = 256
    M.NHEAD = 8
    M.NUM_ENCODER_LAYERS = 12
    M.DIM_FEEDFORWARD = 2048
    M.QUERY_NUM = 15                   # max people detected per clip; raise for crowded scenes
    M.NORMALIZE_BEFORE = False
    M.DROPOUT = 0.1
    M.DS_RATE = 8
    M.TEMP_LEN = 32                    # must equal DATA.TEMP_LEN
    M.SAMPLE_RATE = 2
    M.PRETRAINED = False               # True = load the Kinetics .mat backbone (training from scratch only)
    M.PRETRAIN_BACKBONE_DIR = ""
    M.PRETRAIN_TRANSFORMER_DIR = ""    # detr.pth (training from scratch only)
    M.PRETRAINED_PATH = ""
    M.LOAD = False
    M.LOAD_FC = False

    # ---------------- training ----------------
    T = C.TRAIN
    T.EPOCH_NUM = 20
    T.BATCH_SIZE = 1                   # clips per step per GPU
    T.LR = 1e-4                        # transformer + heads
    T.LR_BACKBONE = 1e-5               # backbone; 0 = backbone frozen (also set by FINETUNE.FREEZE_BACKBONE)
    T.W_DECAY = 1e-4
    T.WARMUP_EPOCHS = 1
    T.LR_MILESTONE = [12, 17]
    T.STEP = 0.1
    T.AUX_LOSS = True
    T.START_EPOCH = 0

    C.VAL.BATCH_SIZE = 1
    C.VAL.FREQ = 1

    C.MATCHER = CN(new_allowed=True)
    C.MATCHER.COST_CLASS = 12
    C.MATCHER.COST_BBOX = 5
    C.MATCHER.COST_GIOU = 2
    C.MATCHER.BNY_LOSS = True
    C.MATCHER.BEFORE = False

    L = C.LOSS_COFS = CN(new_allowed=True)
    L.MASK_COF = 1
    L.DICE_COF = 12                    # weight of the action (multi-label BCE) loss
    L.BBOX_COF = 5
    L.GIOU_COF = 2
    L.EOS_COF = 0.1                    # weight of "no person" in the person/background head
    L.WEIGHT = 10                      # up-weights matched queries inside the BCE
    L.WEIGHT_CHANGE = 1000
    L.LOSS_CHANGE_COF = 2
    L.CLIPS_MAX_NORM = 0.1

    # ---------------- fine-tuning (this project) ----------------
    F = C.FINETUNE = CN(new_allowed=True)
    F.INIT_FROM = "weights/TubeR_CSN152_AVA22.pth"  # full TubeR checkpoint to start from ("" = none)
    F.HEAD_INIT_MAPPING = "configs/ava_to_park.yaml"  # init new class rows from AVA classes ("" = random)
    F.FREEZE_BACKBONE = "layer3"       # none | stem | layer1 | layer2 | layer3 | layer4(=all)
    F.FREEZE_BN = True                 # keep BatchNorm statistics fixed (batch size is tiny)
    F.AMP = "bf16"                     # bf16 | fp16 | none
    F.GRAD_ACCUM = 8                   # effective batch = BATCH_SIZE * GRAD_ACCUM
    F.EVAL_BINARY_THRESH = 0.0         # person-score gate at evaluation (upstream hard-codes 0.8)
    F.VIS_BINARY_THRESH = 0.5          # person-score gate when drawing videos
    F.VIS_ACTION_THRESH = 0.3          # action-score threshold when drawing videos
    F.SEED = 0
    F.KEEP_EMPTY_VAL = True            # evaluate on key-frames with nobody in them too (false positives)

    C.LOG.BASE_PATH = "runs"
    C.LOG.EXP_NAME = "park_tuber"
    C.LOG.DISPLAY_FREQ = 20
    C.LOG.SAVE_FREQ = 1

    # single process, no torch.distributed (works on Windows, laptops, single-GPU Linux)
    cfg.DDP_CONFIG.DISTRIBUTED = False
    cfg.DDP_CONFIG.AUTO_RANK_MATCH = False
    cfg.DDP_CONFIG.GPU_WORLD_SIZE = 1
    cfg.DDP_CONFIG.GPU = 0
    return cfg


def _resolve(p: str) -> str:
    if not p:
        return p
    path = Path(p)
    return str(path if path.is_absolute() else (ROOT / path))


def load_cfg(config_file: str, opts=None):
    """Load defaults <- yaml file <- command-line overrides (KEY VALUE pairs, yacs style)."""
    cfg = _add_project_defaults(get_cfg_defaults())
    cfg.merge_from_file(config_file)
    if opts:
        cfg.merge_from_list(list(opts))
    for group, key in _PATH_KEYS:
        node = getattr(cfg.CONFIG, group)
        setattr(node, key, _resolve(getattr(node, key)))
    # the label map is the single source of truth for the number of classes
    if Path(cfg.CONFIG.DATA.LABEL_PATH).exists():
        from .common import read_pbtxt
        ids = sorted(read_pbtxt(cfg.CONFIG.DATA.LABEL_PATH))
        if ids and ids[-1] != cfg.CONFIG.DATA.NUM_CLASSES:
            print(f"[config] NUM_CLASSES {cfg.CONFIG.DATA.NUM_CLASSES} -> {ids[-1]} (from {cfg.CONFIG.DATA.LABEL_PATH})")
            cfg.CONFIG.DATA.NUM_CLASSES = ids[-1]
    # keep the duplicated keys consistent
    cfg.CONFIG.MODEL.TEMP_LEN = cfg.CONFIG.DATA.TEMP_LEN
    cfg.CONFIG.MODEL.SAMPLE_RATE = cfg.CONFIG.DATA.FRAME_RATE
    if cfg.CONFIG.FINETUNE.FREEZE_BACKBONE in ("layer4", "all"):
        cfg.CONFIG.TRAIN.LR_BACKBONE = 0.0
    assert cfg.CONFIG.DATA.TEMP_LEN % cfg.CONFIG.MODEL.DS_RATE == 0, "TEMP_LEN must be a multiple of DS_RATE (8)"
    return cfg
