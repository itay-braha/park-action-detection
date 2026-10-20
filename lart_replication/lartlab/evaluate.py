"""Run the authors' AVA evaluation on all or part of AVA val, optionally with an input ablation.

Everything here reuses upstream code: PHALP_action_dataset (data), LART_LitModule.validation_step (inference
+ writing per-track predictions), AVA_evaluator.write_ava_csv and the official ActivityNet AVA scorer. We add:
  * subsets made of WHOLE videos, scored against the ground truth of those videos only;
  * "ablations": functions that edit a sample's input tokens before the model sees them.

Ablations work through the mechanism LART already uses for missing detections: a frame/person whose
`has_detection` is 0 is replaced by the learned [MASK] token (bert_mask(..., "zero") in the transformer).
So hiding frames or people is in-distribution for the model. Zeroing a whole feature type (appearance or
pose) is NOT something it saw in training - those results measure reliance, not what a retrained model gets.
"""
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from . import ROOT  # noqa: F401 (also sets HOME / sys.path)


def load_config(config_path=None):
    from omegaconf import OmegaConf
    config_path = config_path or ROOT / "logs/LART_Hiera/0/.hydra/config.yaml"
    cfg = OmegaConf.load(config_path).configs
    # these two use Hydra-only interpolations (${paths...}, ${hydra:...}) that cannot resolve outside a Hydra run
    cfg.data_dir = str(ROOT / "data") + "/"
    cfg.storage_folder = str(ROOT / "outputs" / "_model")
    (ROOT / "outputs" / "_model" / "results").mkdir(parents=True, exist_ok=True)   # LitModule looks here at init
    return cfg


def load_model(cfg, weights=None, device="cuda"):
    from lart.models.lart import LART_LitModule
    weights = weights or ROOT / "logs/LART_Hiera/0/checkpoints/epoch_002-EMA.ckpt"
    model = LART_LitModule(cfg)
    sd = torch.load(weights, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({k.replace("._orig_mod", ""): v for k, v in sd.items()}, strict=True)
    return model.to(device).eval()


def video_of(path):
    return os.path.basename(str(path)).split("ava-val_")[1][:11]


def build_val_dataset(cfg, videos=None, limit=0):
    """PHALP_action_dataset over data/ava_val, optionally restricted to whole videos or the first N tracks."""
    from lart.datamodules.components.phalp_action_datatset import PHALP_action_dataset
    os.chdir(ROOT)
    (ROOT / "data" / "_TMP").mkdir(parents=True, exist_ok=True)
    cfg.data_dir = str(ROOT / "data") + "/"
    ds = PHALP_action_dataset(cfg, train=False)
    if videos:
        keep = [i for i, d in enumerate(ds.data) if video_of(d) in set(videos)]
    elif limit:
        keep = list(range(min(limit, len(ds.data))))
    else:
        return ds
    ds.data, ds.track2video = ds.data[keep], ds.track2video[keep]
    return ds


def all_videos(ds):
    return sorted({video_of(d) for d in ds.data})


# ------------------------------------------------------------------------------------- ablations
def key_frame_index(meta_data, video_name):
    """Index of the key frame inside the sample - found by frame NAME, exactly like LART's scorer
    (AVA_evaluator.store_results_batch). Note: has_gt==2 marks ~6 frames around it, not just one."""
    key = video_name.split("ava-val_")[1][12:].split("_")[0]          # e.g. "000390"
    for i, name in enumerate(meta_data["frame_name"]):
        if name != "-1" and key in os.path.basename(str(name)):
            return i
    return None


class ablate_context:
    """Keep only frames within +-window of the key frame (others become [MASK] tokens). window=0: one frame.
    (A class, not a closure, so DataLoader workers can pickle it on Windows.)"""

    def __init__(self, window):
        self.window = window
        self.__name__ = f"context_pm{window}"

    def __call__(self, inp, out, meta, name):
        k = key_frame_index(meta, name)
        if k is not None:
            t = np.arange(inp["has_detection"].shape[0])
            inp["has_detection"][np.abs(t - k) > self.window] = 0
        return inp, out


def ablate_other_people(inp, out, meta, name):
    """Hide the (up to 4) other people tracked at the same key frame: LART sees only the person being classified."""
    inp["has_detection"][:, 1:] = 0
    return inp, out


def ablate_appearance(inp, out, meta, name):
    """Zero the appearance (Hiera/MViT) features -> pose + 3D joints only. OUT-OF-DISTRIBUTION test."""
    if "apperance_emb" in inp:
        inp["apperance_emb"][:] = 0
    return inp, out


def ablate_pose(inp, out, meta, name):
    """Zero SMPL pose/shape and 3D joints -> appearance only. OUT-OF-DISTRIBUTION test."""
    inp["pose_shape"][:] = 0
    if "joints_3D" in inp:
        inp["joints_3D"][:] = 0
    return inp, out


def _seed_worker(worker_id, seed):
    np.random.seed(seed + worker_id)          # PHALP_action_dataset samples "other people" with np.random


class AblatedDataset(torch.utils.data.Dataset):
    def __init__(self, base, fn=None):
        self.base, self.fn = base, fn

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        input_data, output_data, meta_data, video_name = self.base[i]
        if self.fn is not None:
            input_data, output_data = self.fn(input_data, output_data, meta_data, video_name)
        return input_data, output_data, meta_data, video_name


# ------------------------------------------------------------------------------------ evaluation
def score_csv(pred_csv, videos=None, out_dir=None):
    """Official AVA frame-mAP@0.5 of a prediction CSV, optionally against the GT of some videos only."""
    from lart.ActivityNet.Evaluation.get_ava_performance import run_evaluation
    from phalp.configs.base import CACHE_DIR
    lart_dir = Path(CACHE_DIR) / "lart"
    gt_path = lart_dir / "ava_val_v2.2.csv"
    if videos:
        gt_path = Path(out_dir) / "gt_subset.csv"
        with open(lart_dir / "ava_val_v2.2.csv") as f:
            gt_path.write_text("".join(l for l in f if l.split(",")[0] in set(videos)))
    with open(lart_dir / "ava_action_list_v2.2_for_activitynet_2019.pbtxt") as lm, open(gt_path) as gt, \
            open(pred_csv) as det:
        m = run_evaluation(lm, gt, det)[0]
    per_class = {k.split("IOU/", 1)[1]: v * 100 for k, v in m.items() if "PerformanceByCategory/AP@" in k}
    return m["PascalBoxes_Precision/mAP@0.5IOU"] * 100, per_class


def evaluate(model, cfg, dataset, name, ablation=None, test_type="track.fullframe@avg.6",
             batch_size=8, workers=0, videos=None, precision="32", seed=0):
    """Run validation_step over `dataset` (with optional ablation) and score it. Returns (mAP, per_class)."""
    import lightning as L
    out = ROOT / "outputs" / name
    shutil.rmtree(out, ignore_errors=True)
    for sub in ("slowfast", "results", "videos"):
        (out / sub).mkdir(parents=True)
    cfg.storage_folder = str(out)
    cfg.test_type = test_type
    model.cfg.storage_folder = str(out)
    model.cfg.test_type = test_type
    model.evaluator_ava.cfg = model.cfg
    # validation_step stores per-track predictions (needs compute_map=True); the upstream epoch-end hook would
    # then score against ALL of AVA val - we score ourselves (subset-aware) instead.
    model.on_validation_epoch_end = lambda: None

    # When more than 4 other people share a key frame, PHALP_action_dataset picks 4 at random, and upstream
    # evaluates with seed=None. We seed it (and can measure the spread across seeds).
    import functools
    np.random.seed(seed)
    loader = torch.utils.data.DataLoader(AblatedDataset(dataset, ablation), batch_size=batch_size,
                                         shuffle=False, num_workers=workers, pin_memory=True,
                                         worker_init_fn=functools.partial(_seed_worker, seed=seed))
    trainer = L.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1, logger=False,
                        precision=precision, enable_checkpointing=False, enable_progress_bar=False,
                        enable_model_summary=False, default_root_dir=str(out),
                        callbacks=[L.pytorch.callbacks.Timer()])
    trainer.validate(model=model, dataloaders=loader, verbose=False)
    model.evaluator_ava.write_ava_csv(str(out / "slowfast") + "/", str(out / "ava_val.csv"))
    return score_csv(out / "ava_val.csv", videos, out)
