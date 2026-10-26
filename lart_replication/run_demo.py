"""LART on any video: detect -> 3D-reconstruct -> track -> recognise actions -> render.   (Linux; see setup_linux.sh)

    PYOPENGL_PLATFORM=egl python run_demo.py video.source=third_party/LART/assets/jump.mp4
    PYOPENGL_PLATFORM=egl python run_demo.py video.source=/data/park/cam03.mp4 video.output_dir=outputs/cam03
    python run_demo.py video.source=/path/to/frames_folder/            # a folder of frames also works

Pipeline (same as the authors' scripts/demo.py, two passes):
  1. HMR2_4dhuman tracker = PHALP tracking with ViTDet-H person detection + HMR 2.0 3D human mesh recovery.
     Output: a .pkl with one 3D tracklet (SMPL pose, shape, location, appearance) per person.
  2. LART: for every tracklet, MViT appearance features are extracted around the person (SlowFast fork), then the
     LART transformer predicts AVA action probabilities per frame; PHALP renders meshes + labels into a video.

Differences from upstream scripts/demo.py:
  * `import lartlab` first: PHALP's cache ($HOME/.cache) -> lart_replication/cache, built by tools/setup_lart.py;
  * no downloads from people.eecs.berkeley.edu (403 since 2026) - setup_lart.py fetched everything from the new
    host and rebuilt lart_mvit.config from the checkpoint;
  * SMPL comes from your licensed copy (setup_lart.py --smpl), never from a mirror.
Memory: the authors report ~32 GB RAM for the MViT pass; ViTDet-H + HMR 2.0 need a >= 12-16 GB GPU.
"""
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lartlab  # noqa: E402,F401  (must come before phalp / hmr2 / lart imports)

import hydra  # noqa: E402
from hydra.core.config_store import ConfigStore  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from phalp.configs.base import CACHE_DIR, FullConfig  # noqa: E402
from phalp.models.hmar.hmr import HMR2018Predictor  # noqa: E402
from phalp.trackers.PHALP import PHALP  # noqa: E402
from phalp.utils import get_pylogger  # noqa: E402

warnings.filterwarnings("ignore")
log = get_pylogger(__name__)


class HMR2Predictor(HMR2018Predictor):
    """PHALP's HMR head swapped for HMR 2.0 (4D-Humans), exactly as in the authors' demo."""

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        from hmr2.models import download_models, load_hmr2
        download_models()                      # hmr2_data.tar.gz from www.cs.utexas.edu (still online)
        model, _ = load_hmr2()
        self.model = model
        self.model.eval()

    def forward(self, x):
        hmar_out = self.hmar_old(x)
        batch = {"img": x[:, :3, :, :], "mask": (x[:, 3, :, :]).clip(0, 1)}
        model_out = self.model(batch)
        return hmar_out | {"pose_smpl": model_out["pred_smpl_params"], "pred_cam": model_out["pred_cam"]}


class HMR2_4dhuman(PHALP):
    def setup_hmr(self):
        self.HMAR = HMR2Predictor(self.cfg)


class LART(HMR2_4dhuman):
    def __init__(self, cfg):
        needed = ["lart_mvit.config", "lart_mvit.ckpt", "mvit.yaml", "mvit.pyth"]
        missing = [f for f in needed if not os.path.exists(os.path.join(CACHE_DIR, "phalp/ava", f))]
        if missing:
            sys.exit(f"missing {missing} - run: python tools/setup_lart.py --demo")
        super().__init__(cfg)

    def setup_predictor(self):
        log.info("Loading LART (MViT) action predictor...")
        from lart.utils.wrapper_phalp import Pose_transformer
        self.pose_predictor = Pose_transformer(self.cfg, self)
        self.pose_predictor.load_weights(self.cfg.pose_predictor.weights_path)


@dataclass
class Human4DConfig(FullConfig):
    pass


cs = ConfigStore.instance()
cs.store(name="config", node=Human4DConfig)


@hydra.main(version_base="1.2", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:
    # pass 1: track people in 3D
    # people smaller than this (pixels) are ignored. 100x200 is the authors' demo setting for movies; CCTV people are
    # often 20-60 px wide, so run.py passes a smaller size (--lart-min-box)
    cfg.phalp.small_w = int(os.environ.get("LART_MIN_W", 100))
    cfg.phalp.small_h = int(os.environ.get("LART_MIN_H", 200))
    cfg.render.enable = False
    tracker = HMR2_4dhuman(cfg)
    _, pkl_path = tracker.track()
    del tracker

    # pass 2: LART action recognition on every tracklet + rendering
    # LART_RENDER=0 skips the (slow) 3D-mesh video; the action predictions are saved either way
    cfg.render.enable = os.environ.get("LART_RENDER", "1") == "1"
    cfg.render.colors = "slahmr"
    cfg.render.type = "GHOST_MESH"
    cfg.pose_predictor.config_path = f"{CACHE_DIR}/phalp/ava/lart_mvit.config"
    cfg.pose_predictor.weights_path = f"{CACHE_DIR}/phalp/ava/lart_mvit.ckpt"
    cfg.post_process.save_fast_tracks = True
    lart_model = LART(cfg)
    lart_model.setup_postprocessor()
    lart_model.postprocessor.run_lart(pkl_path)


if __name__ == "__main__":
    main()
