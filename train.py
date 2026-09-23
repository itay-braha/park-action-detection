"""Fine-tune TubeR on your own annotated footage.

    python train.py --config configs/park_csn152_8gb.yaml
    python train.py --config configs/park_csn152_8gb.yaml --eval-first          # baseline before training
    python train.py --config configs/park_csn152_8gb.yaml --resume runs/park_tuber/last.pth
    python train.py --config configs/park_csn152_8gb.yaml CONFIG.TRAIN.LR 5e-5  # yacs-style overrides

Outputs in runs/<EXP_NAME>/: config.yaml, train.log, metrics.jsonl, last.pth, best.pth, tb/ (TensorBoard).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from parktuber.config import load_cfg
from parktuber.dataset import build_loader
from parktuber.engine import build_scheduler, evaluate, train_one_epoch
from parktuber.model import build, param_groups
from parktuber.checkpoint import save_checkpoint


class Logger:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def make_writer(path):
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(str(path))
    except Exception:  # tensorboard not installed - logging to files only
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default="", help="checkpoint written by this script to continue from")
    ap.add_argument("--eval-first", action="store_true", help="evaluate once before training (baseline)")
    ap.add_argument("opts", nargs=argparse.REMAINDER, help="config overrides: KEY VALUE ...")
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.opts)
    run_dir = Path(cfg.CONFIG.LOG.BASE_PATH) / cfg.CONFIG.LOG.EXP_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")
    log = Logger(run_dir / "train.log")
    writer = make_writer(run_dir / "tb")

    seed = cfg.CONFIG.FINETUNE.SEED
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)  # noqa: E702
    if not torch.cuda.is_available():
        log("WARNING: CUDA not available - training on CPU will be extremely slow (use only for smoke tests)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model, criterion, ckpt = build(cfg, checkpoint=args.resume, log=log)
    model.to(device)
    criterion.to(device)
    (run_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")  # incl. architecture from checkpoint

    train_loader = build_loader(cfg, "train")
    val_loader = build_loader(cfg, "val")
    log(f"train: {len(train_loader.dataset)} clips, val: {len(val_loader.dataset)} clips")

    optimizer = torch.optim.AdamW(param_groups(model, cfg), lr=cfg.CONFIG.TRAIN.LR,
                                  weight_decay=cfg.CONFIG.TRAIN.W_DECAY)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.CONFIG.FINETUNE.AMP == "fp16" and device.type == "cuda"))

    start_epoch, best = 0, -1.0
    if args.resume:
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if ckpt.get("scaler"):
                scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best = ckpt["epoch"] + 1, ckpt.get("best_map", -1.0)
        log(f"resumed from {args.resume} at epoch {start_epoch}")

    def run_eval(epoch):
        res = evaluate(model, criterion, val_loader, cfg, run_dir / "eval" / f"epoch_{epoch}", epoch, writer, log)
        with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, **res}) + "\n")
        return res

    if args.eval_first:
        run_eval(-1)

    for epoch in range(start_epoch, cfg.CONFIG.TRAIN.EPOCH_NUM):
        stats = train_one_epoch(model, criterion, train_loader, optimizer, scheduler, scaler, cfg, epoch, writer, log)
        log(f"epoch {epoch} done: " + " ".join(f"{k}={v:.3f}" for k, v in stats.items()))
        save_checkpoint(run_dir / "last.pth", model, optimizer, scheduler, scaler, epoch, cfg, best)
        last = epoch == cfg.CONFIG.TRAIN.EPOCH_NUM - 1
        if (epoch + 1) % cfg.CONFIG.VAL.FREQ == 0 or last:
            res = run_eval(epoch)
            if res["mAP"] > best:
                best = res["mAP"]
                save_checkpoint(run_dir / "best.pth", model, None, None, None, epoch, cfg, best)
                log(f"new best mAP {best:.4f} -> {run_dir / 'best.pth'}")
    log(f"finished. best frame-mAP@0.5 = {best:.4f}")
    if writer:
        writer.close()


if __name__ == "__main__":
    sys.exit(main())
