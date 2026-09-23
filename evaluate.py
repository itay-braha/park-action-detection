"""Evaluate a checkpoint on the validation annotations (frame-mAP@0.5 per class + person AP).

    python evaluate.py --config configs/park_csn152_8gb.yaml --checkpoint runs/park_tuber/best.pth
    python evaluate.py --config configs/park_csn152_8gb.yaml --checkpoint "" CONFIG.DATA.VAL_ANNO data/annotations/test.json

With --checkpoint "" the model is evaluated right after initialisation from FINETUNE.INIT_FROM, i.e. the
AVA model with the AVA->park head mapping: a useful "zero-shot" baseline to beat.
"""
import argparse
import json
from pathlib import Path

import torch

from parktuber.config import load_cfg
from parktuber.dataset import build_loader
from parktuber.engine import evaluate
from parktuber.model import build


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="", help="where to write detections/metrics (default: next to checkpoint)")
    ap.add_argument("opts", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.opts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, criterion, _ = build(cfg, checkpoint=args.checkpoint)
    model.to(device)
    criterion.to(device)
    out = Path(args.out or (Path(args.checkpoint).parent / "eval_standalone" if args.checkpoint else "runs/zero_shot_eval"))
    res = evaluate(model, criterion, build_loader(cfg, "val"), cfg, out)
    (out / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"metrics written to {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
