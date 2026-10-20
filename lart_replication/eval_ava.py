"""Replicate LART's AVA v2.2 result (paper / README: 45.1 frame-mAP with the Hiera checkpoint).

    python eval_ava.py                                  # full AVA val: 64 videos, the paper's number
    python eval_ava.py --videos 8                       # first 8 val videos, scored on their own ground truth
    python eval_ava.py --ablation context_pm0           # tracking ablation: only the key frame is visible
    python eval_ava.py --test-type track.fullframe@GT   # oracle labels on PHALP's boxes (box-quality ceiling)

This runs the authors' own evaluation code - their config (shipped in the checkpoint archive),
PHALP_action_dataset, LART_LitModule.validation_step, AVA_evaluator and the ActivityNet AVA scorer - through
Lightning's validate() on one GPU, instead of their SLURM/submitit launcher. Equivalent to the README's
    python lart/train.py -m --config-name lart.yaml trainer=ddp_unused task_name=LART_eval train=False \
        trainer.devices=8 configs.test_type=track.fullframe@avg.6 \
        configs.weights_path=logs/LART_Hiera/0/checkpoints/epoch_002-EMA.ckpt

Ablations (see lartlab/evaluate.py): context_pm<k> | no_others | no_appearance | no_pose
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import lartlab  # noqa: E402,F401  (sets HOME/cache + sys.path before LART/PHALP imports)
from lartlab import evaluate as E  # noqa: E402


def check_data(data_dir: Path, split="ava_val"):
    """Guard against LART issue #29: evaluating a partial download caches an incomplete file list
    (data/_TMP/*.npy) and silently gives a low mAP even after the download is fixed."""
    d = data_dir / split
    marker = data_dir / f"{split}.COMPLETE"
    n = len([f for f in os.listdir(d) if f.endswith(".pkl")]) if d.exists() else 0
    if not marker.exists():
        sys.exit(f"{d} is not marked complete ({n} tracks found). Finish the download, then run\n"
                 "  python tools/download_ava_val.py --mark-complete   (only if extraction finished without errors)")
    expected = int(marker.read_text().split()[0])
    if n != expected:
        sys.exit(f"{d} has {n} tracks but the archive had {expected}. Re-download.")
    for cache in (ROOT / "data" / "_TMP").glob(f"*{split}*.npy"):
        if len(np.load(cache)) != expected:
            print(f"removing stale file-list cache {cache.name} (issue #29)")
            cache.unlink()


ABLATIONS = {"no_others": E.ablate_other_people, "no_appearance": E.ablate_appearance, "no_pose": E.ablate_pose}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=None, help="default: logs/LART_Hiera/0/checkpoints/epoch_002-EMA.ckpt")
    ap.add_argument("--config", default=None, help="default: the checkpoint's own .hydra/config.yaml")
    ap.add_argument("--test-type", default="track.fullframe@avg.6",
                    help="@avg.6: mean of predictions +-6 frames around the key frame (paper); @GT: oracle labels")
    ap.add_argument("--videos", type=int, default=0, help="only the first K of the 64 val videos (0 = all)")
    ap.add_argument("--ablation", default="", help="context_pm<k> | no_others | no_appearance | no_pose")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--precision", default="32", help="32 (paper) or bf16-mixed")
    ap.add_argument("--seed", type=int, default=0, help="seeds the random choice of other people (upstream: unseeded)")
    ap.add_argument("--name", default="")
    args = ap.parse_args()

    import torch
    check_data(ROOT / "data")
    cfg = E.load_config(args.config)
    model = E.load_model(cfg, args.weights, "cuda" if torch.cuda.is_available() else "cpu")
    ds = E.build_val_dataset(cfg)
    videos = E.all_videos(ds)[: args.videos] if args.videos else None
    if videos:
        ds = E.build_val_dataset(cfg, videos=videos)
    ablation = None
    if args.ablation.startswith("context_pm"):
        ablation = E.ablate_context(int(args.ablation[len("context_pm"):]))
    elif args.ablation:
        ablation = ABLATIONS[args.ablation]
    scope = f"{len(videos)}vid" if videos else "all"
    name = args.name or f"eval_{scope}_{args.ablation or 'full'}_{args.test_type.split('@')[1]}"

    print(f"{len(ds)} tracks from {len(videos) if videos else 64} videos | test_type={args.test_type} | "
          f"ablation={args.ablation or 'none'}")
    t0 = time.time()
    mAP, per_class = E.evaluate(model, cfg, ds, name, ablation, args.test_type, args.batch_size, args.workers,
                                videos, args.precision, args.seed)
    minutes = (time.time() - t0) / 60
    summary = {"mAP": mAP, "videos": videos or "all 64", "tracks": len(ds), "test_type": args.test_type,
               "ablation": args.ablation or None, "minutes": round(minutes, 1), "per_class_AP": per_class}
    out = ROOT / "outputs" / name
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nAVA v2.2 frame-mAP@0.5 = {mAP:.2f} on {'full AVA val' if not videos else f'{len(videos)} videos'}"
          f"   (paper, full val: 45.1)   [{minutes:.0f} min]\n-> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
