"""Single-GPU train / evaluate loops (replaces upstream's DDP-only loops).

Differences from upstream `utils/video_action_recognition.py`:
  * no torch.distributed (upstream calls dist.barrier() unconditionally -> crashes on 1 GPU / Windows)
  * mixed precision (bf16/fp16) + gradient accumulation, so CSN-152 fits on an 8 GB laptop GPU
  * loss is computed in fp32 (F.binary_cross_entropy refuses to run under autocast)
  * evaluation uses the upstream AVA evaluator (frame-mAP @ IoU 0.5) but with a configurable person gate
"""
import contextlib
import io
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import upstream  # noqa: F401
from evaluates.evaluate_ava import STDetectionEvaluater, STDetectionEvaluaterSinglePerson  # upstream

from .model import postprocess, set_train_mode


def autocast_ctx(cfg):
    amp = cfg.CONFIG.FINETUNE.AMP
    if amp == "none" or not torch.cuda.is_available():
        return contextlib.nullcontext()
    if amp == "bf16" and not torch.cuda.is_bf16_supported():
        amp = "fp16"          # pre-Ampere GPUs (e.g. Titan RTX, RTX 20xx, V100) have no bf16 -> use fp16
    return torch.autocast("cuda", dtype=torch.bfloat16 if amp == "bf16" else torch.float16)


def _to_float(x):
    if torch.is_tensor(x):
        return x.float()
    if isinstance(x, dict):
        return {k: _to_float(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_float(v) for v in x]
    return x


def _move(targets, device):
    out = []
    for t in targets:
        t = {k: v for k, v in t.items() if k != "image_id"}
        out.append({k: v.to(device, non_blocking=True) for k, v in t.items()})
    return out


def build_scheduler(optimizer, cfg, iters_per_epoch):
    """Linear warm-up for WARMUP_EPOCHS, then step decay at LR_MILESTONE epochs (per optimizer step)."""
    T = cfg.CONFIG.TRAIN
    accum = cfg.CONFIG.FINETUNE.GRAD_ACCUM
    steps_per_epoch = max(1, iters_per_epoch // accum)
    warm = T.WARMUP_EPOCHS * steps_per_epoch
    milestones = [m * steps_per_epoch for m in T.LR_MILESTONE]

    def factor(step):
        f = T.STEP ** sum(step >= m for m in milestones)
        if warm and step < warm:
            f *= 0.01 + 0.99 * step / warm
        return f

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_one_epoch(model, criterion, loader, optimizer, scheduler, scaler, cfg, epoch, writer=None, log=print):
    set_train_mode(model, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accum = cfg.CONFIG.FINETUNE.GRAD_ACCUM
    max_norm = cfg.CONFIG.LOSS_COFS.CLIPS_MAX_NORM
    weight_dict = criterion.weight_dict
    meters = {}
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    for it, (samples, targets) in enumerate(loader):
        samples = samples.to(device)  # NestedTensor (clips + padding mask)
        targets = _move(targets, device)
        with autocast_ctx(cfg):
            outputs = model(samples)
        loss_dict = criterion(_to_float(outputs), targets)
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"loss is {loss.item()} at epoch {epoch} iter {it}: {loss_dict}")

        scaler.scale(loss / accum).backward()
        if (it + 1) % accum == 0 or it + 1 == len(loader):
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        for k in ("loss_ce", "loss_ce_b", "loss_bbox", "loss_giou", "class_error"):
            if k in loss_dict:
                meters.setdefault(k, []).append(float(torch.as_tensor(loss_dict[k]).detach()))
        meters.setdefault("loss", []).append(loss.item())
        step = epoch * len(loader) + it
        if writer is not None:
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
        if it % cfg.CONFIG.LOG.DISPLAY_FREQ == 0 or it + 1 == len(loader):
            avg = {k: np.mean(v[-50:]) for k, v in meters.items()}
            mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
            log(f"epoch {epoch} [{it + 1}/{len(loader)}] " + " ".join(f"{k}={v:.3f}" for k, v in avg.items())
                + f" lr={optimizer.param_groups[0]['lr']:.2e} mem={mem:.1f}GB {time.time() - t0:.0f}s")
    return {k: float(np.mean(v)) for k, v in meters.items()}


@torch.no_grad()
def evaluate(model, criterion, loader, cfg, out_dir, epoch=0, writer=None, log=print):
    """Frame-level mAP@0.5 on the key frames (the AVA protocol) + person-detection AP."""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresh = cfg.CONFIG.FINETUNE.EVAL_BINARY_THRESH
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    det_path, gt_path = out_dir / "detections.txt", out_dir / "groundtruth.txt"
    val_losses = []
    with open(det_path, "w") as fdet, open(gt_path, "w") as fgt:
        for samples, targets in loader:
            ids = [t["image_id"] for t in targets]
            samples = samples.to(device)  # NestedTensor (clips + padding mask)
            tg = _move(targets, device)
            with autocast_ctx(cfg):
                outputs = model(samples)
            outputs = _to_float(outputs)
            if all(len(t["labels"]) for t in tg):
                ld = criterion(outputs, tg)
                val_losses.append(sum(ld[k] * criterion.weight_dict[k] for k in ld if k in criterion.weight_dict).item())
            sizes = torch.stack([t["size"] for t in tg])
            scores, boxes, person = postprocess(outputs, sizes, thresh)
            for b, (key, key_pos) in enumerate(ids):
                for q in range(scores.shape[1]):
                    row = np.concatenate([boxes[b, q], scores[b, q], person[b, q:q + 1]])
                    fdet.write(f"{key} {row.tolist()}\n")
                raw = targets[b]["raw_boxes"].numpy()
                lab = targets[b]["labels"].numpy()
                for r, l in zip(raw, lab):
                    row = np.concatenate([[0, key_pos], r[2:6], l])
                    fgt.write(f"{key} {row.tolist()}\n")

    # upstream evaluators are extremely chatty -> capture their prints into a log file
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev = STDetectionEvaluater(cfg.CONFIG.DATA.LABEL_PATH, class_num=cfg.CONFIG.DATA.NUM_CLASSES)
        ev.load_GT_from_path([str(gt_path)])
        ev.load_detection_from_path([str(det_path)])
        mAP, metrics = ev.evaluate()
        pev = STDetectionEvaluaterSinglePerson(cfg.CONFIG.DATA.LABEL_PATH)
        pev.load_GT_from_path([str(gt_path)])
        pev.load_detection_from_path([str(det_path)])
        pAP, _ = pev.evaluate()
    (out_dir / "evaluator_log.txt").write_text(buf.getvalue(), encoding="utf-8")

    per_class = {k.split("IOU/", 1)[1]: float(v) for k, v in metrics.items() if "PerformanceByCategory/AP" in k}  # names may contain "/"
    result = {"mAP": float(mAP[0]), "person_AP": float(pAP[0]),
              "val_loss": float(np.mean(val_losses)) if val_losses else float("nan"), "per_class_AP": per_class}
    log(f"[eval epoch {epoch}] frame-mAP@0.5={result['mAP']:.4f}  person-AP@0.5={result['person_AP']:.4f}"
        f"  val_loss={result['val_loss']:.3f}")
    for name, ap in per_class.items():
        log(f"    {name:32s} AP={ap:.4f}" if not np.isnan(ap) else f"    {name:32s} (no ground truth)")
    if writer is not None:
        writer.add_scalar("val/mAP", result["mAP"], epoch)
        writer.add_scalar("val/person_AP", result["person_AP"], epoch)
        writer.add_scalar("val/loss", result["val_loss"], epoch)
    return result
