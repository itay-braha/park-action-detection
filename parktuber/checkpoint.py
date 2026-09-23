"""Loading pretrained weights into a TubeR model whose head size differs from the checkpoint.

Upstream `utils/model_utils.load_model(load_fc=False)` tries to delete `module.fc.*`, keys that
do not exist in TubeR (the classifier is `class_fc`), so fine-tuning to a new label set crashes.
This module does it properly:

  * strips the `module.` prefix that DDP-trained checkpoints carry;
  * copies every tensor whose name and shape match;
  * `query_embed` with a different QUERY_NUM: copies the first min(old, new) queries;
  * `class_fc` with a different NUM_CLASSES: optionally initialises each new class from the mean
    of one or more AVA classes (configs/ava_to_park.yaml), otherwise leaves it randomly initialised.
"""
from pathlib import Path

import torch
import yaml

from . import upstream  # noqa: F401
from utils.utils import read_labelmap  # upstream


def torch_load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


# Settings that change the network's structure. They must match the checkpoint being loaded, so we
# read them from the config stored inside the checkpoint instead of trusting the yaml.
# (Example of why: upstream's AVA yaml says TEMPORAL_DS_STRATEGY 'avg' but the released AVA 2.2
# checkpoint was trained with 'decode' - a small transformer that pools the time axis.)
ARCH_KEYS = ["BACKBONE_NAME", "TEMPORAL_DS_STRATEGY", "LAST_STRIDE", "SINGLE_FRAME", "DS_RATE", "D_MODEL",
             "NHEAD", "ENC_LAYERS", "DEC_LAYERS", "DIM_FEEDFORWARD", "NORMALIZE_BEFORE"]


def checkpoint_cfg(ckpt):
    """The config stored in a checkpoint (upstream: CfgNode under 'config'; ours: yaml text under 'cfg')."""
    from yacs.config import CfgNode
    c = ckpt.get("config") if isinstance(ckpt, dict) else None
    if c is None and isinstance(ckpt, dict) and isinstance(ckpt.get("cfg"), str):
        c = CfgNode.load_cfg(ckpt["cfg"])
    return c


def sync_architecture(cfg, ckpt, log=print):
    """Copy ARCH_KEYS from the checkpoint's own config into cfg (with a note for every change)."""
    c = checkpoint_cfg(ckpt)
    if c is None:
        log("  (checkpoint has no stored config - using architecture from the yaml)")
        return
    for k in ARCH_KEYS:
        if k in c.CONFIG.MODEL and cfg.CONFIG.MODEL.get(k) != c.CONFIG.MODEL[k]:
            log(f"  MODEL.{k}: {cfg.CONFIG.MODEL.get(k)} -> {c.CONFIG.MODEL[k]} (from checkpoint)")
            cfg.CONFIG.MODEL[k] = c.CONFIG.MODEL[k]
    old_t = c.CONFIG.DATA.get("TEMP_LEN")
    if old_t and old_t != cfg.CONFIG.DATA.TEMP_LEN:
        log(f"  note: checkpoint was trained with TEMP_LEN {old_t}, you use {cfg.CONFIG.DATA.TEMP_LEN}")


def _strip(sd):
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def read_label_names(label_path):
    """Returns {id: name} from a .pbtxt label map (ids are 1-based)."""
    labelmap, _ = read_labelmap(label_path)
    return {d["id"]: d["name"] for d in labelmap}


def _ava_head_init(model_sd, ckpt_sd, cfg, log):
    """Build class_fc weights for our classes from AVA rows (AVA output index = AVA id - 1)."""
    mapping_path = cfg.CONFIG.FINETUNE.HEAD_INIT_MAPPING
    if not mapping_path or not Path(mapping_path).exists():
        log("  class_fc: new label set -> random init (no HEAD_INIT_MAPPING)")
        return {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f) or {}
    names = read_label_names(cfg.CONFIG.DATA.LABEL_PATH)
    w_old, b_old = ckpt_sd["class_fc.weight"], ckpt_sd["class_fc.bias"]
    w_new, b_new = model_sd["class_fc.weight"].clone(), model_sd["class_fc.bias"].clone()
    for cid, name in names.items():
        ava_ids = mapping.get(name) or []
        ava_ids = [a for a in ava_ids if 1 <= a <= w_old.shape[0]]
        if not ava_ids:
            log(f"  class_fc[{cid - 1}] '{name}': random init")
            continue
        rows = torch.as_tensor([a - 1 for a in ava_ids])
        w_new[cid - 1] = w_old[rows].mean(0)
        b_new[cid - 1] = b_old[rows].mean(0)
        log(f"  class_fc[{cid - 1}] '{name}': from AVA ids {ava_ids}")
    return {"class_fc.weight": w_new, "class_fc.bias": b_new}


def load_pretrained(model, ckpt, cfg, log=print, name="checkpoint"):
    """Load a TubeR (or DETR) checkpoint dict into `model` as far as shapes allow. Returns a report dict."""
    ckpt_sd = _strip(ckpt.get("model", ckpt))
    model_sd = model.state_dict()

    # DETR checkpoints shipped with TubeR may carry an extra top-level prefix; detect and remove it.
    if not any(k in model_sd for k in ckpt_sd):
        ckpt_sd = {k.split(".", 1)[1]: v for k, v in ckpt_sd.items() if "." in k}

    new_sd, mismatched = {}, []
    for k, v in ckpt_sd.items():
        if k not in model_sd:
            continue
        if model_sd[k].shape == v.shape:
            new_sd[k] = v
        elif k == "query_embed.weight":
            n = min(v.shape[0], model_sd[k].shape[0])
            t = model_sd[k].clone()
            t[:n] = v[:n]
            new_sd[k] = t
            log(f"  query_embed: copied {n} of {model_sd[k].shape[0]} queries")
        elif not k.startswith("class_fc."):
            mismatched.append((k, tuple(v.shape), tuple(model_sd[k].shape)))

    if "class_fc.weight" in ckpt_sd and ckpt_sd["class_fc.weight"].shape != model_sd["class_fc.weight"].shape:
        new_sd.update(_ava_head_init(model_sd, ckpt_sd, cfg, log))

    missing = [k for k in model_sd if k not in new_sd]
    unused = [k for k in ckpt_sd if k not in model_sd]
    model.load_state_dict({**model_sd, **new_sd})
    log(f"loaded {len(new_sd)}/{len(model_sd)} tensors from {name}"
        f" ({len(missing)} left at init, {len(unused)} checkpoint tensors unused, {len(mismatched)} shape mismatches)")
    for k in missing:
        log(f"  left at init: {k}")
    for k in unused[:10]:
        log(f"  WARNING unused checkpoint tensor: {k}")
    for m in mismatched[:10]:
        log(f"  WARNING shape mismatch {m}")
    return {"loaded": len(new_sd), "missing": missing, "unused": unused, "mismatched": mismatched}


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, cfg, best_map):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "best_map": best_map,
        "cfg": cfg.dump(),  # plain yaml text, so loading never needs pickle of config objects
    }, path)


def load_exact(model, ckpt):
    """Load a checkpoint written by save_checkpoint (or a same-shape upstream one) - every tensor must match."""
    model.load_state_dict(_strip(ckpt.get("model", ckpt)))
