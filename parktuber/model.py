"""Build TubeR (upstream code, unchanged) and apply fine-tuning policies: freezing, frozen BN, init."""
import torch
from torch import nn

from . import upstream  # noqa: F401
from models.tuber_ava import build_model as _upstream_build_model  # upstream
from models.transformer.util import box_ops  # upstream

from .checkpoint import load_exact, load_pretrained, sync_architecture, torch_load

_STAGES = ["stem", "layer1", "layer2", "layer3", "layer4"]


def _freeze_backbone(model, upto, log):
    """Freeze the CSN backbone up to and including stage `upto` (none|stem|layer1..layer4|all)."""
    if upto in ("", "none"):
        return
    upto = "layer4" if upto == "all" else upto
    body = model.backbone.body
    last = _STAGES.index(upto)
    for stage in _STAGES[: last + 1]:
        modules = [body.conv1, body.bn1] if stage == "stem" else [getattr(body, stage)]
        for m in modules:
            for p in m.parameters():
                p.requires_grad_(False)
    log(f"backbone frozen up to '{upto}'")


def set_train_mode(model, cfg):
    """model.train(), but with BatchNorm layers kept in eval mode if FREEZE_BN (tiny batches)."""
    model.train()
    if cfg.CONFIG.FINETUNE.FREEZE_BN:
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()


def build(cfg, checkpoint="", log=print):
    """Returns (model, criterion, loaded_checkpoint_dict_or_None).

    checkpoint = "" : start from FINETUNE.INIT_FROM (partial load: new head / query count allowed)
    checkpoint = path of a run's .pth : load it exactly (evaluation, inference, resume)
    Either way the network structure is taken from the config stored in that checkpoint.
    """
    F = cfg.CONFIG.FINETUNE
    src = checkpoint or F.INIT_FROM
    ckpt = None
    if src:
        log(f"reading {src}")
        ckpt = torch_load(src)
        sync_architecture(cfg, ckpt, log)
    model, criterion, _ = _upstream_build_model(cfg)
    if checkpoint:
        load_exact(model, ckpt)
    elif ckpt is not None:
        load_pretrained(model, ckpt, cfg, log, name=src)
    elif cfg.CONFIG.MODEL.PRETRAIN_TRANSFORMER_DIR:
        # training "from scratch": Kinetics backbone (.mat, loaded by upstream when MODEL.PRETRAINED)
        # + DETR transformer
        log(f"initialising transformer from {cfg.CONFIG.MODEL.PRETRAIN_TRANSFORMER_DIR}")
        load_pretrained(model, torch_load(cfg.CONFIG.MODEL.PRETRAIN_TRANSFORMER_DIR), cfg, log)
    _freeze_backbone(model, F.FREEZE_BACKBONE, log)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    log(f"parameters: {n_all / 1e6:.1f}M total, {n_train / 1e6:.1f}M trainable")
    return model, criterion, ckpt


def param_groups(model, cfg):
    T = cfg.CONFIG.TRAIN
    backbone = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    groups = [{"params": rest, "lr": T.LR}]
    if backbone and T.LR_BACKBONE > 0:
        groups.append({"params": backbone, "lr": T.LR_BACKBONE})
    return groups


@torch.no_grad()
def postprocess(outputs, sizes, binary_thresh=0.0):
    """Model outputs -> per-query action scores, pixel boxes and person scores.

    TubeR has two heads per query:
      pred_logits_b [B,Q,3]  softmax; index 1 = "this query is a person", 2 = "no object"
      pred_logits   [B,Q,C]  sigmoid; independent per-action probabilities (multi-label)
    Final action score = P(person) * P(action | person). Upstream gates P(person) at a hard 0.8,
    which makes a freshly fine-tuned model look like it scores 0 mAP; the gate is configurable here.
    """
    prob_person = outputs["pred_logits_b"].float().softmax(-1)[..., 1]
    gate = prob_person * (prob_person > binary_thresh).float()
    scores = outputs["pred_logits"].float().sigmoid() * gate.unsqueeze(-1)
    boxes = box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"].float())
    h, w = sizes.unbind(1)
    boxes = boxes * torch.stack([w, h, w, h], dim=1)[:, None, :].to(boxes)
    return scores.cpu().numpy(), boxes.cpu().numpy(), prob_person.cpu().numpy()
