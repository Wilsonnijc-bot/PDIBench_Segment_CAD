"""Explicit clean/deformed point supervision for a Link5 research ablation.

This is an experiment adapter, not an edit to the pinned upstream loss/network.
The score remains offset L1 magnitude times predicted anomaly probability.
"""
import torch
import torch.nn.functional as F


def raw_score(offset, logits):
    return offset.abs().sum(-1) * logits.sigmoid()


def region_mean(values, mask):
    """Give labeled foreground/background equal weight when both are present."""
    groups = [values[mask], values[~mask]]
    return torch.stack([part.mean() for part in groups if part.numel()]).mean()


def paired_loss(clean_offset, clean_logits, offset, logits, gt_offset, gt_mask):
    mask = gt_mask.bool()
    magnitude = gt_offset.abs().sum(-1)
    scale = magnitude.max().detach().clamp_min(1e-3)
    restoration = region_mean((offset - gt_offset).abs().sum(-1), mask)
    clean_restoration = clean_offset.abs().sum(-1).mean()
    l_offset = (restoration + clean_restoration) / (2 * scale)
    deformed_bce = region_mean(F.binary_cross_entropy_with_logits(logits, gt_mask, reduction='none'), mask)
    clean_bce = F.binary_cross_entropy_with_logits(clean_logits, torch.zeros_like(clean_logits))
    l_mask = (deformed_bce + clean_bce) / 2
    difference = raw_score(offset, logits) - raw_score(clean_offset, clean_logits)
    l_rank = F.relu(0.5 * magnitude[mask] - difference[mask]).mean() / scale if mask.any() else difference.sum() * 0
    loss = l_offset + 0.5 * l_mask + 0.5 * l_rank
    return loss, dict(offset=float(l_offset.detach()), mask=float(l_mask.detach()), ranking=float(l_rank.detach()), total=float(loss.detach()))


def install_point_residual(model):
    """Preserve query information when retrieved attention values are identical.

    Adds the existing frozen point features to attention output; no new parameters.
    A new checkpoint must record this switch and reapply it during inference.
    """
    if getattr(model, '_link5_point_residual', False):
        raise ValueError('point residual already installed')
    model._link5_point_residual = True
    return model.cross_attn.register_forward_hook(lambda module, inputs, result: result + inputs[0])
