"""
Loss functions for MPF-Net / FinalNet multi-task learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def tversky_loss(logit: torch.Tensor, target: torch.Tensor, alpha: float = 0.7, eps: float = 1.0) -> torch.Tensor:
    """
    Tversky loss for binary segmentation.
    Penalizes false negatives (weight alpha) more heavily than false positives (1 - alpha),
    which is critical for imbalanced medical image segmentation (small lesions).
    """
    p = torch.sigmoid(logit.float())
    tp = (p * target).sum()
    fn = ((1.0 - p) * target).sum()
    fp = (p * (1.0 - target)).sum()
    return 1.0 - (tp + eps) / (tp + alpha * fn + (1.0 - alpha) * fp + eps)


def segmentation_loss(out: dict, target: torch.Tensor, valid_mask: torch.Tensor, alpha: float = 0.7) -> torch.Tensor:
    """
    Combined segmentation loss (Tversky + BCEWithLogits) across the main head
    and auxiliary heads for deep supervision.
    Only computed for samples that have ground truth masks (valid_mask is True).
    """
    if valid_mask.sum() == 0:
        return None

    total_loss = 0.0
    outputs = [out["seg"]] + out["aux"]

    for o in outputs:
        o_valid = o[valid_mask]
        t_valid = target[valid_mask]

        if o_valid.shape[-2:] != t_valid.shape[-2:]:
            o_valid = F.interpolate(
                o_valid.float(),
                size=t_valid.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        tversky = tversky_loss(o_valid, t_valid, alpha=alpha)
        bce = F.binary_cross_entropy_with_logits(o_valid.float(), t_valid)
        total_loss = total_loss + tversky + bce

    return total_loss / len(outputs)
