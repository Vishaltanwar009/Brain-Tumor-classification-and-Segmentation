"""
Evaluation and ensemble inference pipelines.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import compute_classification_metrics, compute_segmentation_stats


@torch.no_grad()
def evaluate_ensemble_on_test(
    models: list,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluates a 5-fold ensemble on the held-out test set:
    1. Averages softmax class probabilities across all models.
    2. Averages sigmoid segmentation probabilities across all models.
    3. Thresholds the ensembled mask at fixed threshold (0.5).
    4. Computes full research Table A (classification) and Table B (segmentation) metrics.
    """
    for m in models:
        m.eval()

    all_cls_probs = []
    all_seg_probs = []
    all_targets = []
    all_gt_masks = []

    for images, masks, labels, _ in dataloader:
        images = images.to(device, non_blocking=True).expand(-1, 3, -1, -1)

        batch_cls = []
        batch_seg = []

        with torch.autocast(device.type, enabled=(device.type == "cuda")):
            for m in models:
                out = m(images)
                p_cls = torch.softmax(out["cls"].float(), dim=1).cpu().numpy()
                p_seg = torch.sigmoid(out["seg"].float())[:, 0].cpu().numpy()
                batch_cls.append(p_cls)
                batch_seg.append(p_seg)

        # Average probabilities across ensemble models
        mean_cls = np.mean(batch_cls, axis=0)
        mean_seg = np.mean(batch_seg, axis=0)

        all_cls_probs.append(mean_cls)
        all_seg_probs.append(mean_seg)
        all_targets.append(labels.numpy())
        all_gt_masks.append(masks[:, 0].numpy())

    y_prob = np.concatenate(all_cls_probs, axis=0)
    seg_prob = np.concatenate(all_seg_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    gt_masks = np.concatenate(all_gt_masks, axis=0)

    cls_metrics = compute_classification_metrics(y_true, y_prob)
    seg_metrics = compute_segmentation_stats(seg_prob, gt_masks, thr=threshold)

    return {
        "classification": cls_metrics,
        "segmentation": seg_metrics,
        "ensemble_size": len(models),
        "n_samples": len(y_true),
    }
