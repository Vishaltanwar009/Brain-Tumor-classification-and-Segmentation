"""
Evaluation metrics for Brain Tumor Segmentation and Classification.
Implements exact research metrics:
- Accuracy, Precision, Recall, Macro-F1, OvR AUC
- Per-image Dice, Pooled Global Dice, IoU, Specificity, Pixel Accuracy
- 95th-percentile Hausdorff Distance (HD95) with empty prediction penalties
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
from scipy.ndimage import distance_transform_edt


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Computes top-1 accuracy, macro precision, recall, F1, and one-vs-rest AUC.
    """
    y_pred = y_prob.argmax(axis=1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    # One-vs-rest AUC
    n_classes = y_prob.shape[1]
    aucs = []
    for c in range(n_classes):
        if len(np.unique(y_true == c)) > 1:
            try:
                auc_c = roc_auc_score(y_true == c, y_prob[:, c])
                aucs.append(auc_c)
            except Exception:
                pass
    macro_auc = float(np.mean(aucs)) if aucs else float("nan")

    return {
        "acc": float(acc),
        "prec": float(prec),
        "rec": float(rec),
        "f1": float(f1),
        "macro_auc": macro_auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def compute_dice_per_image(pred_prob: np.ndarray, gt: np.ndarray, has_mask: np.ndarray = None, thr: float = 0.5) -> np.ndarray:
    """
    Computes per-image Dice score vector.
    Conventions:
      - both empty -> Dice = 1.0
      - one empty, one non-empty -> Dice = 0.0
    """
    pred_bin = pred_prob > thr
    gt_bin = gt > 0.5
    if has_mask is not None:
        valid_idx = np.where(has_mask)[0]
    else:
        valid_idx = np.arange(len(pred_prob))

    dices = []
    for i in valid_idx:
        p = pred_bin[i]
        g = gt_bin[i]
        sum_p = p.sum()
        sum_g = g.sum()
        if sum_p == 0 and sum_g == 0:
            dices.append(1.0)
        elif sum_p == 0 or sum_g == 0:
            dices.append(0.0)
        else:
            intersection = (p & g).sum()
            dices.append(float(2.0 * intersection / (sum_p + sum_g)))

    return np.array(dices, dtype=np.float64)


def compute_hd95_single(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """
    Compute 95th-percentile Hausdorff Distance (HD95) between binary masks.
    Returns float('nan') if either mask is completely empty.
    """
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float("nan")

    dt_gt = distance_transform_edt(~gt_bin)
    dt_pred = distance_transform_edt(~pred_bin)

    d_pred_to_gt = dt_gt[pred_bin]
    d_gt_to_pred = dt_pred[gt_bin]

    all_dists = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_dists, 95))


def compute_segmentation_stats(pred_prob: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> dict:
    """
    Comprehensive segmentation evaluation on test images:
    - Per-image and pooled Dice
    - Per-image and pooled IoU
    - Precision, Recall, Specificity, Pixel Accuracy
    - Defined HD95 and penalised HD95
    """
    pred = pred_prob > thr
    g = gt > 0.5
    N, H, W = pred.shape
    diag_penalty = float(np.hypot(H, W))

    tp = (pred & g).sum(axis=(1, 2)).astype(np.float64)
    fp = (pred & ~g).sum(axis=(1, 2)).astype(np.float64)
    fn = (~pred & g).sum(axis=(1, 2)).astype(np.float64)
    tn = (~pred & ~g).sum(axis=(1, 2)).astype(np.float64)

    sum_p = pred.sum(axis=(1, 2)).astype(np.float64)
    sum_g = g.sum(axis=(1, 2)).astype(np.float64)

    # Per-image Dice & IoU handling degenerate cases
    both_empty = (sum_p == 0) & (sum_g == 0)
    denom_dice = sum_p + sum_g
    denom_iou = sum_p + sum_g - tp

    dice = np.where(both_empty, 1.0, np.where(denom_dice == 0, 0.0, (2.0 * tp) / np.maximum(denom_dice, 1.0)))
    iou = np.where(both_empty, 1.0, np.where(denom_iou == 0, 0.0, tp / np.maximum(denom_iou, 1.0)))

    # Precision, Recall, Specificity
    denom_prec = sum_p
    denom_rec = sum_g
    prec = np.where(both_empty, 1.0, np.where(denom_prec == 0, 0.0, tp / np.maximum(denom_prec, 1.0)))
    rec = np.where(both_empty, 1.0, np.where(denom_rec == 0, 0.0, tp / np.maximum(denom_rec, 1.0)))

    neg_pixels = tn + fp
    spec = np.where(neg_pixels == 0, 1.0, tn / np.maximum(neg_pixels, 1.0))

    pixel_acc = (tp + tn) / float(H * W)

    # Global pooled metrics
    tp_tot = float(tp.sum())
    fp_tot = float(fp.sum())
    fn_tot = float(fn.sum())
    tn_tot = float(tn.sum())

    dice_pooled = (2.0 * tp_tot) / max(1.0, 2.0 * tp_tot + fp_tot + fn_tot)
    iou_pooled = tp_tot / max(1.0, tp_tot + fp_tot + fn_tot)
    prec_pooled = tp_tot / max(1.0, tp_tot + fp_tot)
    rec_pooled = tp_tot / max(1.0, tp_tot + fn_tot)
    spec_pooled = tn_tot / max(1.0, tn_tot + fp_tot)
    pix_acc_pooled = (tp_tot + tn_tot) / max(1.0, tp_tot + tn_tot + fp_tot + fn_tot)

    # Compute HD95
    hd95_vals = []
    hd95_penalised = []
    n_undefined = 0
    for i in range(N):
        h = compute_hd95_single(pred[i], g[i])
        if np.isnan(h):
            n_undefined += 1
            hd95_penalised.append(diag_penalty)
        else:
            hd95_vals.append(h)
            hd95_penalised.append(h)

    return {
        "dice_mean": float(np.mean(dice)),
        "dice_std": float(np.std(dice)),
        "dice_median": float(np.median(dice)),
        "dice_pooled": float(dice_pooled),
        "iou_mean": float(np.mean(iou)),
        "iou_pooled": float(iou_pooled),
        "prec_mean": float(np.mean(prec)),
        "prec_pooled": float(prec_pooled),
        "rec_mean": float(np.mean(rec)),
        "rec_pooled": float(rec_pooled),
        "spec_mean": float(np.mean(spec)),
        "spec_pooled": float(spec_pooled),
        "pixel_acc_mean": float(np.mean(pixel_acc)),
        "pixel_acc_pooled": float(pix_acc_pooled),
        "hd95_mean_defined": float(np.mean(hd95_vals)) if hd95_vals else float("nan"),
        "hd95_median_defined": float(np.median(hd95_vals)) if hd95_vals else float("nan"),
        "hd95_mean_penalised": float(np.mean(hd95_penalised)),
        "n_hd95_defined": len(hd95_vals),
        "n_hd95_undefined": n_undefined,
        "n_empty_pred": int((sum_p == 0).sum()),
        "n_dice_zero": int((dice == 0.0).sum()),
    }
