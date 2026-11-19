"""
Training pipeline for MPF-Net / FinalNet.
"""

import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.modules import Uncertainty, EMA
from .losses import segmentation_loss
from ..evaluation.metrics import compute_classification_metrics, compute_dice_per_image


def build_optimizer_and_scheduler(model: nn.Module, uncertainty: nn.Module, total_steps: int, warmup_steps: int, lr: float = 2e-4, wd: float = 1e-4):
    """
    Constructs AdamW optimizer with decoupled learning rate for uncertainty parameters
    and a cosine decay learning rate scheduler with linear warmup.
    """
    param_groups = [
        {"params": model.parameters(), "weight_decay": wd, "lr": lr},
        {"params": uncertainty.parameters(), "weight_decay": 0.0, "lr": 1e-3},
    ]
    optimizer = torch.optim.AdamW(param_groups)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def train_single_epoch(
    model: nn.Module,
    uncertainty: nn.Module,
    ema: EMA,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    criterion_cls: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """
    Runs one training epoch with mixed precision, gradient clipping, EMA updates,
    and uncertainty task balancing.
    """
    model.train()
    total_loss = 0.0

    for images, masks, labels, has_masks in dataloader:
        images = images.to(device, non_blocking=True).expand(-1, 3, -1, -1)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        has_masks = has_masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device.type, enabled=(device.type == "cuda")):
            out = model(images)
            l_seg = segmentation_loss(out, masks, has_masks)
            l_cls = criterion_cls(out["cls"], labels)
            loss = uncertainty(l_seg, l_cls)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        ema.update(model)
        total_loss += float(loss.detach().item())

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluates model on validation fold: returns classification metrics and mean Dice.
    """
    model.eval()
    all_probs = []
    all_targets = []
    all_dices = []

    for images, masks, labels, has_masks in dataloader:
        images = images.to(device, non_blocking=True).expand(-1, 3, -1, -1)
        with torch.autocast(device.type, enabled=(device.type == "cuda")):
            out = model(images)

        probs = torch.softmax(out["cls"].float(), dim=1).cpu().numpy()
        seg_probs = torch.sigmoid(out["seg"].float())[:, 0].cpu().numpy()

        all_probs.append(probs)
        all_targets.append(labels.numpy())

        dices = compute_dice_per_image(seg_probs, masks[:, 0].numpy(), has_masks.numpy())
        all_dices.append(dices)

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    dice_arr = np.concatenate(all_dices, axis=0) if all_dices else np.array([])

    metrics = compute_classification_metrics(y_true, y_prob)
    metrics["dice"] = float(np.mean(dice_arr)) if len(dice_arr) else 0.0
    metrics["val_score"] = metrics["f1"] + metrics["dice"]

    return metrics
