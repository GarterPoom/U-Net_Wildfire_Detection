"""Segmentation evaluation metrics."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def calculate_iou(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Intersection-over-Union for boolean-castable masks."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float((intersection + smooth) / (union + smooth))


def calculate_dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Dice coefficient (F1) for boolean-castable masks."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    return float((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def calculate_pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """Fraction of correctly classified pixels."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    return float(np.sum(pred == target) / pred.size)


def calculate_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute error between a (possibly probabilistic) prediction and a binary target."""
    return float(np.mean(np.abs(pred.astype(np.float32) - target.astype(np.float32))))


def compute_accuracy(outputs: torch.Tensor, masks: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Pixel accuracy directly from raw model logits, suitable for training-loop use."""
    with torch.no_grad():
        preds = torch.sigmoid(outputs) >= threshold
        correct = (preds == masks.bool()).float().sum()
        return correct / masks.numel()


def evaluate_segmentation_metrics(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Run the model over ``dataloader`` and aggregate per-sample segmentation metrics.

    Returns:
        ``(metrics, all_preds_flat, all_masks_flat)``
        - ``metrics`` has mean and std for IoU, Dice, pixel accuracy, MAE.
        - The flat arrays are concatenated binary predictions and ground-truth
          masks, ready to feed into ``sklearn`` classification utilities.
    """
    model.eval()
    ious, dice_scores, pixel_accuracies, maes = [], [], [], []
    all_preds_flat, all_masks_flat = [], []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating segmentation metrics"):
            images = images.to(device)
            masks = masks.to(device).float()

            outputs = model(images)
            pred_probs = torch.sigmoid(outputs)
            pred_binary = (pred_probs > 0.5).float()

            pred_binary_np = pred_binary.cpu().numpy()
            pred_probs_np = pred_probs.cpu().numpy()
            masks_np = masks.cpu().numpy()

            all_preds_flat.append(pred_binary_np.flatten())
            all_masks_flat.append(masks_np.flatten())

            for i in range(pred_binary_np.shape[0]):
                pred_bin = pred_binary_np[i, 0]
                pred_prob = pred_probs_np[i, 0]
                mask = masks_np[i, 0]

                ious.append(calculate_iou(pred_bin, mask))
                dice_scores.append(calculate_dice_coefficient(pred_bin, mask))
                pixel_accuracies.append(calculate_pixel_accuracy(pred_bin, mask))
                maes.append(calculate_mae(pred_prob, mask))

    metrics = {
        "IoU": float(np.mean(ious)),
        "Dice_Coefficient": float(np.mean(dice_scores)),
        "Pixel_Accuracy": float(np.mean(pixel_accuracies)),
        "MAE": float(np.mean(maes)),
        "IoU_std": float(np.std(ious)),
        "Dice_std": float(np.std(dice_scores)),
        "Pixel_Accuracy_std": float(np.std(pixel_accuracies)),
        "MAE_std": float(np.std(maes)),
    }
    return metrics, np.concatenate(all_preds_flat), np.concatenate(all_masks_flat).astype(int)
