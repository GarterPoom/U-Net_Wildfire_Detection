"""Segmentation evaluation metrics.

This module gathers a small set of common segmentation metrics (IoU, Dice,
pixel accuracy and MAE) that can be computed on NumPy arrays or PyTorch tensors.
All functions are deliberately **stateless** – they only operate on the data you
pass in and return plain Python numbers (or a tuple for the batch‑wise evaluator).

The code is written to be easy to test: every function works with binary masks,
but it also accepts *probabilistic* predictions because the internal logic casts
to ``bool`` when appropriate.
"""
from __future__ import annotations                     # Enables forward references (e.g. Tuple, Dict)
from typing import Dict, Tuple                          # Type hints for return types
import numpy as np                                      # Core numeric library (used for all calculations)
import torch                                          # PyTorch – used for the DataLoader and model inference
from torch.utils.data import DataLoader                 # Standard DataLoader abstraction
from tqdm import tqdm                                   # Progress bar for batch loops

# --------------------------------------------------------------------------- #
# Helper functions – each works on **binary** (bool) masks.                     #
# --------------------------------------------------------------------------- #

def calculate_iou(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Compute Intersection‑over‑Union (IoU) for two binary masks.

    Parameters
    ----------
    pred : np.ndarray
        Predicted mask. Can be any numeric type; it will be cast to ``bool``.
    target : np.ndarray
        Ground‑truth mask – also cast to ``bool``.
    smooth : float, optional (default=1e-6)
        Small constant added to numerator and denominator to avoid division by zero.

    Returns
    -------
    float
        IoU = |pred ∩ target| / |pred ∪ target|.  Range is ``[0, 1]``.
    """
    pred = pred.astype(bool)               # Ensure boolean type for logical ops
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()   # Count pixels that are true in both
    union = np.logical_or(pred, target).sum()           # Count pixels that are true in either
    return float((intersection + smooth) / (union + smooth))


def calculate_dice_coefficient(pred: np.ndarray, target: np.ndarray,
                               smooth: float = 1e-6) -> float:
    """
    Compute the Dice coefficient (F1 score) for two binary masks.

    The Dice coefficient is twice the intersection divided by the sum of the
    cardinalities of the two sets:

        dice = 2·|pred ∩ target| / (|pred| + |target|)

    Parameters
    ----------
    pred : np.ndarray
        Predicted mask – any numeric type, cast to ``bool``.
    target : np.ndarray
        Ground‑truth mask – cast to ``bool``.
    smooth : float, optional (default=1e-6)
        Stabilises the denominator when both sets are empty.

    Returns
    -------
    float
        Dice coefficient in the interval ``[0, 1]``.  ``1`` means perfect overlap.
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    return float((2.0 * intersection + smooth) /
                 (pred.sum() + target.sum() + smooth))


def calculate_pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute the fraction of pixels that are classified correctly.

    This is a simple element‑wise equality check between two binary masks.

    Parameters
    ----------
    pred : np.ndarray
        Predicted mask – cast to ``bool``.
    target : np.ndarray
        Ground‑truth mask – cast to ``bool``.

    Returns
    -------
    float
        Pixel accuracy in the interval ``[0, 1]``.
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    return float(np.sum(pred == target) / pred.size)


def calculate_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Mean Absolute Error (MAE) between a *probabilistic* prediction
    and a binary target.

    The function first casts the inputs to ``float32`` so that the subtraction
    yields a meaningful error value even when the inputs are booleans.

    Parameters
    ----------
    pred : np.ndarray
        Prediction – can be probabilities (0‑1) or binary values.
    target : np.ndarray
        Ground‑truth binary mask.

    Returns
    -------
    float
        Mean absolute error, averaged over all pixels.
    """
    return float(np.mean(np.abs(pred.astype(np.float32) -
                             target.astype(np.float32))))


def compute_accuracy(outputs: torch.Tensor,
                     masks: torch.Tensor,
                     threshold: float = 0.5) -> torch.Tensor:
    """
    Compute pixel accuracy directly from raw model logits (no sigmoid needed).

    This helper is intended for use inside a training loop where you already have
    the raw output of the network (logits).  It applies a sigmoid, thresholds
    the result, and then compares it to the ground‑truth mask.

    Parameters
    ----------
    outputs : torch.Tensor
        Model logits with shape ``(B, C, H, W)`` – here ``C`` is assumed to be 1.
    masks : torch.Tensor
        Ground‑truth masks (binary) with shape ``(B, 1, H, W)`` or ``(B, H, W)``.
    threshold : float, optional (default=0.5)
        Decision threshold for converting sigmoid probabilities to binary predictions.

    Returns
    -------
    torch.Tensor
        A scalar tensor containing the fraction of correctly classified pixels
        (i.e., ``correct / total_pixels``).  The dtype is ``torch.float32``.
    """
    with torch.no_grad():                                   # No gradient tracking needed
        probs = torch.sigmoid(outputs)                     # Convert logits → probabilities
        preds = (probs >= threshold).float()              # Binary predictions (0/1)
        correct = (preds == masks.bool()).float().sum()   # Count matching pixels
        return correct / masks.numel()                    # Return fraction


# --------------------------------------------------------------------------- #
# Batch‑wise evaluator – gathers per‑sample metrics and also returns flat arrays.
# --------------------------------------------------------------------------- #

def evaluate_segmentation_metrics(model: torch.nn.Module,
                                  dataloader: DataLoader,
                                  device: torch.device) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Run ``model`` over the entire ``dataloader`` and aggregate segmentation metrics.

    The function performs a forward pass (with ``torch.no_grad()``), computes
    IoU, Dice, pixel accuracy and MAE for each sample, and also concatenates all
    predictions and ground‑truth masks into two flat NumPy arrays.  Those flat
    arrays can later be fed to scikit‑learn classification metrics (e.g.
    ``classification_report``) if desired.

    Parameters
    ----------
    model : torch.nn.Module
        Trained segmentation network (must implement a forward that returns logits).
    dataloader : torch.utils.data.DataLoader
        Iterable that yields ``(images, masks)`` tuples.  Images are expected to be
        tensors of shape ``(B, C, H, W)``; masks should be single‑channel binary
        tensors.
    device : torch.device
        Device on which the model and data should reside (e.g. ``torch.device('cuda')``).

    Returns
    -------
    Tuple[Dict[str, float], np.ndarray, np.ndarray]
        - **metrics** – a dictionary containing mean and standard deviation for each
          metric (IoU, Dice, Pixel Accuracy, MAE).
        - **all_preds_flat** – 1‑D NumPy array of binary predictions (flattened across the batch).
        - **all_masks_flat** – 1‑D NumPy array of ground‑truth masks (also flattened).

    Notes
    -----
    * The function assumes the model outputs are *logits* (i.e. not passed through a
      sigmoid).  Internally ``torch.sigmoid`` is applied with a fixed threshold of
      0.5.
    * MAE uses the raw probability values (before thresholding) to penalise
      mis‑calibrated confidence.
    """
    model.eval()                                            # Switch to evaluation mode (disables dropout, etc.)
    ious, dice_scores, pixel_accuracies, maes = [], [], [], []   # Per‑sample metric containers

    all_preds_flat, all_masks_flat = [], []                  # Lists that will be concatenated later

    with torch.no_grad():                                   # No gradient tracking – saves memory & speed
        for images, masks in tqdm(dataloader,
                                  desc="Evaluating segmentation metrics"):
            # Move data to the correct device
            images = images.to(device)
            masks = masks.to(device).float()                # Ensure float type for MAE

            # Forward pass – model returns raw logits
            outputs = model(images)

            # Convert logits → probabilities (sigmoid) and then binary predictions
            pred_probs = torch.sigmoid(outputs)              # Shape: (B, 1, H, W)
            pred_binary = (pred_probs > 0.5).float()        # Binary mask (0/1)

            # Move everything back to CPU for NumPy operations
            pred_binary_np = pred_binary.cpu().numpy()
            pred_probs_np = pred_probs.cpu().numpy()
            masks_np = masks.cpu().numpy()

            # ------------------------------------------------------------------- #
            #   Per‑sample metric computation (loop over batch dimension)
            # ------------------------------------------------------------------- #
            for i in range(pred_binary_np.shape[0]):          # Iterate over samples in the batch
                pred_bin = pred_binary_np[i, 0]               # Binary mask for sample *i*
                pred_prob = pred_probs_np[i, 0]               # Probability map (still [0,1])
                mask = masks_np[i, 0]                         # Ground‑truth binary mask

                # Compute individual metrics and store them
                ious.append(calculate_iou(pred_bin, mask))
                dice_scores.append(calculate_dice_coefficient(pred_bin, mask))
                pixel_accuracies.append(calculate_pixel_accuracy(pred_bin, mask))
                maes.append(calculate_mae(pred_prob, mask))

            # ------------------------------------------------------------------- #
            #   Accumulate flat arrays for possible downstream use (e.g. sklearn)
            # ------------------------------------------------------------------- #
            all_preds_flat.append(pred_binary_np.flatten())
            all_masks_flat.append(masks_np.flatten())

    # --------------------------------------------------------------- #
    #   Aggregate statistics
    # --------------------------------------------------------------- #
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

    # Concatenate the per‑sample flattened arrays into single 1‑D vectors
    all_preds_flat = np.concatenate(all_preds_flat)
    all_masks_flat = np.concatenate(all_masks_flat).astype(int)

    return metrics, all_preds_flat, all_masks_flat
