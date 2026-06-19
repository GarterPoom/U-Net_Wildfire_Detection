"""Matplotlib visualization helpers for U-Net predictions."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np


def visualize_prediction(
    avg_pred: np.ndarray,
    full_mask: np.ndarray,
    new_image_path: str,
) -> None:
    """Display the probability map and the thresholded binary mask side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    im1 = ax1.imshow(avg_pred, cmap="viridis", vmin=0, vmax=1)
    ax1.set_title(f"Probability Map for {os.path.basename(new_image_path)}")
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(full_mask, cmap="gray", vmin=0, vmax=1)
    ax2.set_title("Binary Mask (threshold=0.5)")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, ticks=[0, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
