"""Matplotlib visualization helpers for U‑Net predictions.

This module provides utilities to display probability maps and binary masks side‑by‑side using Matplotlib,
making it easy to inspect U‑Net outputs.
"""

from __future__ import annotations

import os  # needed to resolve file paths for titles in subplot captions

import matplotlib.pyplot as plt  # core plotting library
import numpy as np  # numerical array handling


def visualize_prediction(
    avg_pred: np.ndarray,
    full_mask: np.ndarray,
    new_image_path: str,
) -> None:
    """Display the probability map and the thresholded binary mask side‑by‑side.

    Args:
        avg_pred: 2D array of predicted probabilities (values between 0 and 1).
        full_mask: 2D array of binary mask (0 = unburned, 1 = burned) after thresholding.
        new_image_path: Path to the source image; used only for the figure title.

    Returns:
        None. Shows a Matplotlib window with two sub‑plots.
    """
    # Create a figure with two side‑by‑side axes and a sensible size.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Plot the probability map using a viridis colormap; fix colour limits to 0‑1.
    im1 = ax1.imshow(avg_pred, cmap="viridis", vmin=0, vmax=1)

    # Add a title that includes the basename of the input image for clarity.
    ax1.set_title(f"Probability Map for {os.path.basename(new_image_path)}")

    # Hide axis ticks and labels to focus on the raster content.
    ax1.axis("off")

    # Draw a colour bar for the probability map; adjust size and padding for readability.
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Plot the binary mask using a gray colormap; also fix limits to 0‑1.
    im2 = ax2.imshow(full_mask, cmap="gray", vmin=0, vmax=1)

    # Title indicates that this is a binary mask with a threshold of 0.5.
    ax2.set_title("Binary Mask (threshold=0.5)")

    # Hide axis ticks and labels for the mask subplot.
    ax2.axis("off")

    # Add a colour bar for the mask; set custom ticks to only show 0 and 1 values.
    plt.colorbar(im2, ax=ax2, ticks=[0, 1], fraction=0.046, pad=0.04)

    # Adjust subplot parameters so that titles, colour bars and margins are not clipped.
    plt.tight_layout()

    # Render the figure in an interactive window (blocking call).
    plt.show()
