"""Reporting helpers: console summaries, plots, and CSV/PNG artefacts.

This module centralises all visualisation and reporting tasks that are typically
performed after a training run:

* **Console summary** – prints mean ± std of segmentation metrics.
* **Classification report & confusion matrix** – uses scikit‑learn,
  saves a CSV file and a high‑resolution PNG plot.
* **Training curves** – loss and accuracy over epochs, saved as a PNG.
* **Metrics CSV** – one‑row CSV containing the final segmentation metrics.

All artefacts are stored under a configurable directory (default: ``Model_Evaluation``)
and are timestamped so that multiple runs can be distinguished.
"""

from __future__ import annotations                              # Enables forward references in type hints

import datetime                                               # For timestamp generation
from pathlib import Path                                      # Convenient path handling
from typing import Dict, Iterable                               # Type hints for collections

import matplotlib.pyplot as plt                               # Plotting library
import numpy as np                                            # Numerical utilities
import pandas as pd                                           # CSV handling
import seaborn as sns                                         # Prettier heatmaps (confusion matrix)
from sklearn.metrics import classification_report, confusion_matrix   # Metrics from scikit‑learn


# --------------------------------------------------------------------------- #
# Helper – timestamp string used for file naming.
# --------------------------------------------------------------------------- #

def _timestamp() -> str:
    """Return the current date‑time as ``YYYYMMDD-HHMMSS``."""
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


# --------------------------------------------------------------------------- #
# Print a nicely formatted table of segmentation metrics to stdout.
# --------------------------------------------------------------------------- #

def print_segmentation_metrics(metrics: Dict[str, float], mode: str = "pixels") -> None:
    """
    Display mean and standard deviation for the four core segmentation metrics.

    Parameters
    ----------
    metrics : dict
        Dictionary produced by ``evaluate_segmentation_metrics``. It must contain
        keys ``IoU``, ``Dice_Coefficient``, ``Pixel_Accuracy``, ``MAE`` as well as
        their associated ``*_std`` entries.
    mode : str, optional (default="pixels")
        Human‑readable label that will appear in the header (e.g. “pixels” or
        “samples”).  The function does not use this value for calculations,
        only for presentation.

    Returns
    -------
    None
    """
    print(f"\n=== Segmentation Metrics ({mode}) ===")
    print(f"IoU (Intersection over Union): {metrics['IoU']:.4f} ± {metrics['IoU_std']:.4f}")
    print(f"Dice Coefficient: {metrics['Dice_Coefficient']:.4f} ± {metrics['Dice_std']:.4f}")
    print(f"Pixel Accuracy: {metrics['Pixel_Accuracy']:.4f} ± {metrics['Pixel_Accuracy_std']:.4f}")
    print(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f} ± {metrics['MAE_std']:.4f}")

    # Build a small pandas DataFrame for a clean console table.
    summary = pd.DataFrame({
        "Metric": ["IoU", "Dice Coefficient", "Pixel Accuracy", "MAE"],
        "Mean": [metrics["IoU"], metrics["Dice_Coefficient"], metrics["Pixel_Accuracy"], metrics["MAE"]],
        "Std": [metrics["IoU_std"], metrics["Dice_std"], metrics["Pixel_Accuracy_std"], metrics["MAE_std"]],
    })
    print("\nSummary Table:")
    print(summary.to_string(index=False))


# --------------------------------------------------------------------------- #
# Save a sklearn classification report (CSV) and a confusion‑matrix plot (PNG).
# --------------------------------------------------------------------------- #

def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: str | Path = "Model_Evaluation",
    mode: str = "validation",
) -> None:
    """
    Generate a scikit‑learn classification report and a confusion‑matrix figure.

    The function writes two files to ``save_dir``:

    * ``classification_report_<mode>_<timestamp>.csv`` – the raw text report.
    * ``confusion_matrix_<mode>_<timestamp>.png`` – a heat‑map visualisation
      (Seaborn) saved at 300 dpi.

    Parameters
    ----------
    y_true : np.ndarray
        Ground‑truth binary labels (expected shape ``(n_samples,)`` or ``(n_samples, 1)``).
    y_pred : np.ndarray
        Predicted binary labels (same shape as ``y_true``).
    save_dir : str | Path, optional (default="Model_Evaluation")
        Directory where the artefacts will be stored.  It is created if it does not exist.
    mode : str, optional (default="validation")
        Identifier that will appear in the filenames (e.g., “train”, “val”, “test”).

    Returns
    -------
    None
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)          # Ensure the folder exists
    ts = _timestamp()

    # ------------------------------------------------------------------- #
    # 1️⃣ Classification report (textual)
    # ------------------------------------------------------------------- #
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Unburned (0)", "Burned (1)"],
        digits=4,
    )
    print(f"\n=== Classification Report ({mode}) ===")
    print(report)

    report_path = save_dir / f"classification_report_{mode}_{ts}.csv"
    report_path.write_text(report)                     # Store the raw string
    print(f"✅ Classification report saved to {report_path}")

    # ------------------------------------------------------------------- #
    # 2️⃣ Confusion matrix (visual)
    # ------------------------------------------------------------------- #
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Unburned", "Burned"],
        yticklabels=["Unburned", "Burned"],
    )
    plt.title(f"Confusion Matrix ({mode})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    cm_path = save_dir / f"confusion_matrix_{mode}_{ts}.png"
    plt.savefig(cm_path, dpi=300)                     # High‑resolution image
    print(f"✅ Confusion matrix plot saved to {cm_path}")
    plt.show()                                        # Show interactively (optional)


# --------------------------------------------------------------------------- #
# Plot training/validation loss and accuracy curves.
# --------------------------------------------------------------------------- #

def plot_metrics(
    train_losses: Iterable[float],
    val_losses: Iterable[float],
    train_accuracies: Iterable[float],
    val_accuracies: Iterable[float],
    save_dir: str | Path = "Model_Evaluation",
) -> None:
    """
    Produce a two‑panel figure containing loss and accuracy curves.

    The left panel shows *Loss* (training vs. validation); the right panel
    shows *Accuracy*.  Both plots are saved as PNG files under ``save_dir``
    with a timestamped filename and also displayed via ``plt.show()``.

    Parameters
    ----------
    train_losses : iterable of float
        Loss values obtained on the training set, one per epoch.
    val_losses : iterable of float
        Corresponding validation loss values.
    train_accuracies : iterable of float
        Accuracy (or any other scalar metric) on the training set per epoch.
    val_accuracies : iterable of float
        Same metric for the validation set.
    save_dir : str | Path, optional (default="Model_Evaluation")
        Destination folder for the PNG output.  Created if missing.

    Returns
    -------
    None
    """
    # Convert iterables to concrete lists so we can compute lengths / epochs.
    train_losses = list(train_losses)
    val_losses = list(val_losses)
    train_accuracies = list(train_accuracies)
    val_accuracies = list(val_accuracies)

    epochs = range(1, len(train_losses) + 1)          # Epoch numbers start at 1

    plt.figure(figsize=(14, 6))                       # Wide figure for side‑by‑side plots

    # ------------------- Loss subplot (left) -------------------
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, "bo-", label="Training Loss")
    plt.plot(epochs, val_losses, "ro-", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # ------------------- Accuracy subplot (right) -------------------
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, "bo-", label="Training Accuracy")
    plt.plot(epochs, val_accuracies, "ro-", label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()                               # Adjust spacing so labels don’t overlap

    # ------------------------------------------------------------------- #
    # Save the figure
    # ------------------------------------------------------------------- #
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"training_metrics_{_timestamp()}.png"
    plt.savefig(save_path, dpi=300)                  # 300 dpi → publication quality
    print(f"✅ Training metrics plot saved to {save_path}")
    plt.show()                                       # Optional interactive display


# --------------------------------------------------------------------------- #
# Persist a single‑row CSV containing the final segmentation metrics.
# --------------------------------------------------------------------------- #

def save_metrics_csv(metrics: Dict[str, float],
                     save_dir: str | Path = "Model_Evaluation") -> Path:
    """
    Write a one‑row CSV file that contains all segmentation metrics (mean and std).

    The file name is timestamped, e.g. ``segmentation_metrics_20231201-153045.csv``,
    and the directory ``save_dir`` is created if it does not already exist.

    Parameters
    ----------
    metrics : dict
        Dictionary with the same keys that ``evaluate_segmentation_metrics`` returns.
        Expected keys: ``IoU``, ``Dice_Coefficient``, ``Pixel_Accuracy``,
        ``MAE`` and their corresponding ``*_std`` entries.
    save_dir : str | Path, optional (default="Model_Evaluation")
        Folder where the CSV will be stored.

    Returns
    -------
    pathlib.Path
        The full path to the created CSV file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"segmentation_metrics_{_timestamp()}.csv"
    pd.DataFrame([metrics]).to_csv(path, index=False)   # One row, no index column
    print(f"\nMetrics saved to: {path}")
    return path
