"""Reporting helpers: console summaries, plots, and CSV/PNG artefacts."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def print_segmentation_metrics(metrics: Dict[str, float], mode: str = "pixels") -> None:
    """Print metric means/stds and a tabular summary to stdout."""
    print(f"\n=== Segmentation Metrics ({mode}) ===")
    print(f"IoU (Intersection over Union): {metrics['IoU']:.4f} ± {metrics['IoU_std']:.4f}")
    print(f"Dice Coefficient: {metrics['Dice_Coefficient']:.4f} ± {metrics['Dice_std']:.4f}")
    print(f"Pixel Accuracy: {metrics['Pixel_Accuracy']:.4f} ± {metrics['Pixel_Accuracy_std']:.4f}")
    print(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f} ± {metrics['MAE_std']:.4f}")

    summary = pd.DataFrame({
        "Metric": ["IoU", "Dice Coefficient", "Pixel Accuracy", "MAE"],
        "Mean": [metrics["IoU"], metrics["Dice_Coefficient"], metrics["Pixel_Accuracy"], metrics["MAE"]],
        "Std": [metrics["IoU_std"], metrics["Dice_std"], metrics["Pixel_Accuracy_std"], metrics["MAE_std"]],
    })
    print("\nSummary Table:")
    print(summary.to_string(index=False))


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: str | Path = "Model_Evaluation",
    mode: str = "validation",
) -> None:
    """Generate a sklearn classification report and confusion-matrix figure.

    Outputs are saved to ``save_dir`` with a timestamp suffix and the plot is
    also displayed via ``plt.show()``.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = _timestamp()

    report = classification_report(y_true, y_pred, target_names=["Unburned (0)", "Burned (1)"])
    print(f"\n=== Classification Report ({mode}) ===")
    print(report)

    report_path = save_dir / f"classification_report_{mode}_{ts}.csv"
    report_path.write_text(report)
    print(f"✅ Classification report saved to {report_path}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Unburned", "Burned"],
        yticklabels=["Unburned", "Burned"],
    )
    plt.title(f"Confusion Matrix ({mode})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    cm_path = save_dir / f"confusion_matrix_{mode}_{ts}.png"
    plt.savefig(cm_path, dpi=300)
    print(f"✅ Confusion matrix plot saved to {cm_path}")
    plt.show()


def plot_metrics(
    train_losses: Iterable[float],
    val_losses: Iterable[float],
    train_accuracies: Iterable[float],
    val_accuracies: Iterable[float],
    save_dir: str | Path = "Model_Evaluation",
) -> None:
    """Plot loss/accuracy curves for the training run and persist the figure."""
    train_losses = list(train_losses)
    val_losses = list(val_losses)
    train_accuracies = list(train_accuracies)
    val_accuracies = list(val_accuracies)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, "bo-", label="Training Loss")
    plt.plot(epochs, val_losses, "ro-", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, "bo-", label="Training Accuracy")
    plt.plot(epochs, val_accuracies, "ro-", label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"training_metrics_{_timestamp()}.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Training metrics plot saved to {save_path}")
    plt.show()


def save_metrics_csv(metrics: Dict[str, float], save_dir: str | Path = "Model_Evaluation") -> Path:
    """Persist a single-row metrics CSV to ``save_dir`` and return its path."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"segmentation_metrics_{_timestamp()}.csv"
    pd.DataFrame([metrics]).to_csv(path, index=False)
    print(f"\nMetrics saved to: {path}")
    return path
