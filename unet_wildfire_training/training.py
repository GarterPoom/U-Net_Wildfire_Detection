"""Training orchestration for the U-Net wildfire model."""

from __future__ import annotations

import warnings
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet_wildfire_training.config import TrainingConfig
from unet_wildfire_training.data import build_dataloaders, match_raster_shapefile
from unet_wildfire_training.losses import DownsampledBCEWithLogitsLoss
from unet_wildfire_training.metrics import compute_accuracy, evaluate_segmentation_metrics
from unet_wildfire_training.model import UNet
from unet_wildfire_training.reporting import (
    plot_metrics,
    print_segmentation_metrics,
    save_classification_report,
    save_metrics_csv,
)
from unet_wildfire_training.system_info import print_device_info


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_label: str,
) -> Tuple[float, float, int]:
    """Run one training epoch.

    Returns ``(mean_loss, mean_accuracy, skipped_batches)`` where the means are
    computed only over samples actually processed (batches that produced NaN/Inf
    loss or hit CUDA OOM are skipped and excluded from the denominator).
    """
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    processed_samples = 0
    skipped_batches = 0

    for batch_idx, (images, masks) in enumerate(tqdm(loader, desc=f"{epoch_label} [Train]", leave=True)):
        try:
            images = images.to(device)
            masks = masks.to(device).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n  [{epoch_label}, Batch {batch_idx}] WARNING: NaN/Inf loss — skipping batch.", flush=True)
                skipped_batches += 1
                continue

            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_acc += compute_accuracy(outputs, masks).item() * batch_size
            processed_samples += batch_size

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"\n  [{epoch_label}, Batch {batch_idx}] CUDA OOM — clearing cache and skipping batch.", flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                skipped_batches += 1
                continue
            print(f"\n  [{epoch_label}, Batch {batch_idx}] RuntimeError: {exc}", flush=True)
            raise

    denom = max(processed_samples, 1)
    return running_loss / denom, running_acc / denom, skipped_batches


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch_label: str,
) -> Tuple[float, float]:
    """Run one validation pass and return ``(mean_loss, mean_accuracy)``."""
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    processed_samples = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc=f"{epoch_label} [Val]", leave=True):
            images = images.to(device)
            masks = masks.to(device).float()

            outputs = model(images)
            loss = criterion(outputs, masks)
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_acc += compute_accuracy(outputs, masks).item() * batch_size
            processed_samples += batch_size

    denom = max(processed_samples, 1)
    return running_loss / denom, running_acc / denom


def train_model(config: TrainingConfig) -> None:
    """End-to-end training entry point.

    Discovers raster/label pairs, builds loaders, trains the U-Net for the
    configured number of epochs, then runs evaluation and persists model
    weights plus evaluation artefacts.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="pyogrio.raw")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_device_info(device)

    pairs = match_raster_shapefile(config.image_dir, config.label_dir)
    if not pairs:
        print("No data", flush=True)
        return

    print("\nProcessing all tiles with spatial downsampling...", flush=True)
    train_loader, val_loader, max_channels = build_dataloaders(config, pairs)

    model = UNet(n_channels=max_channels, n_classes=1).to(device)
    criterion = DownsampledBCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(config.num_epochs):
        label = f"Epoch {epoch + 1}/{config.num_epochs}"
        print(f"\n[{label}] Starting training phase...", flush=True)

        train_loss, train_acc, skipped = train_one_epoch(
            model, train_loader, criterion, optimizer, device, label
        )
        if skipped:
            print(f"  [{label}] Skipped {skipped} batch(es) during training.", flush=True)

        print(f"[{label}] Starting validation phase...", flush=True)
        val_loss, val_acc = validate(model, val_loader, criterion, device, label)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(
            f"{label} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}",
            flush=True,
        )

    plot_metrics(
        train_losses, val_losses, train_accuracies, val_accuracies,
        save_dir=config.evaluation_dir,
    )

    print("\nEvaluating model with segmentation metrics...")
    pixel_metrics, y_pred, y_true = evaluate_segmentation_metrics(model, val_loader, device)
    print_segmentation_metrics(pixel_metrics, mode="pixels")
    save_classification_report(y_true, y_pred, save_dir=config.evaluation_dir, mode="validation")
    save_metrics_csv(pixel_metrics, save_dir=config.evaluation_dir)

    config.export_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.model_path()
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved successfully to {model_path}")
