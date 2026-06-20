"""Training orchestration for the U-Net wildfire model."""

from __future__ import annotations # Allows for postponed evaluation of type annotations

import warnings  # Import warnings to suppress unnecessary console noise
from typing import Tuple  # Import Tuple for type hinting

import torch  # Import PyTorch for neural network operations
import torch.nn as nn  # Import neural network modules from PyTorch
from torch.utils.data import DataLoader  # Import DataLoader to handle batching
from tqdm import tqdm  # Import tqdm for the visual progress bars

import logging  # Import logging so we can write to our dynamic log file

from unet_wildfire_training.config import TrainingConfig  # Import config
from unet_wildfire_training.data import build_dataloaders, match_raster_shapefile  # Import data utilities
from unet_wildfire_training.losses import DownsampledBCEWithLogitsLoss  # Import loss function
from unet_wildfire_training.metrics import compute_accuracy, evaluate_segmentation_metrics  # Import metrics
from unet_wildfire_training.model import UNet  # Import the model architecture
from unet_wildfire_training.reporting import ( # Import reporting tools
    plot_metrics,
    print_segmentation_metrics,
    save_classification_report,
    save_metrics_csv,
)
from unet_wildfire_training.system_info import print_device_info  # Import hardware info utility

def train_one_epoch(
    model: nn.Module, # The neural network model
    loader: DataLoader, # The training data loader
    criterion: nn.Module, # The loss function
    optimizer: torch.optim.Optimizer, # The optimizer (e.g., Adam)
    device: torch.device, # The hardware (CPU or CUDA)
    epoch_label: str, # A string label for the current epoch
) -> Tuple[float, float, int]:
    model.train() # Set model to training mode (enables Dropout/BatchNorm training)
    running_loss = 0.0 # Initialize total loss accumulator for the epoch
    running_acc = 0.0 # Initialize total accuracy accumulator for the epoch
    processed_samples = 0 # Initialize a counter for total samples processed
    skipped_batches = 0 # Initialize a counter for batches that failed (e.g., OOM)

    for batch_idx, (images, masks) in enumerate(tqdm(loader, desc=f"{epoch_label} [Train]", leave=True)):
        try: # Use try-except to catch batch-level errors without stopping the whole training
            images = images.to(device) # Move input images to the GPU/CPU
            masks = masks.to(device).float() # Move ground truth masks to the GPU/CPU

            optimizer.zero_grad() # Reset gradients from the previous step
            outputs = model(images) # Perform forward pass to get model predictions
            loss = criterion(outputs, masks) # Calculate the difference between prediction and truth

            if torch.isnan(loss) or torch.isinf(loss): # Check if the loss calculation failed (NaN/Infinity)
                logging.warning(f"[{epoch_label}, Batch {batch_idx}] WARNING: NaN/Inf loss — skipping batch.") # Log the warning
                skipped_batches += 1 # Increment the skip counter
                continue # Skip the rest of the loop and move to the next batch

            loss.backward() # Perform backpropagation to calculate gradients
            optimizer.step() # Update the model weights using the optimizer

            batch_size = images.size(0) # Get the number of items in the current batch
            running_loss += loss.item() * batch_size # Add weighted loss to total
            running_acc += compute_accuracy(outputs, masks).item() * batch_size # Add weighted accuracy to total
            processed_samples += batch_size # Increment the count of processed pixels/images

        except RuntimeError as exc: # Catch runtime errors like CUDA Out Of Memory
            if "out of memory" in str(exc).lower(): # Specifically check if it's a memory error
                logging.warning(f"[{epoch_label}, Batch {batch_idx}] CUDA OOM — clearing cache and skipping batch.") # Log OOM error
                if device.type == "cuda": # If using NVIDIA GPU
                    torch.cuda.empty_cache() # Clear the GPU cache to attempt to recover
                skipped_batches += 1 # Increment the skip counter
                continue # Skip this batch and continue training
            logging.error(f"[{epoch_label}, Batch {batch_idx}] RuntimeError: {exc}") # Log other runtime errors
            raise # Re-raise the error to stop training if it's a serious issue

    denom = max(processed_samples, 1) # Calculate denominator (prevent division by zero)
    return running_loss / denom, running_acc / denom, skipped_batches # Return mean loss, accuracy, and skips

def validate(
    model: nn.Module, # The neural network model
    loader: DataLoader, # The validation data loader
    criterion: nn.Module, # The loss function
    device: torch.device, # The hardware
    epoch_label: str, # Epoch label
) -> Tuple[float, float]:
    model.eval() # Set model to evaluation mode (disables Dropout/BatchNorm training)
    running_loss = 0.0 # Initialize loss accumulator
    running_acc = 0.0 # Initialize accuracy accumulator
    processed_samples = 0 # Initialize sample counter

    with torch.no_grad(): # Disable gradient calculation to save memory and speed up validation
        for images, masks in tqdm(loader, desc=f"{epoch_label} [Val]", leave=True):
            images = images.to(device) # Move images to device
            masks = masks.to(device).float() # Move masks to device

            outputs = model(images) # Forward pass
            loss = criterion(outputs, masks) # Calculate loss
            if torch.isnan(loss) or torch.isinf(loss): # Check for numerical instability
                continue # Skip bad batches

            batch_size = images.size(0) # Get batch size
            running_loss += loss.item() * batch_size # Accumulate loss
            running_acc += compute_accuracy(outputs, masks).item() * batch_size # Accumulate accuracy
            processed_samples += batch_size # Increment processed samples

    denom = max(processed_samples, 1) # Avoid division by zero
    return running_loss / denom, running_acc / denom # Return mean loss and accuracy

def train_model(config: TrainingConfig) -> None:
    """End-to-end training entry point."""
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="pyogrio.raw") # Suppress specific warnings

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Select GPU if available
    print_device_info(device) # Log system/device info to the console

    pairs = match_raster_shapefile(config.image_dir, config.label_dir) # Find matching image/label files
    if not pairs: # If no matches are found
        logging.error("No data: No matching raster/shapefile pairs found.") # Log the error
        return # Stop the script

    logging.info("Processing all tiles with spatial downsampling...") # Log the start of data processing
    train_loader, val_loader, max_channels = build_dataloaders(config, pairs) # Build the data pipelines

    model = UNet(n_channels=max_channels, n_classes=1).to(device) # Initialize the model on the device
    criterion = DownsampledBCEWithLogitsLoss() # Initialize the specialized loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate) # Initialize Adam optimizer

    train_losses, val_losses = [], [] # Lists to store training loss history
    train_accuracies, val_accuracies = [], [] # Lists to store training accuracy history

    for epoch in range(config.num_epochs): # Loop through the specified number of epochs
        label = f"Epoch {epoch + 1}/{config.num_epochs}" # Create label for current epoch
        logging.info(f"[{label}] Starting training phase...") # Log the start of training phase

        train_loss, train_acc, skipped = train_one_epoch( # Run the training loop for one epoch
            model, train_loader, criterion, optimizer, device, label
        )
        if skipped: # If any batches were skipped due to errors
            logging.warning(f"[{label}] Skipped {skipped} batch(es) during training.") # Log warning

        logging.info(f"[{label}] Starting validation phase...") # Log the start of validation
        val_loss, val_acc = validate(model, val_loader, criterion, device, label) # Run validation

        train_losses.append(train_loss) # Save loss to history
        val_losses.append(val_loss) # Save loss to history
        train_accuracies.append(train_acc) # Save accuracy to history
        val_accuracies.append(val_acc) # Save accuracy to history

        # Log summary of the epoch results
        logging.info(f"[{label}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save the training curves (Loss/Accuracy plots)
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_dir=config.evaluation_dir)

    logging.info("Evaluating model with segmentation metrics...") # Log evaluation start
    pixel_metrics, y_pred, y_true = evaluate_segmentation_metrics(model, val_loader, device) # Calculate IoU, Dice, etc.
    print_segmentation_metrics(pixel_metrics, mode="pixels") # Print metrics to console
    save_classification_report(y_true, y_pred, save_dir=config.evaluation_dir, mode="validation") # Save CM and Report
    save_metrics_csv(pixel_metrics, save_dir=config.evaluation_dir) # Save metrics to CSV

    config.export_dir.mkdir(parents=True, exist_ok=True) # Ensure export directory exists
    model_path = config.model_path() # Get path for the model file
    torch.save(model.state_dict(), model_path) # Save the trained model weights to the disk
    logging.info(f"SUCCESS: Model saved successfully to {model_path}") # Log successful save
