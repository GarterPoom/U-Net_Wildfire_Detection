# Core Python utilities
import os  # For file and directory operations
import sys # For system-specific parameters and functions

# Set environment variable to allow multiple OpenMP runtimes on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' 

# Set PROJ_LIB to ensure the correct PROJ database is used from the Conda environment
proj_path = os.path.join(sys.prefix, 'Library', 'share', 'proj')
if os.path.exists(proj_path):
    os.environ['PROJ_LIB'] = proj_path

# Set environment variables for PROJ data and threading issues on Windows
import glob      # For pattern-based file searching
import warnings  # For suppressing warnings
import re        # For date extraction from filenames

# PyTorch (Deep Learning Framework)
import torch
import torch.nn as nn              # Neural network building blocks (layers, activations, etc.)
import torch.optim as optim        # Optimizers (SGD, Adam, etc.) for training models

# System utilities
import platform                    # Get system/OS information
import psutil                      # Monitor system resource usage (CPU, memory, etc.)
import datetime                    # Work with timestamps and logs

# PyTorch Dataset utilities
from torch.utils.data import Dataset, DataLoader  # Dataset & batching utilities for model training

# Raster (geospatial imagery) processing
import rasterio                                    # For reading/writing geospatial raster data
from rasterio.windows import Window                # Read specific windows (subregions) of raster files
from rasterio.features import rasterize            # Convert vector geometries into rasterized masks

# Vector geospatial data handling
import geopandas as gpd                            # Handle shapefiles/GeoJSON and geospatial vector data
from shapely.geometry import mapping, box          # Geometry operations (e.g., bounding boxes, polygons)

# Numerical & image processing
import numpy as np                                 # Core numerical computing library
from skimage.transform import resize               # Resize images (useful for preprocessing)

# Progress tracking
from tqdm import tqdm                              # Progress bars for loops

# Data handling & visualization
import pandas as pd                                # Tabular data manipulation
import seaborn as sns                              # Statistical plotting
import matplotlib.pyplot as plt                    # General plotting

# Evaluation metrics
from sklearn.model_selection import train_test_split                  # Train-test data splitting
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------ U-Net Modules ------------------
class DoubleConv(nn.Module):
    """
    Double Convolution Block for U-Net Architecture.

    This module applies two consecutive convolutional operations, each followed by
    batch normalization and ReLU activation. This pattern is fundamental in U-Net
    for feature extraction and processing at each resolution level.

    The block consists of:
    1. Conv2d (3x3, padding=1) -> BatchNorm2d -> ReLU
    2. Conv2d (3x3, padding=1) -> BatchNorm2d -> ReLU

    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB images)
        out_channels (int): Number of output channels/feature maps

    Attributes:
        double_conv (nn.Sequential): Sequential container holding the two conv blocks
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()  # Initialize the parent nn.Module class
        # Create a sequential block with two convolution-normalization-activation layers
        self.double_conv = nn.Sequential(
            # First convolution: 3x3 kernel, same padding to maintain spatial dimensions
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # Batch normalization to stabilize training and reduce internal covariate shift
            nn.BatchNorm2d(out_channels),
            # ReLU activation for non-linearity
            nn.ReLU(inplace=True),
            # Second convolution: another 3x3 kernel with same padding
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # Another batch normalization layer
            nn.BatchNorm2d(out_channels),
            # Final ReLU activation
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Pass input tensor through the sequential double convolution block
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downsampling Block for U-Net Encoder.

    This module performs spatial downsampling in the encoder path of U-Net by:
    1. Applying max pooling to reduce spatial dimensions by factor of 2
    2. Following with a DoubleConv block to process the downsampled features

    This reduces the spatial resolution while increasing the number of feature channels,
    allowing the network to capture both local and global context.

    Args:
        in_channels (int): Number of input channels from previous layer
        out_channels (int): Number of output channels after downsampling (typically 2x in_channels)

    Attributes:
        maxpool_conv (nn.Sequential): Sequential container with MaxPool2d followed by DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()  # Initialize the parent nn.Module class
        # Sequential block: MaxPool for downsampling, then DoubleConv for feature processing
        self.maxpool_conv = nn.Sequential(
            # Max pooling with 2x2 kernel and stride 2 to halve spatial dimensions
            nn.MaxPool2d(2),
            # Double convolution block to process downsampled features
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        # Apply max pooling followed by double convolution
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling Block for U-Net Decoder.

    This module performs spatial upsampling in the decoder path of U-Net by:
    1. Using transposed convolution to increase spatial dimensions by factor of 2
    2. Concatenating with skip connection features from encoder
    3. Applying DoubleConv to process the combined features

    The skip connections help preserve spatial information lost during downsampling.

    Args:
        in_channels (int): Number of input channels from previous decoder layer
        out_channels (int): Number of output channels after upsampling

    Attributes:
        up (nn.ConvTranspose2d): Transposed convolution layer for upsampling
        conv (DoubleConv): Double convolution block for feature processing after concatenation
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()  # Initialize the parent nn.Module class
        # Transposed convolution for upsampling: reduces channels by 2, increases spatial size by 2
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # Double convolution block to process concatenated features
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # Upsample the input from previous decoder layer
        x1 = self.up(x1)
        
        # Calculate padding needed to match skip connection spatial dimensions
        # x2 is the skip connection from encoder at same spatial resolution
        diffY = x2.size()[2] - x1.size()[2]  # Height difference
        diffX = x2.size()[3] - x1.size()[3]  # Width difference
        
        # Apply symmetric padding to upsampled tensor to match skip connection size
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # Concatenate upsampled features with skip connection along channel dimension
        x = torch.cat([x2, x1], dim=1)
        # Process concatenated features through double convolution
        return self.conv(x)


class OutConv(nn.Module):
    """
    Output Convolution Layer for U-Net.

    This is the final layer of U-Net that produces the segmentation logits.
    It uses a 1x1 convolution to map the feature maps to the number of classes
    without changing spatial dimensions.

    Args:
        in_channels (int): Number of input channels from final decoder block
        out_channels (int): Number of output classes (e.g., 1 for binary segmentation)

    Attributes:
        conv (nn.Conv2d): 1x1 convolution layer for final prediction
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()  # Initialize the parent nn.Module class
        # 1x1 convolution to produce class logits, no spatial change
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Apply 1x1 convolution to get final logits
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net Architecture for Semantic Segmentation.

    U-Net is a convolutional neural network designed for biomedical image segmentation
    that consists of a contracting encoder path and an expansive decoder path.
    The encoder captures context through successive downsampling, while the decoder
    enables precise localization through upsampling and skip connections.

    Architecture:
    - Encoder: 4 downsampling blocks (64 -> 128 -> 256 -> 512 -> 1024 channels)
    - Bottleneck: Additional downsampling to 1024 channels
    - Decoder: 4 upsampling blocks with skip connections (1024 -> 512 -> 256 -> 128 -> 64 channels)
    - Output: 1x1 convolution for final segmentation logits

    Args:
        n_channels (int): Number of input channels (e.g., 3 for RGB images)
        n_classes (int): Number of output classes (e.g., 1 for binary segmentation)

    Attributes:
        inc (DoubleConv): Initial convolution block
        down1-down4 (Down): Encoder downsampling blocks
        up1-up4 (Up): Decoder upsampling blocks with skip connections
        outc (OutConv): Final output convolution
    """
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()  # Initialize the parent nn.Module class
        # Encoder path: progressive downsampling and feature extraction
        self.inc = DoubleConv(n_channels, 64)  # Initial 64-channel features
        self.down1 = Down(64, 128)    # Downsample to 128 channels
        self.down2 = Down(128, 256)   # Downsample to 256 channels
        self.down3 = Down(256, 512)   # Downsample to 512 channels
        self.down4 = Down(512, 1024)  # Downsample to 1024 channels (bottleneck)
        
        # Decoder path: progressive upsampling with skip connections
        self.up1 = Up(1024, 512)  # Upsample with skip from down3
        self.up2 = Up(512, 256)   # Upsample with skip from down2
        self.up3 = Up(256, 128)   # Upsample with skip from down1
        self.up4 = Up(128, 64)    # Upsample with skip from inc
        
        # Final output layer: map to class logits
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder: capture context through downsampling
        x1 = self.inc(x)      # Initial features: [B, 64, H, W]
        x2 = self.down1(x1)   # Level 1: [B, 128, H/2, W/2]
        x3 = self.down2(x2)   # Level 2: [B, 256, H/4, W/4]
        x4 = self.down3(x3)   # Level 3: [B, 512, H/8, W/8]
        x5 = self.down4(x4)   # Bottleneck: [B, 1024, H/16, W/16]
        
        # Decoder: recover spatial resolution with skip connections
        x = self.up1(x5, x4)  # Level 3 up: [B, 512, H/8, W/8] + skip from x4
        x = self.up2(x, x3)   # Level 2 up: [B, 256, H/4, W/4] + skip from x3
        x = self.up3(x, x2)   # Level 1 up: [B, 128, H/2, W/2] + skip from x2
        x = self.up4(x, x1)   # Final up: [B, 64, H, W] + skip from x1
        
        # Output: generate segmentation logits
        logits = self.outc(x)  # [B, n_classes, H, W]
        return logits

# ------------------ Custom Loss for Class Imbalance ------------------
class BalancedBCEWithLogitsLoss(nn.Module):
    """
    Balanced Binary Cross Entropy Loss with Logits.

    This custom loss function addresses class imbalance in segmentation tasks by
    computing BCE loss separately for positive and negative pixels, then averaging
    them equally. This prevents the majority class from dominating the loss.

    Traditional BCE would weight loss by class frequency, but this implementation
    gives equal importance to positive and negative classes regardless of their
    proportions in the image.

    The loss is computed as:
    loss = (mean_positive_loss + mean_negative_loss) / 2

    This is particularly useful for wildfire segmentation where burned areas
    (positive class) are typically much smaller than unburned areas.

    Args:
        None (no parameters required for initialization)

    Returns:
        torch.Tensor: Balanced BCE loss value
    """
    def __init__(self):
        super(BalancedBCEWithLogitsLoss, self).__init__()  # Initialize parent nn.Module

    def forward(self, logits, targets):
        # Flatten predictions and targets to 1D tensors for pixel-wise loss computation
        logits = logits.view(-1)    # Shape: [N*H*W] where N is batch size
        targets = targets.view(-1)  # Shape: [N*H*W]
        
        # Compute BCE loss for each pixel without reduction (keep per-pixel losses)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )  # Shape: [N*H*W]
        
        # Create masks for positive (burned) and negative (unburned) pixels
        pos_mask = (targets == 1)  # Boolean mask for pixels belonging to positive class
        neg_mask = (targets == 0)  # Boolean mask for pixels belonging to negative class
        
        # Extract losses for each class separately
        pos_loss = bce_loss[pos_mask]  # Losses only for positive pixels
        neg_loss = bce_loss[neg_mask]  # Losses only for negative pixels
        
        # Handle edge cases where one class might be absent in the batch
        if pos_loss.numel() == 0:  # No positive pixels in this batch
            return neg_loss.mean()  # Return only negative class loss
        if neg_loss.numel() == 0:  # No negative pixels in this batch
            return pos_loss.mean()  # Return only positive class loss
        
        # Average the mean losses of both classes equally
        return (pos_loss.mean() + neg_loss.mean()) / 2

# ------------------ Segmentation Metrics ------------------
def calculate_iou(pred, target, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU) for Binary Segmentation.

    IoU measures the overlap between predicted and ground truth segmentation masks.
    It is calculated as the ratio of intersection area to union area.

    Formula: IoU = (Intersection + smooth) / (Union + smooth)

    Args:
        pred (numpy.ndarray): Predicted binary mask (0s and 1s)
        target (numpy.ndarray): Ground truth binary mask (0s and 1s)
        smooth (float, optional): Small value added to avoid division by zero.
                                 Defaults to 1e-6.

    Returns:
        float: IoU score between 0 and 1, where 1 indicates perfect overlap

    Notes:
        - Both pred and target are converted to boolean arrays
        - The smooth parameter prevents division by zero when both masks are empty
        - Higher IoU values indicate better segmentation performance
    """
    # Convert arrays to boolean type for logical operations
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    # Calculate intersection: pixels where both pred and target are True
    intersection = np.logical_and(pred, target).sum()
    # Calculate union: pixels where either pred or target (or both) are True
    union = np.logical_or(pred, target).sum()
    
    # Compute IoU with smoothing to handle edge cases
    iou = (intersection + smooth) / (union + smooth)
    return iou

def calculate_dice_coefficient(pred, target, smooth=1e-6):
    """
    Calculate Dice Coefficient (F1 Score) for Binary Segmentation.

    The Dice coefficient measures the similarity between two sets and is commonly
    used in segmentation tasks. It gives equal weight to precision and recall.

    Formula: Dice = (2 * Intersection + smooth) / (Pred_Sum + Target_Sum + smooth)

    Args:
        pred (numpy.ndarray): Predicted binary mask (0s and 1s)
        target (numpy.ndarray): Ground truth binary mask (0s and 1s)
        smooth (float, optional): Small value added to avoid division by zero.
                                 Defaults to 1e-6.

    Returns:
        float: Dice coefficient between 0 and 1, where 1 indicates perfect overlap

    Notes:
        - Also known as F1-score for binary segmentation
        - More sensitive to small objects than IoU
        - Commonly used in medical image segmentation evaluation
    """
    # Convert arrays to boolean type for consistent processing
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    # Calculate intersection: pixels where both masks are True
    intersection = np.logical_and(pred, target).sum()
    
    # Compute Dice coefficient with smoothing
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice

def calculate_pixel_accuracy(pred, target):
    """
    Calculate Pixel-wise Accuracy for Binary Segmentation.

    Pixel accuracy measures the percentage of correctly classified pixels
    in the segmentation mask. It is a simple but potentially misleading metric
    for imbalanced datasets.

    Formula: Accuracy = (Correct_Pixels) / (Total_Pixels)

    Args:
        pred (numpy.ndarray): Predicted binary mask (0s and 1s)
        target (numpy.ndarray): Ground truth binary mask (0s and 1s)

    Returns:
        float: Pixel accuracy between 0 and 1

    Notes:
        - Simple to compute but can be misleading with class imbalance
        - For wildfire segmentation, this might be high due to many unburned pixels
        - Should be used alongside other metrics like IoU and Dice
    """
    # Convert arrays to boolean for element-wise comparison
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    # Count correctly classified pixels (where pred == target)
    correct_pixels = np.sum(pred == target)
    # Total number of pixels in the image
    total_pixels = pred.size
    
    # Calculate accuracy as ratio of correct to total pixels
    accuracy = correct_pixels / total_pixels
    return accuracy

def compute_accuracy(outputs, masks, threshold=0.5):
    """
    Compute Pixel-wise Accuracy for Segmentation Model Outputs.

    This function calculates accuracy directly from raw model logits by:
    1. Applying sigmoid to convert logits to probabilities
    2. Thresholding probabilities to get binary predictions
    3. Comparing predictions with ground truth masks

    Args:
        outputs (torch.Tensor): Raw model outputs/logits [B, C, H, W]
        masks (torch.Tensor): Ground truth binary masks [B, C, H, W]
        threshold (float, optional): Probability threshold for binarization.
                                   Defaults to 0.5.

    Returns:
        torch.Tensor: Pixel accuracy as a scalar tensor

    Notes:
        - Uses torch operations for GPU compatibility
        - Automatically flattens tensors for pixel-wise comparison
        - Returns tensor for gradient flow in training loops
    """
    with torch.no_grad():  # Disable gradient computation for evaluation
        # Convert logits to probabilities using sigmoid
        preds = torch.sigmoid(outputs) >= threshold  # Binary predictions
        # Compare predictions with ground truth (boolean tensors)
        correct = (preds == masks.bool()).float().sum()  # Count correct pixels
        total = masks.numel()  # Total number of elements
        return correct / total  # Return accuracy ratio

def calculate_mae(pred, target):
    """
    Calculate Mean Absolute Error (MAE) for Segmentation.

    MAE measures the average absolute difference between predicted and target values.
    For segmentation, this can be used with probability maps or binary masks.

    Formula: MAE = mean(|pred - target|)

    Args:
        pred (numpy.ndarray): Predicted values (probabilities or binary)
        target (numpy.ndarray): Ground truth values (binary masks)

    Returns:
        float: Mean absolute error value

    Notes:
        - If pred is boolean, it's converted to float (0.0 or 1.0)
        - Target is always converted to float for consistency
        - Lower MAE indicates better performance
        - Useful for evaluating probability map quality
    """
    # Ensure predictions are in float format
    if pred.dtype == bool:
        pred = pred.astype(float)  # Convert binary to 0.0/1.0
    else:
        pred = pred.astype(float)  # Ensure float type
    
    # Convert target to float for computation
    target = target.astype(float)
    
    # Calculate mean absolute difference
    mae = np.mean(np.abs(pred - target))
    return mae

def evaluate_segmentation_metrics(model, dataloader, device):
    """
    Evaluate Segmentation Model Performance Across Multiple Metrics.

    This function comprehensively evaluates a trained segmentation model by computing
    multiple metrics on a validation/test dataset. It processes the entire dataloader
    and aggregates results across all samples.

    Computed Metrics:
    - IoU (Intersection over Union): Measures overlap quality
    - Dice Coefficient: F1-score for segmentation, sensitive to small objects
    - Pixel Accuracy: Simple accuracy, can be misleading with imbalance
    - MAE (Mean Absolute Error): Average difference between predictions and targets

    Args:
        model (nn.Module): Trained PyTorch segmentation model
        dataloader (DataLoader): DataLoader containing evaluation images and masks
        device (torch.device): Device to run inference on (CPU/GPU)

    Returns:
        tuple: (metrics_dict, all_preds_flat, all_masks_flat)
            - metrics_dict: Dictionary with mean and std for each metric
            - all_preds_flat: Concatenated flattened predictions for classification report
            - all_masks_flat: Concatenated flattened ground truth for classification report

    Notes:
        - Model is set to eval mode during evaluation
        - Uses sigmoid activation and 0.5 threshold for binary classification
        - Computes per-sample metrics then aggregates across dataset
        - Returns flattened arrays for generating sklearn classification reports
    """
    # Set model to evaluation mode (disables dropout, batch norm updates, etc.)
    model.eval()
    
    # Initialize lists to store per-sample metrics
    ious = []
    dice_scores = []
    pixel_accuracies = []
    maes = []
    
    # Lists to store flattened predictions and masks for classification metrics
    all_preds_flat = []
    all_masks_flat = []
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        # Iterate through all batches in the dataloader
        for images, masks in tqdm(dataloader, desc="Evaluating segmentation metrics"):
            # Move batch to specified device (GPU/CPU)
            images = images.to(device)
            masks = masks.to(device).float()
            
            # Forward pass: get model predictions
            outputs = model(images)
            # Convert logits to probabilities
            pred_probs = torch.sigmoid(outputs)
            # Convert to binary predictions using 0.5 threshold
            pred_binary = (pred_probs > 0.5).float()
            
            # Convert tensors to numpy arrays for metric computation
            pred_binary_np = pred_binary.cpu().numpy()
            pred_probs_np = pred_probs.cpu().numpy()
            masks_np = masks.cpu().numpy()
            
            # Flatten predictions and masks for classification report generation
            all_preds_flat.append(pred_binary_np.flatten())
            all_masks_flat.append(masks_np.flatten())
            
            # Calculate metrics for each sample in the current batch
            for i in range(pred_binary_np.shape[0]):
                # Extract single sample (remove batch dimension)
                pred_bin = pred_binary_np[i, 0]  # Shape: [H, W]
                pred_prob = pred_probs_np[i, 0]  # Shape: [H, W]
                mask = masks_np[i, 0]            # Shape: [H, W]
                
                # Compute individual metrics for this sample
                iou = calculate_iou(pred_bin, mask)
                dice = calculate_dice_coefficient(pred_bin, mask)
                pixel_acc = calculate_pixel_accuracy(pred_bin, mask)
                mae = calculate_mae(pred_prob, mask)
                
                # Store metrics for later aggregation
                ious.append(iou)
                dice_scores.append(dice)
                pixel_accuracies.append(pixel_acc)
                maes.append(mae)
    
    # Concatenate all flattened predictions and masks across batches
    all_preds_flat = np.concatenate(all_preds_flat)
    all_masks_flat = np.concatenate(all_masks_flat).astype(int)
    
    # Calculate aggregate metrics: mean and standard deviation
    metrics = {
        'IoU': np.mean(ious),
        'Dice_Coefficient': np.mean(dice_scores),
        'Pixel_Accuracy': np.mean(pixel_accuracies),
        'MAE': np.mean(maes),
        'IoU_std': np.std(ious),
        'Dice_std': np.std(dice_scores),
        'Pixel_Accuracy_std': np.std(pixel_accuracies),
        'MAE_std': np.std(maes)
    }
    
    return metrics, all_preds_flat, all_masks_flat

def print_segmentation_metrics(metrics, mode="pixels"):
    """
    Print Segmentation Metrics in a Formatted, Human-Readable Format.

    This function displays segmentation evaluation results in a clear, organized manner.
    It shows both individual metrics with their standard deviations and a summary table.

    Args:
        metrics (dict): Dictionary containing metric names as keys and values as values.
                       Expected keys: IoU, Dice_Coefficient, Pixel_Accuracy, MAE
                       and their corresponding _std versions.
        mode (str, optional): Evaluation context description (e.g., "pixels", "tiles").
                             Defaults to "pixels".

    Returns:
        None: Prints formatted output to console

    Notes:
        - Displays metrics with 4 decimal precision
        - Shows mean ± standard deviation format
        - Creates a pandas DataFrame for tabular display
        - Useful for both console output and logging
    """
    # Print header with evaluation mode
    print(f"\n=== Segmentation Metrics ({mode}) ===")
    
    # Print each metric with mean and standard deviation
    print(f"IoU (Intersection over Union): {metrics['IoU']:.4f} ± {metrics['IoU_std']:.4f}")
    print(f"Dice Coefficient: {metrics['Dice_Coefficient']:.4f} ± {metrics['Dice_std']:.4f}")
    print(f"Pixel Accuracy: {metrics['Pixel_Accuracy']:.4f} ± {metrics['Pixel_Accuracy_std']:.4f}")
    print(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f} ± {metrics['MAE_std']:.4f}")
    
    # Create a summary DataFrame for tabular display
    summary_data = {
        'Metric': ['IoU', 'Dice Coefficient', 'Pixel Accuracy', 'MAE'],
        'Mean': [metrics['IoU'], metrics['Dice_Coefficient'], 
                metrics['Pixel_Accuracy'], metrics['MAE']],
        'Std': [metrics['IoU_std'], metrics['Dice_std'], 
               metrics['Pixel_Accuracy_std'], metrics['MAE_std']]
    }
    summary_df = pd.DataFrame(summary_data)
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))

def save_classification_report(y_true, y_pred, save_dir='Model_Evaluation', mode='validation'):
    """
    Generate, Display, and Save Classification Report and Confusion Matrix.

    This function creates a comprehensive classification evaluation including:
    1. Text-based classification report with precision, recall, F1-score
    2. Confusion matrix visualization
    3. Automatic saving of both to timestamped files

    Args:
        y_true (array-like): Ground truth labels (flattened predictions/masks)
        y_pred (array-like): Predicted labels (flattened predictions)
        save_dir (str, optional): Directory to save outputs. Defaults to 'Model_Evaluation'.
        mode (str, optional): Evaluation context (e.g., 'validation', 'test').
                             Defaults to 'validation'.

    Returns:
        None: Saves files and displays plots

    Notes:
        - Creates directory if it doesn't exist
        - Saves classification report as CSV file
        - Saves confusion matrix as high-resolution PNG
        - Uses timestamp in filenames to avoid overwrites
        - Class names are hardcoded for wildfire segmentation (Unburned/Burned)
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # --- Generate and Save Classification Report ---
    report = classification_report(y_true, y_pred, target_names=['Unburned (0)', 'Burned (1)'])
    print(f"\n=== Classification Report ({mode}) ===")
    print(report)
    
    # Save report as CSV file
    report_path = os.path.join(save_dir, f'classification_report_{mode}_{ts}.csv')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✅ Classification report saved to {report_path}")
    
    # --- Generate and Save Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    
    # Create confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Unburned', 'Burned'], 
                yticklabels=['Unburned', 'Burned'])
    plt.title(f'Confusion Matrix ({mode})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Save confusion matrix plot
    cm_path = os.path.join(save_dir, f'confusion_matrix_{mode}_{ts}.png')
    plt.savefig(cm_path, dpi=300)
    print(f"✅ Confusion matrix plot saved to {cm_path}")
    plt.show()

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_dir='Model_Evaluation'):
    """
    Plot and Save Training History Curves.

    This function creates a comprehensive visualization of the training process by plotting
    both loss and accuracy curves for training and validation sets side by side.

    Args:
        train_losses (list): Training loss values for each epoch
        val_losses (list): Validation loss values for each epoch
        train_accuracies (list): Training accuracy values for each epoch
        val_accuracies (list): Validation accuracy values for each epoch
        save_dir (str, optional): Directory to save the plot. Defaults to 'Model_Evaluation'.

    Returns:
        None: Saves plot to file and displays it

    Notes:
        - Creates a 2-subplot figure (loss and accuracy)
        - Uses blue for training metrics, red for validation
        - Includes grid lines for better readability
        - Saves high-resolution PNG with timestamp
        - Automatically creates save directory if needed
    """
    # Generate epoch numbers based on training history length
    epochs = range(1, len(train_losses) + 1)
    
    # Create figure with two subplots side by side
    plt.figure(figsize=(14, 6))
    
    # Left subplot: Training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')      # Blue circles with lines
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')      # Red circles with lines
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Right subplot: Training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')  # Blue circles with lines
    plt.plot(epochs, val_accuracies, 'ro-', label='Validation Accuracy')  # Red circles with lines
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for unique filename
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(save_dir, f'training_metrics_{ts}.png')
    
    # Save plot with high resolution
    plt.savefig(save_path, dpi=300)
    print(f"✅ Training metrics plot saved to {save_path}")
    plt.show()

# ------------------ Dataset ------------------
# Custom dataset for segmentation tasks (images + masks)
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, label_paths, windows, target_size=(256, 256), bands=None):
        # Store input parameters
        self.image_paths = image_paths       # List of raster image file paths
        self.label_paths = label_paths       # List of vector label file paths (shapefiles/geojson)
        self.windows = windows               # List of (image_index, raster_window) pairs
        self.target_size = target_size       # Desired output size for tiles (e.g., 256x256)
        self.bands = bands                   # Selected bands per image (e.g., RGB=[1,2,3])
        self.num_bands = len(bands[0]) if bands else None  # Number of channels to extract
        
        # Store CRS (coordinate reference system) and geodataframes
        self.crs_list = []   # To keep raster CRS for each image
        self.gdf_list = []   # To keep vector labels aligned to raster CRS
        
        # Loop through images + labels together
        for image_path, label_path in zip(image_paths, label_paths):
            # Open raster to extract CRS
            with rasterio.open(image_path) as src:
                self.crs_list.append(src.crs)
            
            # Load vector labels
            gdf = gpd.read_file(label_path)
            
            # Reproject vector labels to raster CRS if needed
            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)

            # Clean invalid geometries using buffer(0) trick
            gdf.geometry = gdf.geometry.buffer(0)
            
            # Store aligned geodataframe
            self.gdf_list.append(gdf)

    # Return dataset length = number of windows (tiles)
    def __len__(self):
        return len(self.windows)

    # Retrieve an item by index
    def __getitem__(self, idx, _retries=0):
        if _retries > 10:
            raise RuntimeError(f"Exceeded maximum retries ({_retries}) attempting to read a valid tile. Dataset might be severely corrupted.")

        img_idx, window = self.windows[idx]
        image_path = self.image_paths[img_idx]

        try:
            gdf = self.gdf_list[img_idx]
            crs = self.crs_list[img_idx]

            # Open raster and read tile
            with rasterio.open(image_path) as src:
                # Read selected bands (subset of channels) inside given window
                image = src.read(self.bands[img_idx], window=window)
                
                # Transform for this tile window
                tile_transform = src.window_transform(window)
                
                # Dimensions of the tile
                height = window.height
                width = window.width
                
                # Geographic bounding box of the tile
                bounds = rasterio.windows.bounds(window, src.transform)

            # Clip vector labels to the current tile bounding box
            clipped_gdf = gdf.clip(bounds)
            
            # Convert polygons to rasterizable shapes (geometry + class=1)
            shapes = [(mapping(geom), 1) for geom in clipped_gdf.geometry if not geom.is_empty]
            
            # Rasterize clipped polygons into binary mask
            mask = rasterize(
                shapes,
                out_shape=(height, width),    # Match tile dimensions
                transform=tile_transform,     # Use tile’s transform
                fill=0,                       # Background = 0
                all_touched=True,             # Label all pixels touched by geometry
                dtype=np.uint8
            )

            # Resize image to target size (H,W,C), reflect padding + anti-alias
            image = resize(
                image.transpose(1, 2, 0),     # Change from (C,H,W) -> (H,W,C)
                self.target_size + (self.num_bands,),
                mode='reflect',
                anti_aliasing=True
            )
            
            # Back to PyTorch format (C,H,W) and convert to float32
            image = image.transpose(2, 0, 1).astype(np.float32)

            # Normalize each channel (min-max scaling to [0,1])
            for c in range(image.shape[0]):
                channel = image[c]
                min_val, max_val = channel.min(), channel.max()
                if max_val - min_val > 1e-6:   # Avoid division by zero
                    image[c] = (channel - min_val) / (max_val - min_val)
                else:
                    image[c] = 0               # If flat channel, set to zero

            # Resize mask to target size (nearest neighbor, no anti-alias)
            mask = resize(mask, self.target_size, order=0, anti_aliasing=False).astype(np.uint8)
            
            # Add channel dimension -> (1,H,W)
            mask = mask[None, :, :]

            # Return PyTorch tensors (image, mask)
            return torch.from_numpy(image), torch.from_numpy(mask)
        except Exception as e:
            print(f"Warning: Error reading tile at index {idx} from {image_path}: {e}")
            # If a tile is corrupted, try to return a random valid tile to prevent the training from crashing
            new_idx = np.random.randint(0, len(self.windows))
            return self.__getitem__(new_idx, _retries + 1)

def extract_date_string(path):
    """Extracts an 8-digit date string (YYYYMMDD) from a file path."""
    filename = os.path.basename(path)
    match = re.search(r'(\d{8})', filename)
    return match.group(1) if match else None

# Function to match raster (images) with shapefile (labels) by spatial overlap
def match_raster_shapefile(image_base_dir, label_base_dir):
    # Recursively find all .tif image files
    image_files = glob.glob(os.path.join(image_base_dir, "**", "*.tif"), recursive=True)
    # Recursively find all .shp shapefile label files
    label_files = glob.glob(os.path.join(label_base_dir, "**", "*.shp"), recursive=True)

    pairs = []  # To store matched (image, label) pairs
    
    # Loop through all images
    for img_path in image_files:
        img_date = extract_date_string(img_path)
        try:
            # Get raster bounds and CRS
            with rasterio.open(img_path) as src:
                img_bounds = src.bounds  # (left, bottom, right, top)
                img_crs = src.crs
                img_polygon = box(*img_bounds)  # Create polygon from bounds
            
            # Find shapefiles that spatially overlap with this raster
            overlapping_labels = []
            
            for label_path in label_files:
                try:
                    # Match by date if both filenames contain a date string
                    label_date = extract_date_string(label_path)
                    if img_date and label_date and img_date != label_date:
                        continue

                    # Load shapefile
                    gdf = gpd.read_file(label_path)
                    
                    # Skip if empty
                    if gdf.empty:
                        continue
                    
                    # Reproject to raster CRS if needed
                    if gdf.crs != img_crs:
                        gdf = gdf.to_crs(img_crs)

                    # Clean invalid geometries using buffer(0) trick
                    gdf.geometry = gdf.geometry.buffer(0)
                    
                    # Check for spatial overlap
                    # Check if any polygon intersects with raster bounds
                    if gdf.geometry.intersects(img_polygon).any():
                        overlapping_labels.append(label_path)
                        
                except Exception as e:
                    print(f"Warning: Could not process shapefile {label_path}: {e}")
                    continue
            
            # If we found overlapping shapefiles, add the first one
            if overlapping_labels:
                pairs.append((img_path, overlapping_labels[0]))
                print(f"Matched: {os.path.basename(img_path)} <-> {os.path.basename(overlapping_labels[0])}")
            else:
                print(f"No overlapping shapefile found for: {os.path.basename(img_path)}")
                
        except Exception as e:
            print(f"Warning: Could not process raster {img_path}: {e}")
            continue
    
    print(f"\nTotal matched pairs: {len(pairs)}")
    return pairs

# ------------------ Main ------------------
def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="pyogrio.raw")
    
    # Select device: GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Directories for raster imagery and wildfire polygons
    image_base_dir = r"Raster_Train"
    label_base_dir = r"Wildfire_Polygon_Train"

    # --- If using GPU ---
    if device.type == "cuda":
        num_gpus = torch.cuda.device_count()
        print(f" -> {num_gpus} CUDA device(s) available")
        for i in range(num_gpus):
            print(f"   [GPU {i}] {torch.cuda.get_device_name(i)}")
            print(f"       Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
            print(f"       Memory Cached:    {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")
            print(f"       Total Memory:     {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.2f} GB")
            print(f"       Compute Capability: {torch.cuda.get_device_capability(i)}")

    # --- If using CPU ---
    else:
        print(f" -> CPU: {platform.processor() or 'Unknown'}")
        print(f" -> CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
        print(f" -> RAM available: {psutil.virtual_memory().total / 1024 ** 3:.2f} GB")

    # Match rasters with shapefiles
    pairs = match_raster_shapefile(image_base_dir, label_base_dir)
    if not pairs:
        print("No data")
        return

    # --- Prepare data ---
    tile_size = 512  # Extract 512x512 patches (downsampled spatially to 256x256)
    all_burn_tiles, all_unburn_tiles = [], []
    all_image_paths, all_label_paths, all_bands = [], [], []
    max_channels = 0  # Track max number of bands among rasters

    # Iterate over matched (image, label) pairs
    for image_path, label_path in pairs:
        with rasterio.open(image_path) as src:
            height, width = src.shape        # Get raster size
            bands = list(range(1, src.count + 1))  # Band indices
            max_channels = max(max_channels, len(bands))
            img_transform, img_crs = src.transform, src.crs  # Geo info

        # Load vector labels
        gdf = gpd.read_file(label_path)
        if gdf.crs != img_crs:
            gdf = gdf.to_crs(img_crs)  # Reproject to match raster
        # Drop invalid/empty geometries
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
        if gdf.empty:
            continue

        # Generate windows (tiles) for raster
        windows = [Window(j, i, tile_size, tile_size)
                   for i in range(0, height, tile_size)
                   for j in range(0, width, tile_size)
                   if i + tile_size <= height and j + tile_size <= width]

        # Spatial index for faster polygon queries
        sindex = gdf.sindex
        burn_tiles_img, unburn_tiles_img = [], []

        # For each tile, check if it intersects with burn polygons
        for w in windows:
            left, bottom, right, top = rasterio.windows.bounds(w, img_transform)
            tile_poly = box(left, bottom, right, top)
            hits = list(sindex.query(tile_poly, predicate="intersects"))
            if hits:
                burn_tiles_img.append((len(all_image_paths), w))    # Positive tile
            else:
                unburn_tiles_img.append((len(all_image_paths), w))  # Negative tile

        # Collect tiles + paths
        all_burn_tiles.extend(burn_tiles_img)
        all_unburn_tiles.extend(unburn_tiles_img)
        all_image_paths.append(image_path)
        all_label_paths.append(label_path)
        all_bands.append(bands)

    # Use all data instead of downsampling classes at the tile level.
    # Spatial downsampling (pixel-level) is handled by resizing larger tiles (512x512) 
    # down to the model's input size (256x256) in the SegmentationDataset.
    print(f"\nProcessing all tiles with spatial downsampling...")
    if not all_burn_tiles:
        print("❌ Error: No burned tiles found. Check your shapefiles and spatial overlap logic.")
        return

    all_tiles = all_burn_tiles + all_unburn_tiles
    tile_labels = [1] * len(all_burn_tiles) + [0] * len(all_unburn_tiles)
    print(f" -> Total dataset size (no tiles discarded): {len(all_tiles)} tiles")

    # Split into train/validation sets (stratified)
    train_windows, val_windows, _, _ = train_test_split(
        all_tiles, tile_labels, test_size=0.2, random_state=42, stratify=tile_labels
    )

    # Ensure all rasters use same number of bands
    adjusted_bands = [b[:max_channels] for b in all_bands]

    # Build dataset objects
    train_dataset = SegmentationDataset(all_image_paths, all_label_paths, train_windows, bands=adjusted_bands)
    val_dataset = SegmentationDataset(all_image_paths, all_label_paths, val_windows, bands=adjusted_bands)

    # DataLoader for batching
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # --- Model, Loss, Optimizer ---
    model = UNet(n_channels=max_channels, n_classes=1).to(device)
    # Use Balanced BCE to handle class imbalance (pixel-level equal contribution)
    criterion = BalancedBCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # --- Training Loop ---
    num_epochs = 10
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        for images, masks in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
            images, masks = images.to(device), masks.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_accuracy += compute_accuracy(outputs, masks).item() * images.size(0)

        epoch_train_loss = running_loss / len(train_dataloader.dataset)
        epoch_train_acc = running_accuracy / len(train_dataloader.dataset)
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False):
                images, masks = images.to(device), masks.to(device).float()
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item() * images.size(0)
                val_accuracy += compute_accuracy(outputs, masks).item() * images.size(0)

        epoch_val_loss = val_loss / len(val_dataloader.dataset)
        epoch_val_acc = val_accuracy / len(val_dataloader.dataset)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}"
        )

    # --- Plotting ---
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # --- Evaluation ---
    print("\nEvaluating model with segmentation metrics...")
    
    # Evaluate with pixel-level metrics
    pixel_metrics, y_pred, y_true = evaluate_segmentation_metrics(model, val_dataloader, device)
    print_segmentation_metrics(pixel_metrics, mode="pixels")

    # Generate and save classification report and confusion matrix
    save_classification_report(y_true, y_pred, mode='validation')
    
    # Save metrics to file in the specific evaluation directory
    eval_save_dir = 'Model_Evaluation'
    os.makedirs(eval_save_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    metrics_df = pd.DataFrame([pixel_metrics])
    metrics_file = os.path.join(eval_save_dir, f"segmentation_metrics_{ts}.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f"\nMetrics saved to: {metrics_file}")

    # --- Save trained model ---
    export_dir = "Export_Model"
    os.makedirs(export_dir, exist_ok=True)
    model_path = os.path.join(export_dir, "unet_wildfire.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved successfully to {model_path}")


# Run main if script is executed directly
if __name__ == "__main__":
    main()
