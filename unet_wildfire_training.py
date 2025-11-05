# Core Python utilities
import os        # For file and directory operations
import glob      # For pattern-based file searching

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
# Import torch's neural network module base class
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        # Initialize the parent nn.Module class
        super().__init__()
        
        # Define a block with 2 consecutive convolutional layers
        self.double_conv = nn.Sequential(
            # First 2D convolution
            # - Input channels: in_channels
            # - Output channels: out_channels
            # - Kernel size: 3x3
            # - Padding=1 keeps spatial dimensions the same
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            
            # Normalize activations across batch to stabilize training
            nn.BatchNorm2d(out_channels),
            
            # ReLU activation (applied in place to save memory)
            nn.ReLU(inplace=True),
            
            # Second 2D convolution with same out_channels
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            
            # Another batch normalization
            nn.BatchNorm2d(out_channels),
            
            # ReLU activation again
            nn.ReLU(inplace=True)
        )

    # Define the forward pass
    def forward(self, x):
        # Apply the sequential double convolution to input tensor x
        return self.double_conv(x)

# Block for downsampling: MaxPool + DoubleConv
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Downsampling step
        self.maxpool_conv = nn.Sequential(
            # Reduce spatial resolution by factor of 2
            nn.MaxPool2d(2),
            
            # Apply double convolution after pooling
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        # Apply max pooling then double convolution
        return self.maxpool_conv(x)

# Block for upsampling: ConvTranspose + concatenation + DoubleConv
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Transposed convolution (also called deconvolution) to upsample
        # Reduces channel depth by half (in_channels -> in_channels // 2)
        # Increases spatial size by stride=2
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                     kernel_size=2, stride=2)
        
        # Apply double convolution after concatenation
        # Input channels: in_channels (because x1+ x2 concatenated)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 = decoder input (upsampled features)
        # x2 = encoder skip connection features
        
        # Perform upsampling with transposed convolution
        x1 = self.up(x1)
        
        # Compute differences in spatial dimensions (height/width)
        diffY = x2.size()[2] - x1.size()[2]  # Difference in height
        diffX = x2.size()[3] - x1.size()[3]  # Difference in width
        
        # Pad x1 so it matches the size of x2 (important when input dims are odd)
        x1 = nn.functional.pad(x1, [
            diffX // 2, diffX - diffX // 2,   # Left and right padding
            diffY // 2, diffY - diffY // 2    # Top and bottom padding
        ])
        
        # Concatenate along channel dimension (dim=1 for NCHW format)
        x = torch.cat([x2, x1], dim=1)
        
        # Apply double convolution to merged features
        return self.conv(x)

# Final convolution to map features to desired output classes
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 1x1 convolution to reduce channels -> num_classes
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Apply final convolution (no activation here, usually softmax/sigmoid later)
        return self.conv(x)

# Define the full U-Net model
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        # Initialize parent nn.Module class
        super(UNet, self).__init__()

        # Initial convolution block (input -> 64 channels)
        # Takes in n_channels (e.g., 3 for RGB images, or 1 for grayscale)
        self.inc = DoubleConv(n_channels, 64)

        # Encoder (downsampling path)
        # Each Down block halves spatial dimensions and increases feature depth
        self.down1 = Down(64, 128)     # 64 -> 128 channels
        self.down2 = Down(128, 256)    # 128 -> 256 channels
        self.down3 = Down(256, 512)    # 256 -> 512 channels
        self.down4 = Down(512, 1024)   # 512 -> 1024 channels (bottleneck)

        # Decoder (upsampling path with skip connections)
        # Each Up block doubles spatial resolution and concatenates encoder features
        self.up1 = Up(1024, 512)       # 1024 -> 512 channels
        self.up2 = Up(512, 256)        # 512 -> 256 channels
        self.up3 = Up(256, 128)        # 256 -> 128 channels
        self.up4 = Up(128, 64)         # 128 -> 64 channels

        # Final output convolution (maps 64 -> n_classes)
        # n_classes = 1 for binary segmentation, or >1 for multi-class
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder forward pass
        x1 = self.inc(x)     # First conv block (input -> 64)
        x2 = self.down1(x1)  # Downsample (64 -> 128)
        x3 = self.down2(x2)  # Downsample (128 -> 256)
        x4 = self.down3(x3)  # Downsample (256 -> 512)
        x5 = self.down4(x4)  # Bottleneck (512 -> 1024)

        # Decoder forward pass (upsample + concatenate with encoder features)
        x = self.up1(x5, x4) # Up 1024 -> 512, concat with x4
        x = self.up2(x, x3)  # Up 512 -> 256, concat with x3
        x = self.up3(x, x2)  # Up 256 -> 128, concat with x2
        x = self.up4(x, x1)  # Up 128 -> 64, concat with x1

        # Final output layer (per-pixel class logits)
        logits = self.outc(x)

        # Return predicted segmentation map
        return logits

# ------------------ Segmentation Metrics ------------------
def calculate_iou(pred, target, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU) for binary segmentation.
    
    Args:
        pred: Predicted binary mask (numpy array)
        target: Ground truth binary mask (numpy array)
        smooth: Small value to avoid division by zero
    
    Returns:
        IoU score (float)
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    
    iou = (intersection + smooth) / (union + smooth)
    return iou

def calculate_dice_coefficient(pred, target, smooth=1e-6):
    """
    Calculate Dice coefficient for binary segmentation.
    
    Args:
        pred: Predicted binary mask (numpy array)
        target: Ground truth binary mask (numpy array)
        smooth: Small value to avoid division by zero
    
    Returns:
        Dice coefficient (float)
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    intersection = np.logical_and(pred, target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice

def calculate_pixel_accuracy(pred, target):
    """
    Calculate pixel accuracy for binary segmentation.
    
    Args:
        pred: Predicted binary mask (numpy array)
        target: Ground truth binary mask (numpy array)
    
    Returns:
        Pixel accuracy (float)
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    correct_pixels = np.sum(pred == target)
    total_pixels = pred.size
    
    accuracy = correct_pixels / total_pixels
    return accuracy

def compute_accuracy(outputs, masks, threshold=0.5):
    """
    Compute pixel-wise accuracy for segmentation.
    """
    with torch.no_grad():
        preds = torch.sigmoid(outputs) >= threshold
        correct = (preds == masks.bool()).float().sum()
        total = masks.numel()
        return correct / total

def calculate_mae(pred, target):
    """
    Calculate Mean Absolute Error (MAE) for segmentation.
    
    Args:
        pred: Predicted mask (numpy array, can be probabilities or binary)
        target: Ground truth binary mask (numpy array)
    
    Returns:
        MAE (float)
    """
    # Convert predictions to probabilities if they're binary
    if pred.dtype == bool:
        pred = pred.astype(float)
    else:
        pred = pred.astype(float)
    
    target = target.astype(float)
    
    mae = np.mean(np.abs(pred - target))
    return mae

def evaluate_segmentation_metrics(model, dataloader, device):
    """
    Evaluate segmentation model using IoU, Dice, Pixel Accuracy, and MAE.
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run inference on
    
    Returns:
        Dictionary containing all metrics
    """
    model.eval()
    
    ious = []
    dice_scores = []
    pixel_accuracies = []
    maes = []
    
    all_preds_flat = []
    all_masks_flat = []
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating segmentation metrics"):
            images = images.to(device)
            masks = masks.to(device).float()
            
            # Get predictions
            outputs = model(images)
            pred_probs = torch.sigmoid(outputs)
            pred_binary = (pred_probs > 0.5).float()
            
            # Convert to numpy for metric calculation
            pred_binary_np = pred_binary.cpu().numpy()
            pred_probs_np = pred_probs.cpu().numpy()
            masks_np = masks.cpu().numpy()
            
            # Flatten predictions and masks for classification report and confusion matrix
            all_preds_flat.append(pred_binary_np.flatten())
            all_masks_flat.append(masks_np.flatten())
            
            # Calculate metrics for each sample in the batch
            for i in range(pred_binary_np.shape[0]):
                pred_bin = pred_binary_np[i, 0]  # Remove channel dimension
                pred_prob = pred_probs_np[i, 0]  # Remove channel dimension
                mask = masks_np[i, 0]  # Remove channel dimension
                
                # Calculate metrics
                iou = calculate_iou(pred_bin, mask)
                dice = calculate_dice_coefficient(pred_bin, mask)
                pixel_acc = calculate_pixel_accuracy(pred_bin, mask)
                mae = calculate_mae(pred_prob, mask)
                
                ious.append(iou)
                dice_scores.append(dice)
                pixel_accuracies.append(pixel_acc)
                maes.append(mae)
    
    # Concatenate all flattened predictions and masks
    all_preds_flat = np.concatenate(all_preds_flat)
    all_masks_flat = np.concatenate(all_masks_flat).astype(int)

    # Calculate mean metrics
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
    Print segmentation metrics in a formatted way.
    
    Args:
        metrics: Dictionary containing segmentation metrics
        mode: Evaluation mode ("pixels" or "tiles")
    """
    print(f"\n=== Segmentation Metrics ({mode}) ===")
    print(f"IoU (Intersection over Union): {metrics['IoU']:.4f} ± {metrics['IoU_std']:.4f}")
    print(f"Dice Coefficient: {metrics['Dice_Coefficient']:.4f} ± {metrics['Dice_std']:.4f}")
    print(f"Pixel Accuracy: {metrics['Pixel_Accuracy']:.4f} ± {metrics['Pixel_Accuracy_std']:.4f}")
    print(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f} ± {metrics['MAE_std']:.4f}")
    
    # Create a summary DataFrame
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
    Generates, prints, and saves a classification report and confusion matrix.
    """
    # --- Classification Report ---
    report = classification_report(y_true, y_pred, target_names=['Unburned (0)', 'Burned (1)'])
    print(f"\n=== Classification Report ({mode}) ===")
    print(report)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = os.path.join(save_dir, f'classification_report_{mode}_{ts}.csv')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✅ Classification report saved to {report_path}")

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Unburned', 'Burned'], 
                yticklabels=['Unburned', 'Burned'])
    plt.title(f'Confusion Matrix ({mode})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    cm_path = os.path.join(save_dir, f'confusion_matrix_{mode}_{ts}.png')
    plt.savefig(cm_path, dpi=300)
    print(f"✅ Confusion matrix plot saved to {cm_path}")
    plt.show()

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_dir='Model_Evaluation'):
    """
    Plots and saves the training/validation loss and accuracy curves.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(save_dir, f'training_metrics_{ts}.png')
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
    def __getitem__(self, idx):
        # Extract image index + raster window for this sample
        img_idx, window = self.windows[idx]
        
        # Get paths and CRS for this image
        image_path = self.image_paths[img_idx]
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

# Function to match raster (images) with shapefile (labels) by spatial overlap
def match_raster_shapefile(image_base_dir, label_base_dir):
    # Recursively find all .tif image files
    image_files = glob.glob(os.path.join(image_base_dir, "**", "*.tif"), recursive=True)
    # Recursively find all .shp shapefile label files
    label_files = glob.glob(os.path.join(label_base_dir, "**", "*.shp"), recursive=True)

    pairs = []  # To store matched (image, label) pairs
    
    # Loop through all images
    for img_path in image_files:
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
    tile_size = 256  # Extract 256x256 patches
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

    # --- Balance dataset ---
    rng = np.random.RandomState(42)
    n_keep = min(len(all_burn_tiles), len(all_unburn_tiles))
    burn_indices = rng.choice(len(all_burn_tiles), size=n_keep, replace=False)
    unburn_indices = rng.choice(len(all_unburn_tiles), size=n_keep, replace=False)

    burn_tiles = [all_burn_tiles[i] for i in burn_indices]
    unburn_tiles = [all_unburn_tiles[i] for i in unburn_indices]

    balanced_tiles = burn_tiles + unburn_tiles
    tile_labels = [1] * len(burn_tiles) + [0] * len(unburn_tiles)

    # Split into train/validation sets (stratified)
    train_windows, val_windows, _, _ = train_test_split(
        balanced_tiles, tile_labels, test_size=0.2, random_state=42, stratify=tile_labels
    )

    # Ensure all rasters use same number of bands
    adjusted_bands = [b[:max_channels] for b in all_bands]

    # Build dataset objects
    train_dataset = SegmentationDataset(all_image_paths, all_label_paths, train_windows, bands=adjusted_bands)
    val_dataset = SegmentationDataset(all_image_paths, all_label_paths, val_windows, bands=adjusted_bands)

    # DataLoader for batching
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    # --- Loss weighting ---
    print("Computing pixel class weights...")
    burn_pixels, unburn_pixels = 0, 0
    for _, masks in train_dataloader:
        burn_pixels += masks.sum().item()                      # Count positive pixels
        unburn_pixels += masks.numel() - masks.sum().item()    # Count negative pixels
    pos_weight = torch.tensor([unburn_pixels / (burn_pixels + 1e-6)], device=device)

    # --- Model, Loss, Optimizer ---
    model = UNet(n_channels=max_channels, n_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Weighted binary cross-entropy
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # --- Training Loop ---
    num_epochs = 50
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
    
    # Save metrics to file
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    metrics_df = pd.DataFrame([pixel_metrics])
    metrics_file = f"segmentation_metrics_{ts}.csv"
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
