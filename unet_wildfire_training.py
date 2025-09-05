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
from sklearn.metrics import classification_report, confusion_matrix  # Classification evaluation
from sklearn.model_selection import train_test_split                  # Train-test data splitting


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

# Function to match raster (images) with shapefile (labels)
def match_raster_shapefile(image_base_dir, label_base_dir):
    # Recursively find all .tif image files
    image_files = glob.glob(os.path.join(image_base_dir, "**", "*.tif"), recursive=True)
    # Recursively find all .shp shapefile label files
    label_files = glob.glob(os.path.join(label_base_dir, "**", "*.shp"), recursive=True)

    # Helper function: extract file base name (no extension)
    def get_base_name(path):
        return os.path.splitext(os.path.basename(path))[0]

    pairs = []  # To store matched (image, label) pairs
    
    # Loop through all images
    for img in image_files:
        base = get_base_name(img)  # Base name of image
        # Look for shapefile labels that share the same prefix
        matches = [lbl for lbl in label_files if get_base_name(lbl).startswith(base)]
        # If found, keep first match
        if matches:
            pairs.append((img, matches[0]))
    
    # Return list of matched image-label pairs
    return pairs

# Function to get predictions and labels from a trained model
def get_preds_labels(model, dataloader, device, mode="pixels"):
    model.eval()  # Set model to evaluation mode
    all_preds, all_labels = [], []  # Store predictions + ground-truth labels

    # Disable gradient computation (faster inference)
    with torch.no_grad():
        # Iterate through dataset batches
        for images, masks in tqdm(dataloader, desc="Collecting predictions"):
            images = images.to(device)       # Move images to GPU/CPU
            masks = masks.to(device).float() # Move masks to device
            
            # Forward pass → get logits
            outputs = model(images)
            
            # Apply sigmoid activation, then threshold at 0.5 → binary prediction
            preds = torch.sigmoid(outputs) > 0.5

            # Mode 1: pixel-wise evaluation
            if mode == "pixels":
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(masks.cpu().numpy().flatten())

            # Mode 2: tile-wise evaluation (treat whole tile as one prediction)
            elif mode == "tiles":
                for p, m in zip(preds, masks):
                    # If mean predicted probability > 0.1, classify tile as positive (Burn)
                    tile_pred = (p.cpu().numpy().mean() > 0.1).astype(int)
                    # Same rule for ground truth
                    tile_label = (m.cpu().numpy().mean() > 0.1).astype(int)
                    all_preds.append(tile_pred)
                    all_labels.append(tile_label)
    
    # Return numpy arrays of predictions + labels
    return np.array(all_preds), np.array(all_labels)

# Function to compute classification report + confusion matrix
def compute_report_and_cm(all_labels, all_preds, sample_pixels=False, mode="pixels"):
    # Optionally balance dataset by sampling equal number of burn/unburn pixels
    if sample_pixels:
        burn_idx = np.where(all_labels == 1)[0]   # Indices of positive samples
        unburn_idx = np.where(all_labels == 0)[0] # Indices of negative samples
        n = min(len(burn_idx), len(unburn_idx))   # Balance to smaller group size
        
        # Randomly sample balanced set
        rng = np.random.RandomState(42)
        sel = np.concatenate([
            rng.choice(burn_idx, n, replace=False),
            rng.choice(unburn_idx, n, replace=False)
        ])
        all_labels = all_labels[sel]
        all_preds = all_preds[sel]

    # Generate sklearn classification report (precision, recall, f1)
    report = classification_report(
        all_labels, all_preds,
        target_names=['Unburn', 'Burn'],
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    print(report_df)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix as heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Unburn', 'Burn'],
                yticklabels=['Unburn', 'Burn'])
    plt.title(f'Confusion Matrix ({mode}, {"balanced" if sample_pixels else "raw"})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Save confusion matrix with timestamp
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"confusion_matrix_{mode}_{'balanced' if sample_pixels else 'raw'}_{ts}.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"Saved confusion matrix → {fname}")
    plt.show()

    # Return report dataframe + confusion matrix
    return report_df, cm


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
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images, masks = images.to(device), masks.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_dataloader):.4f}")

    # --- Evaluation ---
    all_preds, all_labels = get_preds_labels(model, val_dataloader, device, mode="pixels")
    compute_report_and_cm(all_labels, all_preds, sample_pixels=True)

    # Optional: tile-level confusion matrix
    all_preds, all_labels = get_preds_labels(model, val_dataloader, device, mode="tiles")
    compute_report_and_cm(all_labels, all_preds, sample_pixels=False)

    # --- Save trained model ---
    export_dir = "Export_Model"
    os.makedirs(export_dir, exist_ok=True)
    model_path = os.path.join(export_dir, "unet_wildfire.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved successfully to {model_path}")


# Run main if script is executed directly
if __name__ == "__main__":
    main()

