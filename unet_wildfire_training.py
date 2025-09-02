# Import necessary libraries for deep learning, geospatial processing, and visualization
import os  # Library for interacting with the operating system
import glob  # Library for working with file paths
import torch  # PyTorch library for tensor operations and neural networks
import torch.nn as nn  # PyTorch module for neural network layers and functions
import torch.optim as optim  # PyTorch module for optimization algorithms
import platform # Library for accessing system information
import psutil # Library for accessing system resources
from torch.utils.data import Dataset, DataLoader  # PyTorch utilities for dataset handling and batching
import rasterio  # Library for reading and writing geospatial raster data (e.g., GeoTIFF)
from rasterio.windows import Window  # Utility for defining windowed reads from raster data
from rasterio.features import rasterize  # Function to convert vector geometries to raster format
import geopandas as gpd  # Library for handling geospatial vector data (e.g., shapefiles)
from shapely.geometry import mapping, box  # Utilities for working with vector geometries
import numpy as np  # Library for numerical operations on arrays
from skimage.transform import resize  # Function for resizing images or arrays
from tqdm import tqdm  # Progress bar for iterating over loops
import pandas as pd  # Library for data manipulation and analysis
import seaborn as sns  # Visualization library for creating heatmaps and plots
import matplotlib.pyplot as plt  # Plotting library for visualizations
from sklearn.metrics import classification_report, confusion_matrix  # Metrics for model evaluation
from sklearn.model_selection import train_test_split  # Utility to split data into training and validation sets

# DoubleConv Module: Two consecutive convolutional layers with batch normalization and ReLU activation
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        # Initialize the parent class (nn.Module)
        super().__init__()
        # Define a sequential block with two Conv2d -> BatchNorm -> ReLU layers
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 3x3 convolution preserving input size
            nn.BatchNorm2d(out_channels),  # Normalize the output to stabilize training
            nn.ReLU(inplace=True),  # Apply ReLU activation in-place for memory efficiency
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # Second 3x3 convolution
            nn.BatchNorm2d(out_channels),  # Second normalization
            nn.ReLU(inplace=True)  # Second ReLU activation
        )

    def forward(self, x):
        # Forward pass: apply the double convolution block to the input tensor
        return self.double_conv(x)

# Down Module: Max pooling followed by a DoubleConv block for downsampling
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        # Initialize the parent class (nn.Module)
        super().__init__()
        # Define a sequential block with max pooling and DoubleConv
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # 2x2 max pooling to reduce spatial dimensions by half
            DoubleConv(in_channels, out_channels)  # Apply DoubleConv after pooling
        )

    def forward(self, x):
        # Forward pass: apply max pooling and DoubleConv
        return self.maxpool_conv(x)

# Up Module: Upsampling followed by concatenation with skip connection and DoubleConv
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        # Initialize the parent class (nn.Module)
        super().__init__()
        # Define transposed convolution for upsampling
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)  # Upsample by factor of 2
        # Define DoubleConv for processing concatenated feature maps
        self.conv = DoubleConv(in_channels, out_channels)  # Input channels include skip connection

    def forward(self, x1, x2):
        # Forward pass
        x1 = self.up(x1)  # Upsample the input tensor
        # Calculate padding to match x2's dimensions (skip connection)
        diffY = x2.size()[2] - x1.size()[2]  # Difference in height
        diffX = x2.size()[3] - x1.size()[3]  # Difference in width
        # Pad x1 to match x2's spatial dimensions
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # Concatenate skip connection (x2) with upsampled input (x1) along channel dimension
        x = torch.cat([x2, x1], dim=1)
        # Apply DoubleConv to concatenated tensor
        return self.conv(x)

# OutConv Module: Final 1x1 convolution to produce output channels
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        # Initialize the parent class (nn.Module)
        super().__init__()
        # Define 1x1 convolution to map to desired number of output channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Forward pass: apply 1x1 convolution
        return self.conv(x)

# UNet Model: Full U-Net architecture for image segmentation
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        # Initialize the parent class (nn.Module)
        super(UNet, self).__init__()
        # Store input and output channel counts
        self.n_channels = n_channels  # Number of input channels (e.g., image bands)
        self.n_classes = n_classes  # Number of output classes (e.g., 1 for binary segmentation)
        # Define encoder (contracting path)
        self.inc = DoubleConv(n_channels, 64)  # Initial DoubleConv block
        self.down1 = Down(64, 128)  # First downsampling block
        self.down2 = Down(128, 256)  # Second downsampling block
        self.down3 = Down(256, 512)  # Third downsampling block
        self.down4 = Down(512, 1024)  # Fourth downsampling block (bottleneck)
        # Define decoder (expansive path)
        self.up1 = Up(1024, 512)  # First upsampling block
        self.up2 = Up(512, 256)  # Second upsampling block
        self.up3 = Up(256, 128)  # Third upsampling block
        self.up4 = Up(128, 64)  # Fourth upsampling block
        # Define output layer
        self.outc = OutConv(64, n_classes)  # Final 1x1 convolution to produce segmentation map

    def forward(self, x):
        # Forward pass through encoder
        x1 = self.inc(x)  # Initial convolution
        x2 = self.down1(x1)  # First downsampling
        x3 = self.down2(x2)  # Second downsampling
        x4 = self.down3(x3)  # Third downsampling
        x5 = self.down4(x4)  # Bottleneck
        # Forward pass through decoder with skip connections
        x = self.up1(x5, x4)  # First upsampling with skip from x4
        x = self.up2(x, x3)  # Second upsampling with skip from x3
        x = self.up3(x, x2)  # Third upsampling with skip from x2
        x = self.up4(x, x1)  # Fourth upsampling with skip from x1
        # Produce final segmentation map
        logits = self.outc(x)
        return logits

# SegmentationDataset: Custom dataset for loading tiled GeoTIFF images and shapefile labels
class SegmentationDataset(Dataset):
    def __init__(self, image_path, label_path, windows, target_size=(256, 256), bands=None):
        # Initialize dataset parameters
        self.image_path = image_path  # Path to GeoTIFF image
        self.label_path = label_path  # Path to shapefile with labels
        self.windows = windows  # List of rasterio Window objects for tiling
        self.target_size = target_size  # Desired size for resized images/masks
        # Open GeoTIFF to get metadata
        with rasterio.open(image_path) as src:
            self.crs = src.crs  # Coordinate reference system of the image
            self.bands = bands if bands else list(range(1, src.count + 1))  # Select bands (default: all)
            self.num_bands = len(self.bands)  # Number of selected bands
        # Load shapefile and ensure CRS matches image
        self.gdf = gpd.read_file(label_path)  # Read shapefile with geopandas
        if self.gdf.crs != self.crs:
            self.gdf = self.gdf.to_crs(self.crs)  # Reproject shapefile to match image CRS

    def __len__(self):
        # Return the number of tiles (windows)
        return len(self.windows)

    def __getitem__(self, idx):
        # Get a single tile (image and mask) by index
        window = self.windows[idx]  # Select the window for this tile
        # Read image tile from GeoTIFF
        with rasterio.open(self.image_path) as src:
            image = src.read(self.bands, window=window)  # Read specified bands for the window
            tile_transform = src.window_transform(window)  # Get affine transform for the window
            height = window.height  # Height of the tile
            width = window.width  # Width of the tile
            bounds = rasterio.windows.bounds(window, src.transform)  # Get geographic bounds of the tile

        # Clip shapefile geometries to tile bounds
        clipped_gdf = self.gdf.clip(bounds)  # Clip shapefile to tile's geographic extent
        # Convert geometries to raster mask
        shapes = [(mapping(geom), 1) for geom in clipped_gdf.geometry if not geom.is_empty]  # Map valid geometries
        mask = rasterize(
            shapes,  # Geometries to rasterize
            out_shape=(height, width),  # Output shape of the mask
            transform=tile_transform,  # Affine transform for the tile
            fill=0,  # Unburn value (0 for non-labeled areas)
            all_touched=True,  # Include all pixels touched by geometries
            dtype=np.uint8  # 8-bit unsigned integer for binary mask
        )

        # Resize image and mask to target size
        image = resize(image.transpose(1, 2, 0), self.target_size + (self.num_bands,), mode='reflect', anti_aliasing=True)  # Resize image
        image = image.transpose(2, 0, 1).astype(np.float32)  # Transpose back to (C, H, W) and convert to float32
        # Normalize each channel of the image
        for c in range(image.shape[0]):
            channel = image[c]  # Select channel
            min_val = channel.min()  # Minimum value in channel
            max_val = channel.max()  # Maximum value in channel
            if max_val - min_val > 1e-6:  # Avoid division by zero
                image[c] = (channel - min_val) / (max_val - min_val)  # Normalize to [0, 1]
            else:
                image[c] = 0  # Set to 0 if channel has no variation

        # Resize mask (no interpolation for binary data)
        mask = resize(mask, self.target_size, order=0, anti_aliasing=False).astype(np.uint8)  # Resize mask
        mask = mask[None, :, :]  # Add channel dimension (1, H, W)

        # Return image and mask as PyTorch tensors
        return torch.from_numpy(image), torch.from_numpy(mask)

def match_raster_shapefile(image_base_dir, label_base_dir):
    """
    Search for all GeoTIFFs and Shapefiles in given directories (including subdirectories),
    then return matched pairs by checking if the shapefile name starts with the raster base name.
    """
    # Collect all GeoTIFFs and Shapefiles
    image_files = glob.glob(os.path.join(image_base_dir, "**", "*.tif"), recursive=True) # Recursive search
    label_files = glob.glob(os.path.join(label_base_dir, "**", "*.shp"), recursive=True) # Recursive search

    print(f"Found {len(image_files)} GeoTIFFs and {len(label_files)} shapefiles") # Print counts

    def get_base_name(path): # Function to get base name
        return os.path.splitext(os.path.basename(path))[0] # Get base name

    pairs = []
    for img in image_files:
        base = get_base_name(img)  # e.g., "T47PMS_20250329T034601"
        # Find shapefiles whose names start with the raster base
        matches = [lbl for lbl in label_files if get_base_name(lbl).startswith(base)] # e.g., "T47PMS_20250329T034601"
        if matches:
            # If multiple matches, take the first one (or adjust to your rule)
            pairs.append((img, matches[0]))

    print(f"Matched {len(pairs)} raster–label pairs") # Print count
    return pairs

def get_preds_labels(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Collecting predictions"):
            images = images.to(device)
            masks = masks.to(device).float()
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            preds = preds.cpu().numpy().flatten()
            masks = masks.cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(masks)
    return all_preds, all_labels

def compute_report_and_cm(all_labels, all_preds):
    report = classification_report(all_labels, all_preds, target_names=['Unburn', 'Burn'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print("\nCombined Classification Report:")
    print(report_df)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Unburn', 'Burn'], yticklabels=['Unburn', 'Burn'])
    plt.title('Combined Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('U-Net Confusion Matrix.png')
    plt.show()

    return report_df, cm

# Main Training Script
def main():
    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Directories ---
    image_base_dir = r"Raster_Classified"
    label_base_dir = r"Wildfire_Polygon"

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

    # --- Match raster–shapefile pairs ---
    pairs = match_raster_shapefile(image_base_dir, label_base_dir)
    if not pairs:
        print("No matching raster–label pairs found!")
        return

    tile_size = 256

    models_info = []

    # --- Loop over each raster–label pair for training ---
    for image_path, label_path in pairs:
        print(f"\n=== Processing {os.path.basename(image_path)} ===")

        # --- Generate all windows (tiles) ---
        with rasterio.open(image_path) as src:
            height, width = src.shape
            bands = list(range(1, src.count + 1))
            num_channels = len(bands)
            windows = []
            for i in range(0, height, tile_size):
                for j in range(0, width, tile_size):
                    w = min(tile_size, height - i)
                    h = min(tile_size, width - j)
                    if w == tile_size and h == tile_size:
                        windows.append(Window(j, i, tile_size, tile_size))

        if len(windows) == 0:
            print("No full tiles available. Image may be smaller than tile size.")
            continue

        # --- Classify each tile as Burn vs Unburn ---
        with rasterio.open(image_path) as src:
            img_transform = src.transform
            img_crs = src.crs

        gdf = gpd.read_file(label_path)
        if gdf.crs != img_crs:
            gdf = gdf.to_crs(img_crs)

        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
        if gdf.empty:
            print("Label shapefile is empty after cleaning. Skipping...")
            continue

        sindex = gdf.sindex
        from shapely.geometry import box

        burn_tiles, unburn_tiles = [], []
        for w in windows:
            left, bottom, right, top = rasterio.windows.bounds(w, img_transform)
            tile_poly = box(left, bottom, right, top)
            hits = list(sindex.query(tile_poly, predicate="intersects"))
            if len(hits) > 0:
                burn_tiles.append(w)
            else:
                unburn_tiles.append(w)

        print(f"Total tiles: {len(windows)}, Burn: {len(burn_tiles)}, Unburn: {len(unburn_tiles)}")

        # --- Downsample the larger class to balance dataset ---
        rng = np.random.RandomState(42)
        n_burn = len(burn_tiles)
        n_unburn = len(unburn_tiles)
        if n_burn == 0 or n_unburn == 0:
            print("One class has no tiles (Burn or Unburn). Skipping...")
            continue
        n_keep = min(n_burn, n_unburn)

        if n_burn > n_keep:
            burn_tiles = list(rng.choice(burn_tiles, size=n_keep, replace=False))
        elif n_unburn > n_keep:
            unburn_tiles = list(rng.choice(unburn_tiles, size=n_keep, replace=False))

        # Combine tiles and create corresponding labels for stratification
        balanced_tiles = burn_tiles + unburn_tiles
        tile_labels = [1] * len(burn_tiles) + [0] * len(unburn_tiles)

        # --- Train/val split with stratification ---
        train_windows, val_windows, _, _ = train_test_split(
            balanced_tiles,
            tile_labels,
            test_size=0.2,
            random_state=42,
            stratify=tile_labels
        )
        print(f"Balanced dataset → Train: {len(train_windows)}, Val: {len(val_windows)}")

        # --- Create datasets & dataloaders ---
        train_dataset = SegmentationDataset(image_path, label_path, train_windows, bands=bands)
        val_dataset = SegmentationDataset(image_path, label_path, val_windows, bands=bands)

        train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

        # --- Model, loss, optimizer ---
        model = UNet(n_channels=num_channels, n_classes=1).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # --- Training loop ---
        num_epochs = 30
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
            for images, masks in progress_bar:
                images = images.to(device)
                masks = masks.to(device).float()

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            avg_loss = running_loss / len(train_dataloader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # --- Save model ---
        model_name = f"unet_model_{os.path.splitext(os.path.basename(image_path))[0]}.pth"
        save_path = os.path.join("Export_Model", model_name)
        os.makedirs("Export_Model", exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Training complete for this pair. Model saved as {save_path}")

        # Collect info for later evaluation
        models_info.append({
            'model_path': save_path,
            'image_path': image_path,
            'label_path': label_path,
            'val_windows': val_windows,
            'bands': bands,
            'device': device
        })

    # --- After all trainings, perform combined evaluation ---
    if models_info:
        global_all_preds = []
        global_all_labels = []
        for info in models_info:
            model = UNet(n_channels=len(info['bands']), n_classes=1).to(info['device'])
            model.load_state_dict(torch.load(info['model_path']))
            val_dataset = SegmentationDataset(info['image_path'], info['label_path'], info['val_windows'], bands=info['bands'])
            val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
            preds, labels = get_preds_labels(model, val_dataloader, info['device'])
            global_all_preds.extend(preds)
            global_all_labels.extend(labels)

        # Compute and display combined report and confusion matrix
        compute_report_and_cm(global_all_labels, global_all_preds)
    else:
        print("No models were trained. Skipping combined evaluation.")

# Main Training Script
if __name__ == "__main__":
    main()