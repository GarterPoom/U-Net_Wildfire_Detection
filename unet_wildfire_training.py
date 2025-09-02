# Import necessary libraries
import os  # Used for interacting with the operating system, like managing file paths.
import glob  # Used for finding files and directories that match a specific pattern (e.g., all .tif files).
import torch  # The main PyTorch library for building and training neural networks.
import torch.nn as nn  # Contains core neural network modules (like layers, activation functions, etc.).
import torch.optim as optim  # Contains optimization algorithms (like Adam) for training models.
import platform  # Provides access to the underlying platform's system data (e.g., OS name).
import psutil  # A cross-platform library for retrieving information on running processes and system utilization (CPU, memory, disks).
from torch.utils.data import Dataset, DataLoader  # Tools for handling data, creating custom datasets, and loading data in batches.
import rasterio  # A library for reading and writing raster data (like GeoTIFFs), especially useful for geospatial imagery.
from rasterio.windows import Window  # Represents a sub-region (tile) of a raster dataset.
from rasterio.features import rasterize  # A function to convert vector data (polygons) into a raster format.
import geopandas as gpd  # A library that extends the pandas data structure to allow spatial operations on geometric types (like shapefiles).
from shapely.geometry import mapping, box  # 'mapping' converts a geometry to a GeoJSON-like dictionary; 'box' creates a rectangular polygon.
import numpy as np  # The fundamental package for scientific computing in Python, used for array manipulation.
from skimage.transform import resize  # Used for resizing images or masks.
from tqdm import tqdm  # A library to display smart progress bars for loops.
import pandas as pd  # A powerful data analysis and manipulation library, used here for creating dataframes.
import seaborn as sns  # A library for creating informative and attractive statistical graphics, built on top of Matplotlib.
import matplotlib.pyplot as plt  # A comprehensive library for creating static, animated, and interactive visualizations in Python.
from sklearn.metrics import classification_report, confusion_matrix  # Used to evaluate the model's performance.
from sklearn.model_selection import train_test_split  # A function for splitting data into training and validation sets.

# ---------------------------------------------------------------------
# Define the UNet model architecture
# ---------------------------------------------------------------------

# DoubleConv Module
# This module performs two consecutive convolutional operations, each followed by Batch Normalization and ReLU activation.
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # First 2D convolutional layer
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # Normalizes the activations of the previous layer for stable training.
            nn.BatchNorm2d(out_channels),
            # Applies the rectified linear unit activation function.
            nn.ReLU(inplace=True),
            # Second 2D convolutional layer
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # Normalizes the activations again.
            nn.BatchNorm2d(out_channels),
            # Applies ReLU again.
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Defines the forward pass of the module.
        return self.double_conv(x)

# Down Module
# This module handles the down-sampling part of the U-Net architecture.
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            # Down-samples the feature map by taking the maximum value in a 2x2 window.
            nn.MaxPool2d(2),
            # Applies the DoubleConv module to the down-sampled feature map.
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        # Defines the forward pass of the down-sampling module.
        return self.maxpool_conv(x)

# Up Module
# This module handles the up-sampling and concatenation part of the U-Net.
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Transposed convolution layer for up-sampling the feature map.
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # Applies the DoubleConv module after up-sampling and concatenation.
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # Up-samples the first input tensor (x1).
        x1 = self.up(x1)
        # Calculates padding to match the dimensions of the skip connection tensor (x2).
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # Pads x1 to match the height and width of x2.
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # Concatenates the padded x1 with the skip connection x2 along the channel dimension.
        x = torch.cat([x2, x1], dim=1)
        # Applies the double convolution to the concatenated tensor.
        return self.conv(x)

# OutConv Module
# The final 1x1 convolution layer that maps the feature map to the desired number of output classes.
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1x1 convolution to produce the final segmentation map.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Defines the forward pass.
        return self.conv(x)

# UNet Model
# The complete U-Net architecture, combining the Down, Up, and Conv modules.
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # Initial double convolution.
        self.inc = DoubleConv(n_channels, 64)
        # Down-sampling blocks.
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        # Up-sampling blocks with skip connections.
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        # Final output convolution.
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder path (down-sampling).
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # Decoder path (up-sampling with skip connections).
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # Final output logits.
        logits = self.outc(x)
        return logits

# ---------------------------------------------------------------------
# Custom Dataset and Utility Functions
# ---------------------------------------------------------------------

# SegmentationDataset
# Custom PyTorch Dataset for loading image tiles and corresponding segmentation masks.
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, label_paths, windows, target_size=(256, 256), bands=None):
        # Initializes the dataset with file paths, tile windows, and other parameters.
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.windows = windows
        self.target_size = target_size
        self.bands = bands
        # Determines the number of bands (channels) from the first image.
        self.num_bands = len(bands[0]) if bands else None
        self.crs_list = []  # Stores the CRS (Coordinate Reference System) for each image.
        self.gdf_list = []  # Stores the GeoPandas DataFrames for each label shapefile.
        # Loop to pre-load CRS and GeoDataFrames for all images and labels.
        for image_path, label_path in zip(image_paths, label_paths):
            with rasterio.open(image_path) as src:
                self.crs_list.append(src.crs)
            gdf = gpd.read_file(label_path)
            # Projects the shapefile to match the raster's CRS if they are different.
            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)
            self.gdf_list.append(gdf)

    def __len__(self):
        # Returns the total number of tiles (windows) in the dataset.
        return len(self.windows)

    def __getitem__(self, idx):
        # Retrieves an item (image tile and mask) by its index.
        img_idx, window = self.windows[idx]
        image_path = self.image_paths[img_idx]
        label_path = self.label_paths[img_idx]
        gdf = self.gdf_list[img_idx]
        crs = self.crs_list[img_idx]

        with rasterio.open(image_path) as src:
            # Reads the image data for the specified window and bands.
            image = src.read(self.bands[img_idx], window=window)
            # Gets the affine transform for the current window.
            tile_transform = src.window_transform(window)
            height = window.height
            width = window.width
            # Gets the geospatial bounds of the current window.
            bounds = rasterio.windows.bounds(window, src.transform)

        # Clips the GeoDataFrame to the boundaries of the current image tile.
        clipped_gdf = gdf.clip(bounds)
        # Prepares the shapes (geometries) for rasterization.
        shapes = [(mapping(geom), 1) for geom in clipped_gdf.geometry if not geom.is_empty]
        # Rasterizes the clipped polygons into a binary mask.
        mask = rasterize(
            shapes,
            out_shape=(height, width),
            transform=tile_transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8
        )

        # Resizes the image to the target size.
        image = resize(image.transpose(1, 2, 0), self.target_size + (self.num_bands,), mode='reflect', anti_aliasing=True)
        # Changes the image dimensions from (H, W, C) to (C, H, W) for PyTorch.
        image = image.transpose(2, 0, 1).astype(np.float32)
        # Normalizes each channel of the image to a 0-1 range.
        for c in range(image.shape[0]):
            channel = image[c]
            min_val = channel.min()
            max_val = channel.max()
            if max_val - min_val > 1e-6:
                image[c] = (channel - min_val) / (max_val - min_val)
            else:
                image[c] = 0

        # Resizes the mask to the target size. `order=0` means nearest-neighbor interpolation.
        mask = resize(mask, self.target_size, order=0, anti_aliasing=False).astype(np.uint8)
        # Adds a channel dimension to the mask.
        mask = mask[None, :, :]

        # Returns the image and mask as PyTorch tensors.
        return torch.from_numpy(image), torch.from_numpy(mask)

# match_raster_shapefile
# Finds matching raster and shapefile pairs based on their base filename.
def match_raster_shapefile(image_base_dir, label_base_dir):
    # Finds all GeoTIFF files recursively in the image directory.
    image_files = glob.glob(os.path.join(image_base_dir, "**", "*.tif"), recursive=True)
    # Finds all shapefiles recursively in the label directory.
    label_files = glob.glob(os.path.join(label_base_dir, "**", "*.shp"), recursive=True)
    print(f"Found {len(image_files)} GeoTIFFs and {len(label_files)} shapefiles")

    # Helper function to get the filename without extension.
    def get_base_name(path):
        return os.path.splitext(os.path.basename(path))[0]

    pairs = []
    # Iterates through image files to find a matching label file.
    for img in image_files:
        base = get_base_name(img)
        # Finds shapefiles whose base name starts with the image's base name.
        matches = [lbl for lbl in label_files if get_base_name(lbl).startswith(base)]
        if matches:
            pairs.append((img, matches[0]))

    print(f"Matched {len(pairs)} raster–label pairs")
    return pairs

# get_preds_labels
# Performs inference on the validation set and collects predictions and true labels.
def get_preds_labels(model, dataloader, device):
    model.eval()  # Sets the model to evaluation mode.
    all_preds = []
    all_labels = []
    # Disables gradient calculation to save memory and speed up inference.
    with torch.no_grad():
        # Loops through the dataloader with a progress bar.
        for images, masks in tqdm(dataloader, desc="Collecting predictions"):
            # Moves data to the specified device (GPU or CPU).
            images = images.to(device)
            masks = masks.to(device).float()
            # Performs a forward pass to get the model's output.
            outputs = model(images)
            # Applies sigmoid and a threshold to convert logits to binary predictions.
            preds = torch.sigmoid(outputs) > 0.5
            # Flattens the predictions and masks to a 1D array for evaluation.
            preds = preds.cpu().numpy().flatten()
            masks = masks.cpu().numpy().flatten()
            # Adds the batch predictions and labels to the overall lists.
            all_preds.extend(preds)
            all_labels.extend(masks)
    return all_preds, all_labels

# compute_report_and_cm
# Calculates and visualizes the classification report and confusion matrix.
def compute_report_and_cm(all_labels, all_preds):
    # Generates a classification report from scikit-learn.
    report = classification_report(all_labels, all_preds, target_names=['Unburn', 'Burn'], output_dict=True)
    # Converts the report dictionary to a pandas DataFrame for better readability.
    report_df = pd.DataFrame(report).transpose()
    print("\nCombined Classification Report:")
    print(report_df)

    # Computes the confusion matrix.
    cm = confusion_matrix(all_labels, all_preds)
    # Creates a plot for the confusion matrix using seaborn.
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Unburn', 'Burn'], yticklabels=['Unburn', 'Burn'])
    plt.title('Combined Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    # Saves the plot to a file.
    plt.savefig('U-Net Confusion Matrix.png')
    plt.show()  # Displays the plot.

    return report_df, cm

# ---------------------------------------------------------------------
# Main Execution Function
# ---------------------------------------------------------------------

def main():
    # Determines and prints the device to be used for training (GPU or CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Defines the directories where raster images and shapefile labels are stored.
    image_base_dir = r"Raster_Classified"
    label_base_dir = r"Wildfire_Polygon"

    # Gathers and prints system information, including GPU or CPU details.
    if device.type == "cuda":
        num_gpus = torch.cuda.device_count()
        print(f" -> {num_gpus} CUDA device(s) available")
        for i in range(num_gpus):
            print(f"   [GPU {i}] {torch.cuda.get_device_name(i)}")
            print(f"       Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
            print(f"       Memory Cached:    {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")
            print(f"       Total Memory:     {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.2f} GB")
            print(f"       Compute Capability: {torch.cuda.get_device_capability(i)}")
    else:
        print(f" -> CPU: {platform.processor() or 'Unknown'}")
        print(f" -> CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
        print(f" -> RAM available: {psutil.virtual_memory().total / 1024 ** 3:.2f} GB")

    # Matches image and label files.
    pairs = match_raster_shapefile(image_base_dir, label_base_dir)
    if not pairs:
        print("No matching raster–label pairs found!")
        return

    # Defines the tile size for splitting large images.
    tile_size = 256
    all_windows = []
    all_image_paths = []
    all_label_paths = []
    all_bands = []
    max_channels = 0

    # Collects all windows (tiles) from all matched image pairs.
    for image_path, label_path in pairs:
        print(f"\n=== Processing {os.path.basename(image_path)} ===")
        with rasterio.open(image_path) as src:
            height, width = src.shape
            bands = list(range(1, src.count + 1))
            max_channels = max(max_channels, len(bands))
            img_transform = src.transform
            img_crs = src.crs

        gdf = gpd.read_file(label_path)
        if gdf.crs != img_crs:
            gdf = gdf.to_crs(img_crs)
        # Removes invalid or empty geometries from the GeoDataFrame.
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
        if gdf.empty:
            print("Label shapefile is empty after cleaning. Skipping...")
            continue

        windows = []
        # Generates windows for all possible tiles within the image.
        for i in range(0, height, tile_size):
            for j in range(0, width, tile_size):
                w = min(tile_size, height - i)
                h = min(tile_size, width - j)
                if w == tile_size and h == tile_size:
                    windows.append(Window(j, i, tile_size, tile_size))

        if not windows:
            print("No full tiles available. Image may be smaller than tile size.")
            continue

        # Uses a spatial index to quickly find intersections between tile windows and polygons.
        sindex = gdf.sindex
        burn_tiles, unburn_tiles = [], []
        # Iterates through windows to classify them as 'burn' or 'unburn' based on intersection with labels.
        for w in windows:
            left, bottom, right, top = rasterio.windows.bounds(w, img_transform)
            tile_poly = box(left, bottom, right, top)
            # Queries the spatial index for intersecting polygons.
            hits = list(sindex.query(tile_poly, predicate="intersects"))
            if len(hits) > 0:
                burn_tiles.append((len(all_image_paths), w))
            else:
                unburn_tiles.append((len(all_image_paths), w))

        print(f"Total tiles: {len(windows)}, Burn: {len(burn_tiles)}, Unburn: {len(unburn_tiles)}")
        # Appends the tiles to the master list.
        all_windows.extend(burn_tiles + unburn_tiles)
        all_image_paths.append(image_path)
        all_label_paths.append(label_path)
        all_bands.append(bands)

    if not all_windows:
        print("No tiles available across all pairs. Exiting...")
        return

    # Balances the dataset by oversampling or undersampling to have an equal number of 'burn' and 'unburn' tiles.
    burn_tiles = [w for w in all_windows if w in burn_tiles]
    unburn_tiles = [w for w in all_windows if w not in burn_tiles]
    rng = np.random.RandomState(42)  # Sets a random seed for reproducibility.
    n_burn = len(burn_tiles)
    n_unburn = len(unburn_tiles)
    if n_burn == 0 or n_unburn == 0:
        print("One class has no tiles (Burn or Unburn). Exiting...")
        return
    n_keep = min(n_burn, n_unburn)

    if n_burn > n_keep:
        burn_tiles = list(rng.choice(burn_tiles, size=n_keep, replace=False))
    elif n_unburn > n_keep:
        unburn_tiles = list(rng.choice(unburn_tiles, size=n_keep, replace=False))

    balanced_tiles = burn_tiles + unburn_tiles
    tile_labels = [1] * len(burn_tiles) + [0] * len(unburn_tiles)

    # Splits the balanced dataset into training and validation sets.
    train_windows, val_windows, _, _ = train_test_split(
        balanced_tiles,
        tile_labels,
        test_size=0.2,
        random_state=42,
        stratify=tile_labels  # Ensures the split maintains the same class ratio.
    )
    print(f"Combined dataset → Train: {len(train_windows)}, Val: {len(val_windows)}")

    # Adjusts the bands to be consistent across all images, using the max number of channels found.
    adjusted_bands = [b[:max_channels] for b in all_bands]

    # Creates instances of the custom dataset for training and validation.
    train_dataset = SegmentationDataset(all_image_paths, all_label_paths, train_windows, bands=adjusted_bands)
    val_dataset = SegmentationDataset(all_image_paths, all_label_paths, val_windows, bands=adjusted_bands)

    # Creates data loaders to handle batching and shuffling of data.
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Initializes the U-Net model, the loss function, and the optimizer.
    model = UNet(n_channels=max_channels, n_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy with Logits loss, suitable for binary segmentation.
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adam optimizer for training.

    # Training loop.
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()  # Sets the model to training mode.
        running_loss = 0.0
        # Loops through the training data with a progress bar.
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device).float()
            optimizer.zero_grad()  # Clears previous gradients.
            outputs = model(images)  # Forward pass.
            loss = criterion(outputs, masks)  # Calculates the loss.
            loss.backward()  # Backpropagation to compute gradients.
            optimizer.step()  # Updates model parameters.
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        avg_loss = running_loss / len(train_dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Saves the trained model's state dictionary.
    save_path = os.path.join("Export_Model", "unet_single_model.pth")
    os.makedirs("Export_Model", exist_ok=True)  # Creates the export directory if it doesn't exist.
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved as {save_path}")

    # Evaluation phase.
    all_preds, all_labels = get_preds_labels(model, val_dataloader, device)
    compute_report_and_cm(all_labels, all_preds)

# Entry point of the script.
if __name__ == "__main__":
    main()