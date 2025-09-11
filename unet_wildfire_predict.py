# unet_wildfire_predict.py
# Standalone script to use a trained U-Net model for predicting wildfire burn masks on new GeoTIFF images.

# --------------------- Imports and Environment Setup ---------------------

import os
# The following environment variables are set to prevent numerical libraries (like NumPy, PyTorch)
# from using multiple CPU threads. This can sometimes improve performance for inference tasks by
# avoiding overhead, especially when the main workload is on the GPU.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1" 

import torch  # The main PyTorch library for deep learning.
import torch.nn as nn  # PyTorch's neural network module, used to build the U-Net.
import rasterio  # A library for reading and writing geospatial raster data, like GeoTIFF files.
import numpy as np  # A library for numerical operations, especially with arrays.
from tqdm import tqdm  # A library for creating smart progress bars for loops.
import matplotlib.pyplot as plt  # A library for plotting and visualizing data.
import argparse  # A library for parsing command-line arguments, making the script user-friendly.
import glob  # A library for finding files that match a specific pattern (e.g., all '.tif' files).


# --------------------- U-Net Model Definition ---------------------
# This section defines the U-Net architecture, which is composed of several building blocks.

class DoubleConv(nn.Module):
    """(Convolution => Batch Normalization => ReLU) * 2"""
    # This block applies two consecutive 2D convolutions, each followed by
    # batch normalization and a ReLU activation function. This is a fundamental
    # building block in the U-Net architecture.
    def __init__(self, in_channels, out_channels):
        # The constructor for the DoubleConv block.
        super().__init__() # Initializes the parent nn.Module class.
        self.double_conv = nn.Sequential(
            # First convolutional layer. `padding=1` keeps the image size the same for a 3x3 kernel.
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # Batch normalization helps stabilize and speed up training.
            nn.BatchNorm2d(out_channels),
            # ReLU (Rectified Linear Unit) is the activation function. `inplace=True` modifies the input directly to save memory.
            nn.ReLU(inplace=True),
            # Second convolutional layer.
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # Second batch normalization layer.
            nn.BatchNorm2d(out_channels),
            # Second ReLU activation.
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Defines the forward pass of the data `x` through the layers.
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    # This block is for the "contracting path" (encoder) of the U-Net.
    # It first halves the image dimensions using max pooling and then applies a DoubleConv block.
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            # Max pooling layer with a 2x2 window, which reduces height and width by half.
            nn.MaxPool2d(2),
            # The DoubleConv block to process the feature maps after downscaling.
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        # Defines the forward pass.
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    # This block is for the "expansive path" (decoder) of the U-Net. It upscales the feature map
    # and concatenates it with the corresponding feature map from the contracting path (skip connection).
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Transposed convolution (sometimes called deconvolution) to double the height and width of the feature map.
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # A DoubleConv block to process the concatenated feature maps.
        # The input channels are doubled because we concatenate the upscaled map with the skip connection.
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # `x1` is the feature map from the previous decoder layer (to be upscaled).
        # `x2` is the feature map from the corresponding encoder layer (the skip connection).
        
        # Upscale the feature map from the previous layer.
        x1 = self.up(x1)
        
        # Due to rounding in convolutions, there might be a size mismatch between the upscaled map (x1)
        # and the skip connection map (x2). This code adds padding to x1 to make them the same size.
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        # Pad x1 to match the dimensions of x2.
        x1 = nn.functional.pad(x1,
                               [diffX // 2, diffX - diffX // 2, # Pad left/right
                                diffY // 2, diffY - diffY // 2]) # Pad top/bottom
                               
        # Concatenate the skip connection tensor (x2) and the upscaled tensor (x1) along the channel dimension.
        x = torch.cat([x2, x1], dim=1)
        
        # Apply the DoubleConv block to the concatenated tensor.
        return self.conv(x)


class OutConv(nn.Module):
    """Final output convolution"""
    # This is the final layer of the U-Net. It uses a 1x1 convolution to map the feature channels
    # to the desired number of output classes.
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # A single 1x1 convolutional layer.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Defines the forward pass.
        return self.conv(x)


class UNet(nn.Module):
    """The full U-Net model architecture."""
    # This class assembles all the building blocks (DoubleConv, Down, Up, OutConv)
    # into the complete U-Net architecture.
    def __init__(self, n_channels, n_classes):
        # `n_channels`: number of input channels in the image (e.g., 3 for RGB).
        # `n_classes`: number of output classes (e.g., 1 for a binary burn mask).
        super(UNet, self).__init__()
        
        # --- Encoder (Contracting Path) ---
        self.inc = DoubleConv(n_channels, 64)    # Initial convolution block.
        self.down1 = Down(64, 128)               # First downscaling block.
        self.down2 = Down(128, 256)              # Second downscaling block.
        self.down3 = Down(256, 512)              # Third downscaling block.
        self.down4 = Down(512, 1024)             # Fourth downscaling block (bottleneck).

        # --- Decoder (Expansive Path) ---
        self.up1 = Up(1024, 512)                 # First upscaling block.
        self.up2 = Up(512, 256)                  # Second upscaling block.
        self.up3 = Up(256, 128)                  # Third upscaling block.
        self.up4 = Up(128, 64)                   # Fourth upscaling block.
        self.outc = OutConv(64, n_classes)       # Final output convolution.

    def forward(self, x):
        # Defines the complete forward pass, connecting all the blocks.
        # The output of each encoder block is saved to be used as a skip connection.
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # The decoder path uses the output from the previous decoder layer AND the skip connection.
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # The final output layer produces the raw prediction values (logits).
        logits = self.outc(x)
        return logits


# --------------------- Helper Functions ---------------------

def get_tiff_files(input_path, recursive=True):
    """Finds all GeoTIFF files in a given directory."""
    # Sets the search pattern to be recursive ('**/*') or non-recursive ('*').
    pattern = '**/*.tif' if recursive else '*.tif'
    # Use glob to find all files matching the '.tif' pattern.
    files = glob.glob(os.path.join(input_path, pattern), recursive=recursive)
    # Also search for files with the '.tiff' extension and add them to the list.
    files += glob.glob(os.path.join(input_path, pattern.replace('.tif', '.tiff')), recursive=recursive)
    # Filter the list to ensure only files (not directories) are included, and then sort it.
    files = sorted([f for f in files if os.path.isfile(f)])
    return files


def generate_output_path(input_path, input_base_dir, output_dir, suffix="_predicted_mask", preserve_structure=False):
    """Generates a full path for the output file."""
    # Create the main output directory if it doesn't already exist.
    os.makedirs(output_dir, exist_ok=True)
    # Get the base name of the input file without its extension (e.g., 'image1' from 'image1.tif').
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # If the user wants to keep the original folder structure inside the output directory.
    if preserve_structure:
        # Get the relative path of the input file from its base directory.
        rel_path = os.path.relpath(os.path.dirname(input_path), input_base_dir)
        # Create the same subdirectory structure within the output directory.
        output_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(output_subdir, exist_ok=True)
        # Return the full path for the new file inside the replicated structure.
        return os.path.join(output_subdir, f"{base_name}{suffix}.tif")
    else:
        # If not preserving structure, just save the file directly in the main output directory.
        return os.path.join(output_dir, f"{base_name}{suffix}.tif")


# --------------------- Prediction Function ---------------------

def predict_on_new_image(
    model_path, new_image_path, output_path, prob_output_path,
    tile_size=256, overlap=32, device=None, visualize=False
):
    """
    Processes a single large GeoTIFF image by tiling it, running predictions,
    and stitching the results back together.
    """
    # If a specific device (CPU/GPU) is not provided, automatically detect if a CUDA-enabled GPU is available.
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    # Temporarily open the image just to get the number of channels/bands.
    with rasterio.open(new_image_path) as src:
        num_channels = src.count
    # Instantiate the U-Net model with the correct number of input channels.
    model = UNet(n_channels=num_channels, n_classes=1).to(device)
    # Load the saved weights from the trained model file. `map_location=device` ensures it loads correctly on CPU or GPU.
    model.load_state_dict(torch.load(model_path, map_location=device))
    # Set the model to evaluation mode. This disables layers like Dropout and BatchNorm that behave differently during training.
    model.eval()

    # --- Load and Normalize Image ---
    # Open the GeoTIFF file for reading.
    with rasterio.open(new_image_path) as src:
        # Get the image dimensions (height, width).
        height, width = src.shape
        # Copy the metadata (projection, transform, etc.) to use when saving the output.
        meta = src.meta.copy()
        # Read all bands of the image into a NumPy array with float32 data type.
        image = src.read().astype(np.float32)  # Shape: (bands, H, W)

    # Normalize each band of the image independently.
    for c in range(image.shape[0]):
        band = image[c]
        # Calculate the 2nd and 98th percentiles to clip extreme values (like sensor noise), making normalization more robust.
        min_val, max_val = np.percentile(band, 2), np.percentile(band, 98)
        # Avoid division by zero if the band is flat (all pixels have the same value).
        if max_val - min_val > 1e-6:
            # Scale the pixel values to be between 0 and 1. `np.clip` ensures values don't go outside this range.
            image[c] = np.clip((band - min_val) / (max_val - min_val), 0, 1)
        else:
            # If the band is flat, set it to all zeros.
            image[c] = 0

    # --- Create Buffers for Stitching ---
    # A buffer to accumulate the sum of predictions in overlapping areas.
    pred_sum = np.zeros((height, width), dtype=np.float32)
    # A buffer to count how many predictions have been made for each pixel.
    pred_count = np.zeros((height, width), dtype=np.uint16)

    # --- Tiled Prediction Loop ---
    # Calculate the step size for moving the sliding window (tile).
    step = tile_size - overlap
    # Disable gradient calculations to speed up inference and reduce memory usage.
    with torch.no_grad():
        # Loop over the image height with the calculated step size. `tqdm` creates a progress bar.
        for i in tqdm(range(0, height, step), desc="Predicting"):
            # Loop over the image width.
            for j in range(0, width, step):
                # Calculate the end coordinates for the current tile, ensuring it doesn't go beyond the image boundary.
                row_end = min(i + tile_size, height)
                col_end = min(j + tile_size, width)

                # Extract the tile (a small patch of the image).
                tile = image[:, i:row_end, j:col_end]

                # --- Pad Tile if Necessary ---
                # Tiles at the right and bottom edges might be smaller than `tile_size`.
                # We need to pad them with zeros to make them the standard size for the model.
                pad_h = tile_size - tile.shape[1]
                pad_w = tile_size - tile.shape[2]
                if pad_h > 0 or pad_w > 0:
                    tile = np.pad(
                        tile, ((0, 0), (0, pad_h), (0, pad_w)), # Pad height and width dims, not channel dim.
                        mode="constant", constant_values=0
                    )
                
                # --- Predict on Tile ---
                # Convert the NumPy tile to a PyTorch tensor, add a batch dimension, and move it to the correct device (GPU/CPU).
                tile_tensor = torch.from_numpy(tile).unsqueeze(0).to(device)
                # Run the model's forward pass to get the prediction (logits).
                output = model(tile_tensor)
                # Apply the sigmoid function to convert logits to probabilities (0-1), then move back to CPU and NumPy.
                prob = torch.sigmoid(output).cpu().numpy().squeeze()

                # --- Add Prediction to Buffers ---
                # Remove the padding that was added to the prediction before accumulating.
                prob = prob[: row_end - i, : col_end - j]
                
                # Add the tile's probability mask to the overall sum buffer.
                pred_sum[i:row_end, j:col_end] += prob
                # Increment the count for each pixel in this tile's location.
                pred_count[i:row_end, j:col_end] += 1

    # --- Finalize Predictions ---
    # Average the predictions by dividing the sum by the count. `np.maximum` prevents division by zero.
    avg_pred = pred_sum / np.maximum(pred_count, 1)
    
    # --- Save Probability Map (without threshold) ---
    # Update metadata for probability map: single band, float32, no nodata value, LZW compression
    prob_meta = meta.copy()
    prob_meta.update(count=1, dtype="float32", nodata=None, compress="lzw")
    
    # Create probability output directory if it doesn't exist
    os.makedirs(os.path.dirname(prob_output_path), exist_ok=True)
    
    # Save the probability map
    with rasterio.open(prob_output_path, "w", **prob_meta) as dst:
        dst.write(avg_pred.astype(np.float32), 1)
    
    print(f"✅ Probability map saved to {prob_output_path}")

    # --- Create Binary Mask and Save ---
    # Create the final binary mask by applying a threshold of 0.5. Pixels with probability > 0.5 are classified as "burned" (1).
    full_mask = (avg_pred > 0.5).astype(np.uint8)

    # --- Save the Binary Mask GeoTIFF ---
    # Update the metadata for the output file: single band, 8-bit integer, a 'no data' value, and LZW compression.
    meta.update(count=1, dtype="uint8", nodata=255, compress="lzw")
    # Open a new GeoTIFF file in write mode with the updated metadata.
    with rasterio.open(output_path, "w", **meta) as dst:
        # Write the final mask to the first band of the new file.
        dst.write(full_mask, 1)

    print(f"✅ Binary mask saved to {output_path}")
    print("To view in QGIS: Style → Singleband pseudocolor → 0=Unburned, 1=Burned")

    # --- Visualize (Optional) ---
    # If the visualize flag is set, display the predicted mask using matplotlib.
    if visualize:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot probability map
        im1 = ax1.imshow(avg_pred, cmap="viridis", vmin=0, vmax=1)
        ax1.set_title(f"Probability Map for {os.path.basename(new_image_path)}")
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Plot binary mask
        im2 = ax2.imshow(full_mask, cmap="gray", vmin=0, vmax=1)
        ax2.set_title(f"Binary Mask (threshold=0.5)")
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, ticks=[0, 1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()


# --------------------- Main Execution Block ---------------------

def main():
    """Parses command-line arguments and orchestrates the prediction process."""
    # Create an ArgumentParser object to handle command-line inputs.
    parser = argparse.ArgumentParser(description="Predict wildfire burn mask using U-Net model")
    
    # Define all the command-line arguments the script can accept.
    parser.add_argument('--model_path', default="Export_Model/unet_wildfire_no_shape_file.pth", help="Path to trained model")
    parser.add_argument('--image_path', default="Raster_Classified", help="Path to input GeoTIFF or directory of GeoTIFFs")
    parser.add_argument('--output_dir', default="Predicted_Mask", help="Directory to save output binary masks")
    parser.add_argument('--prob_output_dir', default="Predicted_Probability", help="Directory to save output probability maps")
    parser.add_argument('--tile_size', type=int, default=256, help="Tile size for processing")
    parser.add_argument('--overlap', type=int, default=32, help="Overlap between tiles")
    parser.add_argument('--visualize', action='store_true', help="Visualize the predicted mask")
    parser.add_argument('--recursive', action='store_true', default=True, help="Search for GeoTIFFs in subdirectories")
    parser.add_argument('--preserve_structure', action='store_true', help="Preserve input directory structure in output")
    
    # Parse the arguments provided by the user when running the script.
    args = parser.parse_args()

    # Normalize file paths to handle different OS path separators (e.g., / vs \).
    args.image_path = os.path.normpath(args.image_path)
    args.output_dir = os.path.normpath(args.output_dir)
    args.prob_output_dir = os.path.normpath(args.prob_output_dir)
    args.model_path = os.path.normpath(args.model_path)

    # --- Find Input Files ---
    # Check if the provided image path is a single file.
    if os.path.isfile(args.image_path):
        tiff_files = [args.image_path] # Create a list containing just that file.
        base_dir = os.path.dirname(args.image_path) # The base directory for calculating relative paths.
    # Check if the path is a directory.
    elif os.path.isdir(args.image_path):
        # Use the helper function to find all TIFF files inside the directory.
        tiff_files = get_tiff_files(args.image_path, recursive=args.recursive)
        base_dir = args.image_path
    # If it's neither a file nor a directory, raise an error.
    else:
        raise ValueError(f"Invalid image_path: {args.image_path}")

    # If no TIFF files were found, raise an error.
    if not tiff_files:
        raise ValueError(f"No GeoTIFF files found in {args.image_path}")

    # --- Process Each File ---
    # Loop through every TIFF file that was found.
    for tiff_file in tiff_files:
        # Generate the full path for the output mask file.
        output_path = generate_output_path(tiff_file, base_dir, args.output_dir, 
                                         suffix="_predicted_mask", preserve_structure=args.preserve_structure)
        # Generate the full path for the probability map file.
        prob_output_path = generate_output_path(tiff_file, base_dir, args.prob_output_dir, 
                                              suffix="_probability", preserve_structure=args.preserve_structure)
        
        print(f"Processing {tiff_file}...")
        # Call the main prediction function for the current file.
        predict_on_new_image(args.model_path, tiff_file, output_path, prob_output_path,
                             tile_size=args.tile_size,
                             overlap=args.overlap,
                             visualize=args.visualize)


# This is a standard Python construct. The code inside this block will only run
# when the script is executed directly (e.g., `python unet_wildfire_predict.py`),
# not when it's imported as a module into another script.
if __name__ == "__main__":
    main()