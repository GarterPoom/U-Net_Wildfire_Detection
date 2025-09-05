# unet_wildfire_predict.py
# Standalone script to use a trained U-Net model for predicting wildfire burn masks on new GeoTIFF images, ensuring binary (0,1) output compatible with QGIS visualization.

import os  # Import os module for operating system interactions, such as file handling and environment variable management.
os.environ["OMP_NUM_THREADS"] = "1"  # Set OpenMP to use a single thread to avoid threading conflicts in numerical libraries.
os.environ["MKL_NUM_THREADS"] = "1"  # Set Intel MKL to use a single thread to prevent performance issues with parallel processing.
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Set OpenBLAS to use a single thread for consistent behavior in linear algebra operations.
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # Set VECLIB to use a single thread for compatibility with macOS numerical libraries.
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # Set NumExpr to use a single thread to avoid conflicts in numerical expression evaluation.

import torch  # Import PyTorch library for deep learning operations, including model creation and tensor handling.
import torch.nn as nn  # Import PyTorch neural network module for defining layers and architectures.
import rasterio  # Import rasterio library for reading and writing GeoTIFF raster files.
from rasterio.windows import Window  # Import Window class from rasterio for handling specific regions of raster data.
import numpy as np  # Import NumPy library for efficient numerical array operations.
from skimage.transform import resize  # Import resize function from scikit-image for resizing image data.
from tqdm import tqdm  # Import tqdm for displaying progress bars during iterative processes.
import matplotlib.pyplot as plt  # Import matplotlib.pyplot for visualizing prediction results.
import argparse  # Import argparse for parsing command-line arguments.
import glob  # Import glob for finding files matching specific patterns, such as GeoTIFFs.

# DoubleConv Module: Defines a block with two convolutional layers, each followed by batch normalization and ReLU activation, used in the U-Net architecture.
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):  # Initialize the DoubleConv module with input and output channel counts.
        super().__init__()  # Call the parent class (nn.Module) constructor to initialize the module.
        self.double_conv = nn.Sequential(  # Define a sequential container for the double convolution block.
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # First 3x3 convolution layer to extract features.
            nn.BatchNorm2d(out_channels),  # Batch normalization to stabilize and accelerate training.
            nn.ReLU(inplace=True),  # ReLU activation to introduce non-linearity, applied in-place to save memory.
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # Second 3x3 convolution layer for further feature extraction.
            nn.BatchNorm2d(out_channels),  # Second batch normalization for the output channels.
            nn.ReLU(inplace=True)  # Second ReLU activation to enhance non-linearity.
        )

    def forward(self, x):  # Define the forward pass for the DoubleConv module.
        return self.double_conv(x)  # Pass the input through the sequential double convolution block.

# Down Module: Performs max pooling followed by a DoubleConv block for downsampling in the U-Net encoder path.
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):  # Initialize the Down module with input and output channel counts.
        super().__init__()  # Call the parent class (nn.Module) constructor.
        self.maxpool_conv = nn.Sequential(  # Define a sequential container for max pooling and double convolution.
            nn.MaxPool2d(2),  # 2x2 max pooling to reduce spatial dimensions by half.
            DoubleConv(in_channels, out_channels)  # DoubleConv block to process features after pooling.
        )

    def forward(self, x):  # Define the forward pass for the Down module.
        return self.maxpool_conv(x)  # Pass the input through max pooling and double convolution.

# Up Module: Performs upsampling, concatenates with skip connections, and applies a DoubleConv block for the U-Net decoder path.
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):  # Initialize the Up module with input and output channel counts.
        super().__init__()  # Call the parent class (nn.Module) constructor.
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)  # Transposed convolution for upsampling.
        self.conv = DoubleConv(in_channels, out_channels)  # DoubleConv block to process concatenated features.

    def forward(self, x1, x2):  # Define the forward pass, taking the upsampled input (x1) and skip connection (x2).
        x1 = self.up(x1)  # Upsample the input using transposed convolution.
        diffY = x2.size()[2] - x1.size()[2]  # Calculate height difference between skip connection and upsampled input.
        diffX = x2.size()[3] - x1.size()[3]  # Calculate width difference between skip connection and upsampled input.
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])  # Pad upsampled input to match skip connection size.
        x = torch.cat([x2, x1], dim=1)  # Concatenate skip connection and upsampled input along the channel dimension.
        return self.conv(x)  # Pass the concatenated tensor through the DoubleConv block.

# OutConv Module: Applies a final 1x1 convolution to produce the desired number of output channels.
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):  # Initialize the OutConv module with input and output channel counts.
        super().__init__()  # Call the parent class (nn.Module) constructor.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 1x1 convolution to map to output channels.

    def forward(self, x):  # Define the forward pass for the OutConv module.
        return self.conv(x)  # Pass the input through the 1x1 convolution layer.

# UNet Model: Implements the full U-Net architecture for image segmentation, combining encoder and decoder paths with skip connections.
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):  # Initialize UNet with input channels and number of output classes.
        super(UNet, self).__init__()  # Call the parent class (nn.Module) constructor.
        self.n_channels = n_channels  # Store the number of input channels.
        self.n_classes = n_classes  # Store the number of output classes.
        self.inc = DoubleConv(n_channels, 64)  # Initial DoubleConv block for input processing.
        self.down1 = Down(64, 128)  # First downsampling block.
        self.down2 = Down(128, 256)  # Second downsampling block.
        self.down3 = Down(256, 512)  # Third downsampling block.
        self.down4 = Down(512, 1024)  # Fourth downsampling block (bottleneck).
        self.up1 = Up(1024, 512)  # First upsampling block.
        self.up2 = Up(512, 256)  # Second upsampling block.
        self.up3 = Up(256, 128)  # Third upsampling block.
        self.up4 = Up(128, 64)  # Fourth upsampling block.
        self.outc = OutConv(64, n_classes)  # Final convolution to produce output.

    def forward(self, x):  # Define the forward pass for the UNet model.
        x1 = self.inc(x)  # Process input through initial DoubleConv block.
        x2 = self.down1(x1)  # First downsampling step.
        x3 = self.down2(x2)  # Second downsampling step.
        x4 = self.down3(x3)  # Third downsampling step.
        x5 = self.down4(x4)  # Fourth downsampling step (bottleneck).
        x = self.up1(x5, x4)  # First upsampling with skip connection from x4.
        x = self.up2(x, x3)  # Second upsampling with skip connection from x3.
        x = self.up3(x, x2)  # Third upsampling with skip connection from x2.
        x = self.up4(x, x1)  # Fourth upsampling with skip connection from x1.
        logits = self.outc(x)  # Final convolution to produce output logits.
        return logits  # Return the output logits.

# get_tiff_files: Retrieves all GeoTIFF files (.tif or .tiff) from a specified path, supporting recursive directory search.
def get_tiff_files(input_path, recursive=True):  # Define function to find GeoTIFF files with optional recursive search.
    """Retrieve all .tif/.tiff files from input_path."""
    pattern = '**/*.tif' if recursive else '*.tif'  # Set file pattern to include subdirectories if recursive is True.
    files = glob.glob(os.path.join(input_path, pattern), recursive=recursive)  # Find all .tif files matching the pattern.
    files += glob.glob(os.path.join(input_path, pattern.replace('.tif', '.tiff')), recursive=recursive)  # Add .tiff files.
    files = sorted([f for f in files if os.path.isfile(f)])  # Sort files and ensure they are valid files.
    if not files:  # Check if no files were found.
        print(f"No GeoTIFF files found in {input_path} (recursive={recursive})")  # Print message if no files are found.
    else:  # If files are found, print details.
        print(f"Found {len(files)} GeoTIFF files in {input_path}:")  # Print the number of files found.
        for f in files:  # Iterate over found files.
            print(f"  - {f}")  # Print each file path.
    return files  # Return the sorted list of GeoTIFF files.

# generate_output_path: Generates an output file path for the predicted mask based on the input filename and directory structure.
def generate_output_path(input_path, input_base_dir, output_dir, suffix="_predicted_mask", preserve_structure=False):  # Define function to create output path.
    """Generate output path by appending suffix to the input filename, optionally preserving directory structure."""
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it does not exist.
    base_name = os.path.splitext(os.path.basename(input_path))[0]  # Extract the base filename without extension.
    if preserve_structure:  # Check if directory structure should be preserved.
        rel_path = os.path.relpath(os.path.dirname(input_path), input_base_dir)  # Get relative path from input base directory.
        output_subdir = os.path.join(output_dir, rel_path)  # Create output subdirectory path.
        os.makedirs(output_subdir, exist_ok=True)  # Create output subdirectory if it does not exist.
        output_filename = f"{base_name}{suffix}.tif"  # Create output filename with suffix.
        return os.path.join(output_subdir, output_filename)  # Return full output path with preserved structure.
    else:  # If not preserving structure, save directly in output directory.
        output_filename = f"{base_name}{suffix}.tif"  # Create output filename with suffix.
        return os.path.join(output_dir, output_filename)  # Return output path in the output directory.

# predict_on_new_image: Predicts a binary wildfire burn mask on a new GeoTIFF image using tiling and saves the result.
def predict_on_new_image(model_path, new_image_path, output_path, tile_size=256, overlap=32, device=None, visualize=False):
    """
    Predicts a binary wildfire burn mask on a new GeoTIFF image using a trained U-Net model with tiling and overlap.
    Args:
        model_path (str): Path to the trained U-Net model file.
        new_image_path (str): Path to the input GeoTIFF image.
        output_path (str): Path to save the predicted binary mask.
        tile_size (int): Size of the tiles for processing (default: 256).
        overlap (int): Pixel overlap between tiles to avoid edge artifacts (default: 32).
        device (torch.device): Device to run the model on (default: CUDA if available, else CPU).
        visualize (bool): Whether to display the predicted mask using matplotlib (default: False).
    """
    if device is None:  # Check if device is not specified.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device to CUDA if available, else CPU.
    print(f"Using device: {device}")  # Print the selected device for model inference.

    # Load the U-Net model
    try:  # Attempt to load the model and image.
        with rasterio.open(new_image_path) as src:  # Open the input GeoTIFF image.
            num_channels = src.count  # Get the number of channels in the image.
        model = UNet(n_channels=num_channels, n_classes=1).to(device)  # Initialize UNet model with input channels and 1 output class.
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))  # Load pre-trained model weights.
        model.eval()  # Set model to evaluation mode (disables dropout and batch norm updates).
        print(f"Model loaded successfully for {new_image_path}")  # Confirm successful model loading.
    except FileNotFoundError:  # Handle case where model file is not found.
        raise FileNotFoundError(f"Model file not found: {model_path}")  # Raise error with model path.
    except Exception as e:  # Handle other potential errors during model loading.
        raise RuntimeError(f"Error loading model for {new_image_path}: {str(e)}")  # Raise error with details.

    with rasterio.open(new_image_path) as src:  # Open the input GeoTIFF image for processing.
        height, width = src.shape  # Get the height and width of the image.
        meta = src.meta.copy()  # Copy the image metadata for output GeoTIFF.
        bands = list(range(1, src.count + 1))  # Create a list of band indices (1-based for rasterio).

        # Initialize full mask array
        full_mask = np.zeros((height, width), dtype=np.uint8)  # Create an empty binary mask array with the same dimensions as the input image.

        # Calculate tiling windows
        windows = []  # Initialize list to store window information for tiling.
        step = tile_size - 2 * overlap  # Calculate step size for tiles, accounting for overlap.
        for i in range(0, height, step):  # Iterate over image height with step size.
            for j in range(0, width, step):  # Iterate over image width with step size.
                row_start = max(0, i - overlap)  # Define row start with overlap, ensuring non-negative.
                col_start = max(0, j - overlap)  # Define column start with overlap, ensuring non-negative.
                row_end = min(height, i + tile_size + overlap)  # Define row end, limited by image height.
                col_end = min(width, j + tile_size + overlap)  # Define column end, limited by image width.

                win_h = row_end - row_start  # Calculate window height.
                win_w = col_end - col_start  # Calculate window width.

                if win_h <= 0 or win_w <= 0:  # Skip invalid windows with zero or negative size.
                    continue

                read_window = Window(col_start, row_start, win_w, win_h)  # Define rasterio read window for the tile.
                write_row_start = i  # Define row start for writing to full mask.
                write_col_start = j  # Define column start for writing to full mask.
                write_row_end = min(height, i + step)  # Define row end for writing, limited by image height.
                write_col_end = min(width, j + step)  # Define column end for writing, limited by image width.

                write_window = Window(write_col_start, write_row_start, write_col_end - write_col_start, write_row_end - write_row_start)  # Define write window.

                windows.append({'read': read_window, 'write': write_window})  # Store read and write window information.

    if not windows:  # Check if no valid windows were generated.
        print(f"No valid tiles found for {new_image_path}.")  # Print message if no tiles are available.
        return  # Exit function if no tiles to process.

    # Process tiles for prediction
    with torch.no_grad():  # Disable gradient computation for inference to save memory and speed up processing.
        for tile_info in tqdm(windows, desc=f"Predicting tiles for {os.path.basename(new_image_path)}"):  # Iterate over tiles with progress bar.
            read_window = tile_info['read']  # Get the read window for the current tile.
            write_window = tile_info['write']  # Get the write window for the current tile.

            with rasterio.open(new_image_path) as src:  # Open the input GeoTIFF to read the tile.
                tile = src.read(bands, window=read_window)  # Read the tile data for specified bands.

            # Pad tile if smaller than required size
            pad_h = (tile_size + 2 * overlap) - tile.shape[1]  # Calculate padding needed for height.
            pad_w = (tile_size + 2 * overlap) - tile.shape[2]  # Calculate padding needed for width.
            if pad_h > 0 or pad_w > 0:  # Check if padding is needed.
                tile = np.pad(tile, ((0, 0), (0, max(0, pad_h)), (0, max(0, pad_w))), mode="constant", constant_values=0)  # Pad tile with zeros.

            # Normalize channels
            tile = tile.astype(np.float32)  # Convert tile data to float32 for processing.
            for c in range(tile.shape[0]):  # Iterate over each channel.
                ch = tile[c]  # Get the current channel data.
                min_val, max_val = ch.min(), ch.max()  # Compute min and max values for normalization.
                if max_val - min_val > 1e-6:  # Check if channel has non-zero range to avoid division by zero.
                    tile[c] = (ch - min_val) / (max_val - min_val)  # Normalize channel to [0, 1].
                else:  # If range is near zero, set channel to zero.
                    tile[c] = 0  # Set channel to zero for uniform data.

            # Extract central region of the tile
            central_tile = tile[:, overlap:overlap + tile_size, overlap:overlap + tile_size]  # Crop to central tile_size x tile_size region.

            # Perform prediction
            image_tensor = torch.from_numpy(central_tile).unsqueeze(0).to(device)  # Convert tile to PyTorch tensor and add batch dimension.
            output = model(image_tensor)  # Run the tile through the U-Net model.
            pred = (torch.sigmoid(output) > 0.6).cpu().numpy().squeeze().astype(np.uint8)  # Apply sigmoid and threshold at 0.6 for binary output.

            # Write prediction to full mask
            full_mask[write_window.row_off:write_window.row_off + write_window.height, write_window.col_off:write_window.col_off + write_window.width] = pred[:write_window.height, :write_window.width]  # Copy prediction to corresponding region in full mask.

    # Verify binary mask
    unique_values = np.unique(full_mask)  # Get unique values in the predicted mask.
    if not np.array_equal(unique_values, [0, 1]) and len(unique_values) > 0:  # Check if mask contains only 0 and 1.
        raise ValueError(f"Mask contains non-binary values: {unique_values}")  # Raise error if non-binary values are found.
    print(f"Generated mask unique values: {unique_values}")  # Print unique values in the mask.

    # Optional visualization of the predicted mask
    if visualize:  # Check if visualization is enabled.
        plt.figure(figsize=(10, 10))  # Create a new figure with size 10x10 inches.
        plt.imshow(full_mask, cmap='gray', vmin=0, vmax=1)  # Display the mask as a grayscale image.
        plt.title(f'Predicted Burn Mask for {os.path.basename(new_image_path)}')  # Set the title of the plot.
        plt.colorbar(ticks=[0, 1])  # Add a colorbar with ticks at 0 and 1.
        plt.show()  # Display the plot.

    # Save the predicted mask as a GeoTIFF
    try:  # Attempt to save the output GeoTIFF.
        meta.update(count=1, dtype='uint8', nodata=255, compress='lzw')  # Update metadata for single-band uint8 output with LZW compression.
        with rasterio.open(output_path, 'w', **meta) as dst:  # Open a new GeoTIFF file for writing.
            dst.write(full_mask, 1)  # Write the predicted mask to the first band.
    except Exception as e:  # Handle errors during saving.
        raise RuntimeError(f"Error saving output GeoTIFF {output_path}: {str(e)}")  # Raise error with details.

    # Verify the saved GeoTIFF
    with rasterio.open(output_path) as dst:  # Open the saved GeoTIFF.
        output_data = dst.read(1)  # Read the first band of the output GeoTIFF.
        unique_output_values = np.unique(output_data)  # Get unique values in the saved GeoTIFF.
        if not np.array_equal(unique_output_values, [0, 1]) and len(unique_output_values) > 0:  # Check if output is binary.
            raise ValueError(f"Output GeoTIFF contains non-binary values: {unique_output_values}")  # Raise error if non-binary values are found.
        print(f"Output GeoTIFF unique values: {unique_output_values}")  # Print unique values in the output GeoTIFF.

    print(f"✅ Prediction complete. Output saved to {output_path}")  # Confirm successful prediction and saving.
    print("To view in QGIS: Load raster → Style → Singleband pseudocolor → 0=Unburned, 1=Burned")  # Provide QGIS visualization instructions.

# Main function: Orchestrates the processing of multiple GeoTIFFs using command-line arguments.
if __name__ == "__main__":  # Check if the script is run directly (not imported as a module).
    parser = argparse.ArgumentParser(description="Predict wildfire burn mask using U-Net model")  # Initialize argument parser with description.
    parser.add_argument('--model_path', default="Export_Model\\unet_wildfire.pth", help="Path to trained model")  # Add argument for model path.
    parser.add_argument('--image_path', default="Raster_Classified", help="Path to input GeoTIFF or directory of GeoTIFFs")  # Add argument for input image path.
    parser.add_argument('--output_dir', default="Predicted_Mask", help="Directory to save output GeoTIFFs")  # Add argument for output directory.
    parser.add_argument('--tile_size', type=int, default=256, help="Tile size for processing")  # Add argument for tile size.
    parser.add_argument('--visualize', action='store_true', help="Visualize the predicted mask")  # Add flag for enabling visualization.
    parser.add_argument('--recursive', action='store_true', default=True, help="Search for GeoTIFFs in subdirectories")  # Add flag for recursive search.
    parser.add_argument('--preserve_structure', action='store_true', help="Preserve input directory structure in output")  # Add flag for preserving directory structure.
    args = parser.parse_args()  # Parse command-line arguments.

    # Normalize file paths for Windows compatibility
    args.image_path = os.path.normpath(args.image_path)  # Normalize input image path to handle Windows path separators.
    args.output_dir = os.path.normpath(args.output_dir)  # Normalize output directory path.
    args.model_path = os.path.normpath(args.model_path)  # Normalize model path.

    # Determine if input is a single file or directory
    if os.path.isfile(args.image_path):  # Check if input path is a single file.
        tiff_files = [args.image_path]  # Set single file as the list of files to process.
        output_dir = args.output_dir  # Use specified output directory.
    elif os.path.isdir(args.image_path):  # Check if input path is a directory.
        tiff_files = get_tiff_files(args.image_path, recursive=args.recursive)  # Get list of GeoTIFF files from directory.
        output_dir = args.output_dir  # Use specified output directory.
    else:  # Handle invalid input path.
        raise ValueError(f"Invalid image_path: {args.image_path}. Must be a file or directory.")  # Raise error for invalid path.

    if not tiff_files:  # Check if no GeoTIFF files were found.
        raise ValueError(f"No GeoTIFF files found in {args.image_path}. Ensure the directory contains .tif/.tiff files, and check if --recursive is needed.")  # Raise error if no files.

    # Process each GeoTIFF file
    for tiff_file in tiff_files:  # Iterate over each GeoTIFF file.
        output_path = generate_output_path(tiff_file, args.image_path, args.output_dir, preserve_structure=args.preserve_structure)  # Generate output path for the file.
        print(f"Processing {tiff_file}...")  # Print message indicating current file being processed.
        predict_on_new_image(args.model_path, tiff_file, output_path, tile_size=args.tile_size, visualize=args.visualize)  # Run prediction on the file.