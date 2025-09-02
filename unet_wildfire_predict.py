# prediction.py
# Standalone script to use the trained UNet model for prediction on a new GeoTIFF, ensuring binary (0,1) output for QGIS

import torch
import torch.nn as nn
import rasterio
from rasterio.windows import Window
import numpy as np
from skimage.transform import resize
from tqdm import tqdm
import matplotlib.pyplot as plt  # For optional visualization

# DoubleConv Module: Two consecutive convolutional layers with batch normalization and ReLU activation
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# Down Module: Max pooling followed by a DoubleConv block for downsampling
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# Up Module: Upsampling followed by concatenation with skip connection and DoubleConv
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# OutConv Module: Final 1x1 convolution to produce output channels
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# UNet Model: Full U-Net architecture for image segmentation
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# Function to predict on a new GeoTIFF and save the mask with only 0 and 1 values
def predict_on_new_image(model_path, new_image_path, output_path, tile_size=256, device=None, visualize=False):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    with rasterio.open(new_image_path) as src:
        num_channels = src.count  # Assume same number of bands as training
    model = UNet(n_channels=num_channels, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully.")

    # Open the new GeoTIFF to get metadata and generate windows
    with rasterio.open(new_image_path) as src:
        height, width = src.shape
        meta = src.meta  # Metadata for output (transform, crs, etc.)
        bands = list(range(1, src.count + 1))  # All bands
        windows = []
        for i in range(0, height, tile_size):
            for j in range(0, width, tile_size):
                w = min(tile_size, width - j)
                h = min(tile_size, height - i)
                if w == tile_size and h == tile_size:  # Only full tiles
                    windows.append(Window(j, i, w, h))

    if len(windows) == 0:
        raise ValueError("No full tiles available. Image may be smaller than tile size or not a multiple of it.")

    # Prepare full mask array (single band, same height/width as input)
    full_mask = np.zeros((height, width), dtype=np.uint8)

    # Process each tile
    with torch.no_grad():
        for window in tqdm(windows, desc="Predicting tiles"):
            with rasterio.open(new_image_path) as src:
                image = src.read(bands, window=window)  # Read tile (C, H, W)
                height_win, width_win = image.shape[1], image.shape[2]

            # Normalize and resize (like in dataset)
            image = resize(image.transpose(1, 2, 0), (tile_size, tile_size, num_channels), mode='reflect', anti_aliasing=True)
            image = image.transpose(2, 0, 1).astype(np.float32)
            for c in range(image.shape[0]):
                channel = image[c]
                min_val = channel.min()
                max_val = channel.max()
                if max_val - min_val > 1e-6:
                    image[c] = (channel - min_val) / (max_val - min_val)
                else:
                    image[c] = 0

            # Convert to tensor and predict
            image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)  # (1, C, H, W)
            output = model(image_tensor)
            pred = torch.sigmoid(output) > 0.5  # Binary mask (True/False)
            pred = pred.cpu().numpy().squeeze(0).squeeze(0).astype(np.uint8)  # (H, W), ensure uint8 for 0,1
            pred = resize(pred, (height_win, width_win), order=0, anti_aliasing=False).astype(np.uint8)  # Resize, keep binary

            # Ensure binary values (0 or 1)
            pred = np.where(pred > 0, 1, 0).astype(np.uint8)

            # Write to full mask
            full_mask[window.row_off:window.row_off + window.height, window.col_off:window.col_off + window.width] = pred

    # Verify that full_mask contains only 0 and 1
    unique_values = np.unique(full_mask)
    if not np.array_equal(unique_values, [0, 1]) and len(unique_values) > 0:
        raise ValueError(f"Mask contains non-binary values: {unique_values}")
    print(f"Generated mask unique values: {unique_values}")

    # Optional visualization
    if visualize:
        plt.figure(figsize=(10, 10))
        plt.imshow(full_mask, cmap='gray', vmin=0, vmax=1)
        plt.title('Predicted Burn Mask (0: Unburned, 1: Burned)')
        plt.colorbar(ticks=[0, 1])
        plt.show()

    # Save the mask as GeoTIFF (update metadata for single band)
    meta.update(count=1, dtype='uint8', nodata=0, compress='lzw')  # Added LZW compression for smaller file
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(full_mask, 1)  # Write single band

    # Verify output GeoTIFF
    with rasterio.open(output_path) as dst:
        output_data = dst.read(1)
        unique_output_values = np.unique(output_data)
        if not np.array_equal(unique_output_values, [0, 1]) and len(unique_output_values) > 0:
            raise ValueError(f"Output GeoTIFF contains non-binary values: {unique_output_values}")
        print(f"Output GeoTIFF unique values: {unique_output_values}")

    print(f"Prediction complete. Output saved to {output_path}")
    print("To view in QGIS: Load the raster and set the style to 'Singleband pseudocolor' with a binary palette (0=Unburned, 1=Burned).")

# Example usage
if __name__ == "__main__":
    model_path = "Export_Model\\unet_model.pth"  # Path to saved model
    new_image_path = r"path\to\your\new\T47QLA_new_date.tif"  # Replace with your new GeoTIFF path
    output_path = r"path\to\output\predicted_mask.tif"  # Desired output path
    predict_on_new_image(model_path, new_image_path, output_path, visualize=True)