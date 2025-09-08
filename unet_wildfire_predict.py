# unet_wildfire_predict.py
# Standalone script to use a trained U-Net model for predicting wildfire burn masks on new GeoTIFF images.

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1" 

import torch
import torch.nn as nn
import rasterio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import glob


# --------------------- U-Net definition ---------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1,
                               [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
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


# --------------------- Helper functions ---------------------
def get_tiff_files(input_path, recursive=True):
    pattern = '**/*.tif' if recursive else '*.tif'
    files = glob.glob(os.path.join(input_path, pattern), recursive=recursive)
    files += glob.glob(os.path.join(input_path, pattern.replace('.tif', '.tiff')), recursive=recursive)
    files = sorted([f for f in files if os.path.isfile(f)])
    return files


def generate_output_path(input_path, input_base_dir, output_dir, suffix="_predicted_mask", preserve_structure=False):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    if preserve_structure:
        rel_path = os.path.relpath(os.path.dirname(input_path), input_base_dir)
        output_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(output_subdir, exist_ok=True)
        return os.path.join(output_subdir, f"{base_name}{suffix}.tif")
    else:
        return os.path.join(output_dir, f"{base_name}{suffix}.tif")


# --------------------- Prediction function (fixed) ---------------------
def predict_on_new_image(
    model_path, new_image_path, output_path,
    tile_size=256, overlap=32, device=None, visualize=False
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    with rasterio.open(new_image_path) as src:
        num_channels = src.count
    model = UNet(n_channels=num_channels, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load and normalize globally
    with rasterio.open(new_image_path) as src:
        height, width = src.shape
        meta = src.meta.copy()
        image = src.read().astype(np.float32)  # (bands, H, W)

    for c in range(image.shape[0]):
        band = image[c]
        min_val, max_val = np.percentile(band, 2), np.percentile(band, 98)
        if max_val - min_val > 1e-6:
            image[c] = np.clip((band - min_val) / (max_val - min_val), 0, 1)
        else:
            image[c] = 0

    # Buffers
    pred_sum = np.zeros((height, width), dtype=np.float32)
    pred_count = np.zeros((height, width), dtype=np.uint16)

    step = tile_size - overlap
    with torch.no_grad():
        for i in tqdm(range(0, height, step), desc="Predicting"):
            for j in range(0, width, step):
                row_end = min(i + tile_size, height)
                col_end = min(j + tile_size, width)

                tile = image[:, i:row_end, j:col_end]

                pad_h = tile_size - tile.shape[1]
                pad_w = tile_size - tile.shape[2]
                if pad_h > 0 or pad_w > 0:
                    tile = np.pad(
                        tile, ((0, 0), (0, pad_h), (0, pad_w)),
                        mode="constant", constant_values=0
                    )

                tile_tensor = torch.from_numpy(tile).unsqueeze(0).to(device)
                output = model(tile_tensor)
                prob = torch.sigmoid(output).cpu().numpy().squeeze()

                prob = prob[: row_end - i, : col_end - j]  # remove padding

                pred_sum[i:row_end, j:col_end] += prob
                pred_count[i:row_end, j:col_end] += 1

    # Average predictions
    avg_pred = pred_sum / np.maximum(pred_count, 1)
    full_mask = (avg_pred >= 0.5).astype(np.uint8)

    # Save
    meta.update(count=1, dtype="uint8", nodata=255, compress="lzw")
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(full_mask, 1)

    print(f"✅ Prediction complete. Output saved to {output_path}")
    print("To view in QGIS: Style → Singleband pseudocolor → 0=Unburned, 1=Burned")

    if visualize:
        plt.figure(figsize=(10, 10))
        plt.imshow(full_mask, cmap="gray", vmin=0, vmax=1)
        plt.title(f"Predicted Burn Mask for {os.path.basename(new_image_path)}")
        plt.colorbar(ticks=[0, 1])
        plt.show()


# --------------------- Main ---------------------
def main():
    parser = argparse.ArgumentParser(description="Predict wildfire burn mask using U-Net model")
    parser.add_argument('--model_path', default="Export_Model/unet_wildfire_no_shape_file.pth", help="Path to trained model")
    parser.add_argument('--image_path', default="Raster_Classified", help="Path to input GeoTIFF or directory of GeoTIFFs")
    parser.add_argument('--output_dir', default="Predicted_Mask", help="Directory to save output GeoTIFFs")
    parser.add_argument('--tile_size', type=int, default=256, help="Tile size for processing")
    parser.add_argument('--overlap', type=int, default=32, help="Overlap between tiles")
    parser.add_argument('--visualize', action='store_true', help="Visualize the predicted mask")
    parser.add_argument('--recursive', action='store_true', default=True, help="Search for GeoTIFFs in subdirectories")
    parser.add_argument('--preserve_structure', action='store_true', help="Preserve input directory structure in output")
    args = parser.parse_args()

    args.image_path = os.path.normpath(args.image_path)
    args.output_dir = os.path.normpath(args.output_dir)
    args.model_path = os.path.normpath(args.model_path)

    if os.path.isfile(args.image_path):
        tiff_files = [args.image_path]
        base_dir = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        tiff_files = get_tiff_files(args.image_path, recursive=args.recursive)
        base_dir = args.image_path
    else:
        raise ValueError(f"Invalid image_path: {args.image_path}")

    if not tiff_files:
        raise ValueError(f"No GeoTIFF files found in {args.image_path}")

    for tiff_file in tiff_files:
        output_path = generate_output_path(tiff_file, base_dir, args.output_dir, preserve_structure=args.preserve_structure)
        print(f"Processing {tiff_file}...")
        predict_on_new_image(args.model_path, tiff_file, output_path,
                             tile_size=args.tile_size,
                             overlap=args.overlap,
                             visualize=args.visualize)


if __name__ == "__main__":
    main()
