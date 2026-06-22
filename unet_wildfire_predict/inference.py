"""Tile-and-stitch U-Net inference on GeoTIFF rasters."""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import rasterio
import torch
from skimage.transform import resize
from tqdm import tqdm

from unet_wildfire_predict.config import PredictionConfig
from unet_wildfire_predict.visualization import visualize_prediction
from unet_wildfire_training import UNet
from unet_wildfire_training.data import compute_sentinel2_indices, percentile_normalize

from pathlib import Path
from unet_wildfire_predict.config import PredictionConfig
from unet_wildfire_predict.paths import get_tiff_files 

def run_prediction(config: PredictionConfig) -> None:
    """
    Wrapper function that iterates through all GeoTIFFs in the input directory
    provided by the config and runs inference on each.
    """
    image_dir = Path(config.image_path)
    
    # Find all TIFF files in the input directory
    input_files = get_tiff_files(image_dir, recursive=True)
    
    if not input_files:
        print(f"⚠️ No input images found in {image_dir}")
        return

    print(f"Found {len(input_files)} images. Starting batch inference...")

    for img_path_str in input_files:
        img_path = Path(img_path_str)
        
        # 1. Determine the output paths
        # We follow the naming convention: 
        # Probability: [filename]_probability.tif
        # Mask: [filename]_predicted_mask.tif
        rel_path = img_path.relative_to(image_dir.parent if config.preserve_structure else image_dir)
        
        prob_output = Path(config.prob_output_dir) / rel_path.with_name(f"{img_path.stem}_probability.tif")
        mask_output = Path(config.output_dir) / rel_path.with_name(f"{img_path.stem}_predicted_mask.tif")

        # 2. Ensure destination directories exist
        prob_output.parent.mkdir(parents=True, exist_ok=True)
        mask_output.parent.mkdir(parents=True, exist_ok=True)

        print(f"Processing: {img_path.name}")
        
        # 3. Call the single-image inference function
        try:
            predict_on_new_image(
                model_path=str(config.model_path),
                new_image_path=str(img_path),
                output_path=str(mask_output),
                prob_output_path=str(prob_output),
                config=config,
                device=None # Will default to CUDA/CPU inside the function
            )
        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")

    print("✅ Batch inference complete.")

def _resize_chw(array: np.ndarray, out_hw: Tuple[int, int], anti_aliasing: bool = True) -> np.ndarray:
    """Resize a ``(C, H, W)`` array to ``(C, out_h, out_w)`` using skimage."""
    out_h, out_w = out_hw
    return resize(
        array.transpose(1, 2, 0),
        (out_h, out_w, array.shape[0]),
        mode="reflect",
        anti_aliasing=anti_aliasing,
    ).transpose(2, 0, 1).astype(np.float32)

def predict_on_new_image(
    model_path: str,
    new_image_path: str,
    output_path: str,
    prob_output_path: str,
    config: PredictionConfig,
    device: Optional[torch.device] = None,
    band_layout: Optional[Dict[str, int]] = None,
) -> None:
    """Tile-and-stitch inference on a single GeoTIFF.

    Each tile is read at ``config.tile_size``, resized to ``config.target_size``
    for the network, predicted, then upsampled back so probabilities can be
    averaged in the raster's native resolution. When ``band_layout`` is given
    (defaulting to ``config.band_layout``), the four Sentinel-2 indices (NDVI,
    NDWI, SAVI, BAIS2) are appended to the input stack, matching the
    training-time behavior of ``build_dataloaders``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if band_layout is None:
        band_layout = config.band_layout
    reflectance_scale = config.reflectance_scale

    with rasterio.open(new_image_path) as src:
        height, width = src.shape
        meta = src.meta.copy()
        image = src.read().astype(np.float32)
        # Identify cloud-masked areas to preserve them in the output
        nan_mask = np.isnan(image).any(axis=0)

    if band_layout is not None:
        indices = compute_sentinel2_indices(image, band_layout, reflectance_scale)
        image = np.concatenate([image, indices], axis=0)
    model_channels = image.shape[0]

    model = UNet(n_channels=model_channels, n_classes=1).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    norm_low, norm_high = config.normalization_percentiles
    image = percentile_normalize(image, norm_low, norm_high)

    tile_size = config.tile_size
    target_h, target_w = config.target_size
    overlap = config.overlap
    step = tile_size - overlap
    if step <= 0:
        raise ValueError(f"overlap ({overlap}) must be smaller than tile_size ({tile_size})")

    pred_sum = np.zeros((height, width), dtype=np.float32)
    pred_count = np.zeros((height, width), dtype=np.uint16)

    with torch.no_grad():
        for i in tqdm(range(0, height, step), desc="Predicting"):
            for j in range(0, width, step):
                row_end = min(i + tile_size, height)
                col_end = min(j + tile_size, width)
                actual_h = row_end - i
                actual_w = col_end - j
                tile = image[:, i:row_end, j:col_end]

                pad_h = tile_size - actual_h
                pad_w = tile_size - actual_w
                if pad_h > 0 or pad_w > 0:
                    tile = np.pad(
                        tile, ((0, 0), (0, pad_h), (0, pad_w)),
                        mode="constant", constant_values=0,
                    )

                tile_resized = _resize_chw(tile, (target_h, target_w))
                tile_tensor = torch.from_numpy(tile_resized).unsqueeze(0).to(device)
                logits = model(tile_tensor)
                prob = torch.sigmoid(logits).cpu().numpy().squeeze()
                
                prob_full = resize(
                    prob, (tile_size, tile_size),
                    mode="reflect", anti_aliasing=True,
                ).astype(np.float32)
                # Ensure probability stays strictly in [0, 1] after resampling
                prob_full = np.clip(prob_full, 0, 1)

                prob_full = prob_full[:actual_h, :actual_w]
                pred_sum[i:row_end, j:col_end] += prob_full
                pred_count[i:row_end, j:col_end] += 1

    avg_pred = pred_sum / np.maximum(pred_count, 1)
    # Mask out cloud areas in the probability map
    avg_pred[nan_mask] = np.nan

    prob_meta = meta.copy()
    prob_meta.update(count=1, dtype="float32", nodata=np.nan, compress="lzw")
    os.makedirs(os.path.dirname(prob_output_path) or ".", exist_ok=True)
    with rasterio.open(prob_output_path, "w", **prob_meta) as dst:
        dst.write(avg_pred.astype(np.float32), 1)
    print(f"✅ Probability map saved to {prob_output_path}")

    # Create binary mask: set cloud-masked areas to nodata (255)
    full_mask = np.where(nan_mask, 255, (avg_pred > 0.5).astype(np.uint8)).astype(np.uint8)
    meta.update(count=1, dtype="uint8", nodata=255, compress="lzw")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(full_mask, 1)
    print(f"✅ Binary mask saved to {output_path}")
    print("To view in QGIS: Style → Singleband pseudocolor → 0=Unburned, 1=Burned")

    if config.visualize:
        visualize_prediction(avg_pred, full_mask, new_image_path)
