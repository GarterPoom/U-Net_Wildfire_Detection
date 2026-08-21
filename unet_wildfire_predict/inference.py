"""Tile‑and‑stitch U‑Net inference on GeoTIFF rasters.

This module implements a robust inference pipeline that:
* reads each raster tile,
* optionally appends derived Sentinel‑2 indices (NDVI, NDWI, SAVI, BAIS2) to the input stack,
* normalises pixel values using percentiles matching training,
* runs U‑Net prediction on resized tiles,
* aggregates predictions via weighted averaging,
* writes both probability maps and binary masks,
* optionally visualises results with Matplotlib.
"""

from __future__ import annotations

import os  # for directory creation and path handling
from typing import Dict, Optional, Tuple  # tuple for target_size, dict for band layout

import numpy as np  # numerical operations on raster arrays
import rasterio  # read/write geospatial raster data
import torch  # deep learning framework
from skimage.transform import resize  # resample arrays to different sizes
from tqdm import tqdm  # progress bar for long loops

# Local imports from the package.
from unet_wildfire_predict.config import PredictionConfig
from unet_wildfire_predict.inference import predict_on_new_image  # re‑imported for clarity; actual usage below
from unet_wildfire_predict.visualization import visualize_prediction # visualization
from unet_wildfire_training import UNet  # model definition
from unet_wildfire_training.data import compute_sentinel2_indices, percentile_normalize

# Helper utilities.
from pathlib import Path
from unet_wildfire_predict.paths import get_tiff_files  # discover files


def run_prediction(config: PredictionConfig) -> None:
    """Wrapper that iterates over all GeoTIFFs in ``config.image_path`` and runs inference on each.

    This function is the entry point for batch processing; it discovers input
    rasters, creates output directories if needed, and calls :func:`predict_on_new_image`
    for every file.
    """
    # Convert the image directory to a pathlib Path object for convenient operations.
    image_dir = Path(config.image_path)

    # Find all GeoTIFF files (including sub‑directories) under ``image_dir``.
    input_files = get_tiff_files(image_dir, recursive=True)

    # If no images are found, notify the user and exit early.
    if not input_files:
        print(f"⚠️ No input images found in {image_dir}")
        return

    # Report how many images will be processed.
    print(f"Found {len(input_files)} images. Starting batch inference...")

    # Process each raster individually.
    for img_path_str in input_files:
        img_path = Path(img_path_str)  # ensure pathlib Path for later operations

        # Determine output file names while respecting the ``preserve_structure`` flag.
        rel_path = img_path.relative_to(
            image_dir.parent if config.preserve_structure else image_dir
        )
        prob_output = Path(config.prob_output_dir) / rel_path.with_name(f"{img_path.stem}_probability.tif")
        mask_output = Path(config.output_dir) / rel_path.with_name(f"{img_path.stem}_predicted_mask.tif")

        # Ensure target directories exist before writing.
        prob_output.parent.mkdir(parents=True, exist_ok=True)
        mask_output.parent.mkdir(parents=True, exist_ok=True)

        print(f"Processing: {img_path.name}")

        # Call the core inference routine; any errors are caught and reported locally.
        try:
            predict_on_new_image(
                model_path=str(config.model_path),
                new_image_path=str(img_path),
                output_path=str(mask_output),
                prob_output_path=str(prob_output),
                config=config,
                device=None  # let the called function choose device automatically
            )
        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")

    # Indicate successful completion of batch inference.
    print("✅ Batch inference complete.")


def _resize_chw(array: np.ndarray, out_hw: Tuple[int, int], anti_aliasing: bool = True) -> np.ndarray:
    """Resize a ``(C, H, W)`` array to ``(C, out_h, out_w)`` using :func:`skimage.transform.resize`.

    The function transposes the array to (H, W, C), resizes spatially, then
    transposes back to the original channel order.

    Args:
        array: Input NumPy array with shape ``(C, H, W)``.
        out_hw: Desired output height and width as a tuple ``(out_h, out_w)``.
        anti_aliasing: Whether to apply antialiasing during resampling (default True).

    Returns:
        Resized array with shape ``(C, out_h, out_w)`` and dtype ``float32``.
    """
    out_h, out_w = out_hw
    # Transpose to HWC for skimage, resize, then transpose back to CHW.
    return (
        resize(
            array.transpose(1, 2, 0),          # (H, W, C)
            (out_h, out_w, array.shape[0]),   # (out_h, out_w, C)
            mode="reflect",
            anti_aliasing=anti_aliasing,
        )
        .transpose(2, 0, 1)                  # back to (C, out_h, out_w)
        .astype(np.float32)
    )


def predict_on_new_image(
    model_path: str,
    new_image_path: str,
    output_path: str,
    prob_output_path: str,
    config: PredictionConfig,
    device: Optional[torch.device] = None,
    band_layout: Optional[Dict[str, int]] = None,
) -> None:
    """Perform tile‑and‑stitch inference on a single GeoTIFF raster.

    The function reads the raster, optionally appends derived Sentinel‑2 indices
    (based on ``band_layout``), normalises the data, then processes it in overlapping tiles.
    Predictions are averaged across tiles to produce a seamless probability map,
    which is then thresholded to obtain a binary mask.  Cloud‑masked pixels are left as NaN
    in the probability map and rendered as nodata (255) in the mask.

    Args:
        model_path: Path to the trained U‑Net checkpoint (``.pth`` file).
        new_image_path: Full path to the input GeoTIFF raster.
        output_path: Destination path for the binary mask raster.
        prob_output_path: Destination path for the probability map raster.
        config: Configuration object containing tiling, normalisation and other settings.
        device: Torch device to use; if ``None`` defaults to CUDA when available.
        band_layout: Optional mapping of Sentinel‑2 band names to 0‑indexed positions;
            when supplied indices are computed and appended to the input stack.

    Returns:
        None. Results are written to disk; a Matplotlib figure is shown if ``config.visualize``.
    """
    # Choose device: GPU if available, otherwise CPU.
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use the provided band layout or fall back to the config's default.
    if band_layout is None:
        band_layout = config.band_layout

    # Scale raw reflectance values according to the config (e.g., divide Sentinel‑2 L2A uint16 by 10000).
    reflectance_scale = config.reflectance_scale

    # Open the raster file, read its shape and metadata.
    with rasterio.open(new_image_path) as src:
        height, width = src.shape          # number of bands × spatial dimensions
        meta = src.meta.copy()             # copy metadata for output files
        image = src.read().astype(np.float32)  # read all bands as float32

        # Create a mask indicating cloud‑masked pixels (any NaN in any band).
        nan_mask = np.isnan(image).any(axis=0)

    # If a custom band layout is supplied, compute additional indices and append them.
    if band_layout is not None:
        indices = compute_sentinel2_indices(image, band_layout, reflectance_scale)
        image = np.concatenate([image, indices], axis=0)  # increase channel count

    # Number of channels after optional index addition.
    model_channels = image.shape[0]

    # Load the U‑Net model architecture and instantiate it with the correct number of input channels.
    model = UNet(n_channels=model_channels, n_classes=1).to(device)

    # Load trained weights from disk; map_location ensures loading onto the chosen device.
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()  # disable dropout/batchnorm updates

    # Clip pixel values to the percentiles defined during training for consistent scaling.
    norm_low, norm_high = config.normalization_percentiles
    image = percentile_normalize(image, norm_low, norm_high)

    # Tile configuration: size of each inference window and overlap between adjacent tiles.
    tile_size = config.tile_size
    target_h, target_w = config.target_size  # network input size (must match training)
    overlap = config.overlap
    step = tile_size - overlap

    # Validate that the stride is positive; otherwise tiling would be impossible.
    if step <= 0:
        raise ValueError(f"overlap ({overlap}) must be smaller than tile_size ({tile_size})")

    # Prepare arrays to accumulate weighted predictions and counts for averaging.
    pred_sum = np.zeros((height, width), dtype=np.float32)   # sum of probabilities per pixel
    pred_count = np.zeros((height, width), dtype=np.uint16)  # how many tiles contributed

    # Iterate over the image in steps defined by ``step`` to create overlapping windows.
    with torch.no_grad():  # no gradient computation needed for inference
        for i in tqdm(range(0, height, step), desc="Predicting"):
            for j in range(0, width, step):
                row_end = min(i + tile_size, height)
                col_end = min(j + tile_size, width)
                actual_h = row_end - i          # actual height of this tile
                actual_w = col_end - j          # actual width of this tile

                tile = image[:, i:row_end, j:col_end]  # extract raw tile

                # Pad the tile to the full ``tile_size`` if its dimensions are smaller.
                pad_h = tile_size - actual_h
                pad_w = tile_size - actual_w
                if pad_h > 0 or pad_w > 0:
                    tile = np.pad(
                        tile,
                        ((0, 0), (0, pad_h), (0, pad_w)),
                        mode="constant",
                        constant_values=0,
                    )

                # Resize tile to the network's expected input size while preserving spatial resolution.
                tile_resized = _resize_chw(tile, (target_h, target_w))

                # Convert resized NumPy array to a Torch tensor and move to the selected device.
                tile_tensor = torch.from_numpy(tile_resized).unsqueeze(0).to(device)

                # Forward pass through the model; apply sigmoid to obtain probabilities in [0,1].
                logits = model(tile_tensor)
                prob = torch.sigmoid(logits).cpu().numpy().squeeze()

                # Upsample the probability map back to the original tile size using reflect padding.
                prob_full = resize(
                    prob,
                    (tile_size, tile_size),
                    mode="reflect",
                    anti_aliasing=True,
                ).astype(np.float32)

                # Clip to strict [0,1] range after resampling to avoid numerical errors.
                prob_full = np.clip(prob_full, 0, 1)

                # Extract the portion that corresponds to the actual tile region (undo padding).
                prob_full = prob_full[:actual_h, :actual_w]

                # Accumulate weighted contributions.
                pred_sum[i:row_end, j:col_end] += prob_full
                pred_count[i:row_end, j:col_end] += 1

    # Compute the final averaged probability map; avoid division by zero with max(1, count).
    avg_pred = pred_sum / np.maximum(pred_count, 1)

    # Remove probability values from cloud‑masked regions.
    avg_pred[nan_mask] = np.nan

    # Prepare metadata for the probability GeoTIFF, copying most tags and adjusting datatype/count.
    prob_meta = meta.copy()
    prob_meta.update(count=1, dtype="float32", nodata=np.nan, compress="lzw")
    os.makedirs(os.path.dirname(prob_output_path) or ".", exist_ok=True)
    with rasterio.open(prob_output_path, "w", **prob_meta) as dst:
        dst.write(avg_pred.astype(np.float32), 1)
    print(f"✅ Probability map saved to {prob_output_path}")

    # Build the binary mask: cloud areas become nodata (255); everything else is thresholded at 0.5.
    full_mask = np.where(nan_mask, 255, (avg_pred > 0.5).astype(np.uint8)).astype(np.uint8)

    # Update metadata for the mask output (uint8 type, same nodata value).
    meta.update(count=1, dtype="uint8", nodata=255, compress="lzw")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(full_mask, 1)
    print(f"✅ Binary mask saved to {output_path}")
    print("To view in QGIS: Style → Singleband pseudocolor → 0=Unburned, 1=Burned")

    # Optional visualisation of the prediction using Matplotlib.
    if config.visualize:
        visualize_prediction(avg_pred, full_mask, new_image_path)