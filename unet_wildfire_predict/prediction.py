"""Prediction orchestration: discover rasters, run inference, write outputs.

This module handles the high‑level workflow of locating GeoTIFF files,
generating output paths, and invoking per‑raster inference.
"""

from __future__ import annotations

import os  # for path normalisation
from typing import Optional  # optional config argument

# Import configuration and helper utilities.
from unet_wildfire_predict.config import PredictionConfig
from unet_wildfire_predict.inference import predict_on_new_image
from unet_wildfire_predict.paths import generate_output_path, get_tiff_files


def run_prediction(config: Optional[PredictionConfig] = None) -> None:
    """Discover GeoTIFFs under ``config.image_path`` and run U‑Net inference on each.

    Mirrors ``unet_wildfire_training.training.train_model`` in style: callers
    pass a :class:`PredictionConfig`, the function takes care of file discovery,
    output path generation, and per‑raster inference.
    """
    # If no config is supplied, create a default one using default paths.
    if config is None:
        config = PredictionConfig()

    # Normalise all path strings to OS‑specific format for consistency.
    image_path = os.path.normpath(str(config.image_path))
    output_dir = os.path.normpath(str(config.output_dir))
    prob_output_dir = os.path.normpath(str(config.prob_output_dir))
    model_path = os.path.normpath(str(config.model_path))

    # Determine whether the input is a single file or a directory.
    if os.path.isfile(image_path):
        tiff_files = [image_path]               # single raster case
        base_dir = os.path.dirname(image_path)  # directory containing the file
    elif os.path.isdir(image_path):
        # Recursively collect all GeoTIFF files from the directory tree.
        tiff_files = get_tiff_files(image_path, recursive=config.recursive)
        base_dir = image_path                     # root of the input set
    else:
        raise ValueError(f"Invalid image_path: {image_path}")

    # Ensure at least one TIFF file was found; otherwise raise a clear error.
    if not tiff_files:
        raise ValueError(f"No GeoTIFF files found in {image_path}")

    # Iterate over each raster and run inference, writing separate mask and probability outputs.
    for tiff_file in tiff_files:
        # Compute output file names while optionally preserving input directory structure.
        output_path = generate_output_path(
            tiff_file,
            base_dir,
            output_dir,
            suffix="_predicted_mask",
            preserve_structure=config.preserve_structure,
        )
        prob_output_path = generate_output_path(
            tiff_file,
            base_dir,
            prob_output_dir,
            suffix="_probability",
            preserve_structure=config.preserve_structure,
        )

        # Inform the user which file is being processed.
        print(f"Processing {tiff_file}...")

        # Execute inference for this raster; all heavy lifting occurs inside
        # ``predict_on_new_image`` (tile‑and‑stitch, model loading, etc.).
        try:
            predict_on_new_image(
                model_path=str(config.model_path),
                new_image_path=str(tiff_file),
                output_path=str(output_path),
                prob_output_path=str(prob_output_path),
                config=config,
                device=None  # let the called function decide CPU/GPU
            )
        except Exception as e:
            # Report any error but continue with remaining rasters.
            print(f"❌ Error processing {tiff_file.name}: {e}")

    # Signal completion of the batch process.
    print("✅ Batch inference complete.")
