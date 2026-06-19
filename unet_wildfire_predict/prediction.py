"""Prediction orchestration: discover rasters, run inference, write outputs."""

from __future__ import annotations

import os
from typing import Optional

from unet_wildfire_predict.config import PredictionConfig
from unet_wildfire_predict.inference import predict_on_new_image
from unet_wildfire_predict.paths import generate_output_path, get_tiff_files


def run_prediction(config: Optional[PredictionConfig] = None) -> None:
    """Discover GeoTIFFs under ``config.image_path`` and run U-Net inference on each.

    Mirrors ``unet_wildfire_training.training.train_model`` in style: callers
    pass a :class:`PredictionConfig`, the function takes care of file discovery,
    output path generation, and per-raster inference.
    """
    if config is None:
        config = PredictionConfig()

    image_path = os.path.normpath(str(config.image_path))
    output_dir = os.path.normpath(str(config.output_dir))
    prob_output_dir = os.path.normpath(str(config.prob_output_dir))
    model_path = os.path.normpath(str(config.model_path))

    if os.path.isfile(image_path):
        tiff_files = [image_path]
        base_dir = os.path.dirname(image_path)
    elif os.path.isdir(image_path):
        tiff_files = get_tiff_files(image_path, recursive=config.recursive)
        base_dir = image_path
    else:
        raise ValueError(f"Invalid image_path: {image_path}")

    if not tiff_files:
        raise ValueError(f"No GeoTIFF files found in {image_path}")

    for tiff_file in tiff_files:
        output_path = generate_output_path(
            tiff_file, base_dir, output_dir,
            suffix="_predicted_mask", preserve_structure=config.preserve_structure,
        )
        prob_output_path = generate_output_path(
            tiff_file, base_dir, prob_output_dir,
            suffix="_probability", preserve_structure=config.preserve_structure,
        )

        print(f"Processing {tiff_file}...")
        predict_on_new_image(
            model_path, tiff_file, output_path, prob_output_path,
            config=config,
        )
