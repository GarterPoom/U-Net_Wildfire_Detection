"""End-to-end wildfire prediction entry point.

Single script that runs the full pipeline on raw Sentinel-2 data:

1. :func:`classified_preprocessing.prepare_sentinel2_raw` — extract
   ``.zip`` / ``.SAFE.zip`` archives and flatten ``*.SAFE`` products into
   per-granule folders of the JP2 bands the pipeline needs.
2. :func:`classified_preprocessing.run_image_processing` —
   resample bands to 10 m, stack them with NDVI/NDWI/SAVI/BAIS2, and emit
   compressed GeoTIFFs plus matching SCL rasters.
3. :func:`classified_preprocessing.run_cloud_masking` —
   mask clouds, shadows, and water using the SCL band.
4. :func:`unet_wildfire_predict.run_prediction` — U-Net inference on the
   cloud-masked rasters, writing per-tile probability maps and binary masks.
5. :func:`unet_polygon.main` — vectorize the predicted masks and intersect
   them with administrative boundaries.

Drop raw Sentinel-2 ``.zip`` / ``.SAFE.zip`` archives or ``*.SAFE`` folders
inside ``Sentinel2_Raw`` and run this script; outputs flow through
``Sentinel2_Raw_Extracted`` / ``Sentinel2_Raw_Prepared`` →
``Raster_Classified`` / ``SCL_Classified`` → ``Raster_Classified_Cloud_Mask``
→ ``Predicted_Mask`` / ``Predicted_Probability`` → ``unet_polygon``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio

from classified_preprocessing import (
    prepare_sentinel2_raw,
    run_cloud_masking,
    run_image_processing,
)
from unet_polygon import main as run_polygon_extraction
from unet_wildfire_predict import PredictionConfig, run_prediction
from unet_wildfire_predict.band_layout import parse_band_layout
from unet_wildfire_predict.paths import get_tiff_files
from unet_wildfire_training import TrainingConfig
from unet_wildfire_training.data import SENTINEL2_INDEX_NAMES


def _threshold_arg(value: str) -> float:
    try:
        threshold = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--threshold must be a float, got {value!r}"
        ) from exc
    if not 0.1 <= threshold <= 0.9:
        raise argparse.ArgumentTypeError(
            f"--threshold must be in [0.1, 0.9], got {threshold}"
        )
    return threshold


def _apply_threshold(prob_output_dir: Path, output_dir: Path, threshold: float) -> None:
    """Re-binarize probability rasters with the chosen threshold.

    The package writes binary masks at a fixed 0.5 cut; this walks the
    probability outputs and overwrites the matching ``*_predicted_mask.tif``
    files in ``output_dir`` using ``threshold`` instead.
    """
    prob_files = get_tiff_files(prob_output_dir, recursive=True)
    if not prob_files:
        print(f"No probability rasters found under {prob_output_dir}; skipping threshold pass.")
        return

    for prob_path in prob_files:
        prob_path = Path(prob_path)
        rel = prob_path.relative_to(prob_output_dir)
        mask_name = rel.name.replace("_probability.tif", "_predicted_mask.tif")
        mask_path = output_dir / rel.parent / mask_name
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(prob_path) as src:
            prob = src.read(1)
            meta = src.meta.copy()

        mask = (prob > threshold).astype(np.uint8)
        meta.update(count=1, dtype="uint8", nodata=255, compress="lzw")
        with rasterio.open(mask_path, "w", **meta) as dst:
            dst.write(mask, 1)
        print(f"Re-thresholded {mask_path} (threshold={threshold})")


def _build_parser() -> argparse.ArgumentParser:
    defaults = PredictionConfig.from_training(TrainingConfig())

    parser = argparse.ArgumentParser(
        description="End-to-end wildfire prediction from raw Sentinel-2 data."
    )

    parser.add_argument("--raw_dir", default="Sentinel2_Raw",
                        help="Folder containing raw Sentinel-2 inputs: .zip / .SAFE.zip "
                             "archives, *.SAFE folders, or pre-flattened JP2 granule folders")
    parser.add_argument("--prepared_dir", default=None,
                        help="Folder for per-granule flat JP2 layouts produced from "
                             "archives / SAFE products (default: <raw_dir>_Prepared)")
    parser.add_argument("--extracted_dir", default=None,
                        help="Folder for unpacked .zip / .SAFE.zip archives "
                             "(default: <raw_dir>_Extracted)")
    parser.add_argument("--stacked_dir", default="Raster_Classified",
                        help="Folder for resampled+stacked multi-band GeoTIFFs (step 2 output)")
    parser.add_argument("--scl_dir", default="SCL_Classified",
                        help="Folder for exported SCL rasters (step 2 output)")
    parser.add_argument("--cloud_masked_dir", default="Raster_Classified_Cloud_Mask",
                        help="Folder for SCL-masked rasters (step 3 output, prediction input)")

    parser.add_argument("--model_path", default=str(defaults.model_path),
                        help="Path to trained model")
    parser.add_argument("--output_dir", default=str(defaults.output_dir),
                        help="Directory to save output binary masks")
    parser.add_argument("--prob_output_dir", default=str(defaults.prob_output_dir),
                        help="Directory to save output probability maps")
    parser.add_argument("--tile_size", type=int, default=defaults.tile_size,
                        help="Raster-window tile edge in pixels (matches training)")
    parser.add_argument("--target_size", type=int, default=defaults.target_size[0],
                        help="Network-input tile edge in pixels (matches training)")
    parser.add_argument("--overlap", type=int, default=defaults.overlap,
                        help="Overlap between raster-window tiles in pixels")
    parser.add_argument("--use_indices", action="store_true",
                        help="Append Sentinel-2 NDVI/NDWI/SAVI/BAIS2 to input stack "
                             "(only needed if the stacked rasters don't already include them)")
    parser.add_argument("--band_layout", type=str, default=None,
                        help='JSON band layout, e.g. \'{"B03":2,"B04":3,"B06":5,"B07":6,"B08":7,"B8A":8,"B12":10}\'')
    parser.add_argument("--reflectance_scale", type=float, default=defaults.reflectance_scale,
                        help="Reflectance divisor for index computation (1.0 if already [0,1])")
    parser.add_argument("--threshold", type=_threshold_arg, default=defaults.threshold,
                        help="Probability cutoff for the binary burn mask (must be in [0.1, 0.9])")
    parser.add_argument("--visualize", action="store_true", help="Visualize the predicted mask")
    parser.add_argument("--recursive", action="store_true", default=True,
                        help="Search for GeoTIFFs in subdirectories")
    parser.add_argument("--preserve_structure", action="store_true",
                        help="Preserve input directory structure in output")

    parser.add_argument("--skip_archive_prepare", action="store_true",
                        help="Skip step 1 (archive extraction / SAFE flattening); "
                             "assume --raw_dir is already a folder of flat JP2 granule dirs")
    parser.add_argument("--skip_preprocessing", action="store_true",
                        help="Skip step 2 (image processing); assume stacked rasters already exist")
    parser.add_argument("--skip_cloud_mask", action="store_true",
                        help="Skip step 3 (cloud masking); assume cloud-masked rasters already exist")
    parser.add_argument("--skip_prediction", action="store_true",
                        help="Skip step 4 (U-Net inference); assume predicted masks already exist")
    parser.add_argument("--skip_polygon", action="store_true",
                        help="Skip step 5 (polygon extraction)")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    band_layout = parse_band_layout(args.band_layout)
    if args.use_indices and band_layout is None:
        raise ValueError(
            "--use_indices requires --band_layout describing the Sentinel-2 "
            f"band positions ({', '.join(SENTINEL2_INDEX_NAMES)} need B03, B04, "
            "B06, B07, B08, B8A, B12)."
        )
    if band_layout is not None and not args.use_indices:
        print("Note: --band_layout supplied; enabling spectral-index computation.")

    raw_dir: Path = Path(args.raw_dir)
    if not args.skip_archive_prepare:
        print("[1/5] Preparing Sentinel-2 raw inputs (.zip / .SAFE / .SAFE.zip)...")
        raw_dir = prepare_sentinel2_raw(
            raw_input=raw_dir,
            staging_dir=args.prepared_dir,
            extracted_dir=args.extracted_dir,
        )

    if not args.skip_preprocessing:
        print("\n[2/5] Sentinel-2 band resampling, stacking, and index computation...")
        run_image_processing(
            root_folder=raw_dir,
            output_folder=args.stacked_dir,
            scl_output_folder=args.scl_dir,
        )

    if not args.skip_cloud_mask:
        print("\n[3/5] Cloud / shadow / water masking via SCL...")
        run_cloud_masking(
            scl_dir=args.scl_dir,
            band_dir=args.stacked_dir,
            output_dir=args.cloud_masked_dir,
        )

    if not args.skip_prediction:
        print("\n[4/5] U-Net wildfire inference...")
        config = PredictionConfig(
            model_path=Path(args.model_path),
            image_path=Path(args.cloud_masked_dir),
            output_dir=Path(args.output_dir),
            prob_output_dir=Path(args.prob_output_dir),
            tile_size=args.tile_size,
            target_size=(args.target_size, args.target_size),
            overlap=args.overlap,
            band_layout=band_layout,
            reflectance_scale=args.reflectance_scale,
            recursive=args.recursive,
            preserve_structure=args.preserve_structure,
            visualize=args.visualize,
        )
        run_prediction(config)
        _apply_threshold(Path(args.prob_output_dir), Path(args.output_dir), args.threshold)

    if not args.skip_polygon:
        print("\n[5/5] Vectorizing burn masks into polygon shapefiles...")
        run_polygon_extraction()

    print("\nPrediction workflow completed successfully.")


if __name__ == "__main__":
    main()
