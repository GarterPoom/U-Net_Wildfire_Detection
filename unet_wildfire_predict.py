"""End-to-end wildfire prediction entry point with logging and timing."""

# Import the typing module for future-proofing type hints
from __future__ import annotations

# Import argument parsing for command line interface
import argparse
# Import path manipulation utilities
from pathlib import Path
# Import time to implement the stopwatch functionality
import time
# Import datetime to create dynamic filenames for logs
from datetime import datetime
# Import system utilities for standard output access
import sys
# Import numpy for numerical operations
import numpy as np
# Import rasterio for handling geospatial raster data
import rasterio
# Import logging to handle professional-grade event recording
import logging

# Import preprocessing functions from the local package
from classified_preprocessing import (
    prepare_sentinel2_raw,
    run_cloud_masking,
    run_image_processing,
)
# Import the polygon extraction function
from unet_polygon import main as run_polygon_extraction
# Import configuration and prediction orchestration from the local package
from unet_wildfire_predict.config import PredictionConfig
from unet_wildfire_predict.inference import run_prediction
from unet_wildfire_predict.band_layout import parse_band_layout
from unet_wildfire_predict.paths import get_tiff_files
from unet_wildfire_training import TrainingConfig
from unet_wildfire_training.data import SENTINEL2_INDEX_NAMES


def _threshold_arg(value: str) -> float:
    """Validates and converts the threshold argument to a float."""
    try:
        # Attempt to convert the input string to a floating point number
        threshold = float(value)
    except ValueError as exc:
        # If conversion fails, raise a user-friendly error message
        raise argparse.ArgumentTypeError(
            f"--threshold must be a float, got {value!r}"
        ) from exc
    # Check if the threshold is within the logical range [0.1, 0.9]
    if not 0.1 <= threshold <= 0.9:
        # If outside range, raise error
        raise argparse.ArgumentTypeError(
            f"--threshold must be in [0.1, 0.9], got {threshold}"
        )
    # Return the validated float
    return threshold


def _setup_logging(log_dir: Path) -> str:
    """Configures logging to both the console and a dynamic file in a specific directory."""
    # Ensure the log directory exists; create it if it doesn't
    log_dir.mkdir(parents=True, exist_ok=True)
    # Create a dynamic filename using the current date and time (YearMonthDay_HourMinSec)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Define the full path for the log file
    log_file_path = log_dir / f"wildfire_prediction_{timestamp}.log"
    
    # Configure the logging system
    logging.basicConfig(
        level=logging.INFO,  # Set minimum logging level to INFO
        format='%(asctime)s [%(levelname)s] %(message)s',  # Define log format (Timestamp [Level] Message)
        handlers=[
            logging.FileHandler(log_file_path),  # Save logs to the dynamic file
            logging.StreamHandler(sys.stdout)     # Also print logs to the terminal/console
        ]
    )
    # Log the creation of the log file so the user knows where to find it
    logging.info(f"Logging initialized. Log file: {log_file_path}")
    # Return the path for reference
    return str(log_file_path)


def _apply_threshold(prob_output_dir: Path, output_dir: Path, threshold: float) -> None:
    """Re-binarize probability rasters with the chosen threshold."""
    # Find all probability files in the output directory
    prob_files = get_tiff_files(prob_output_dir, recursive=True)
    
    # Check if any files were actually found
    if not prob_files:
        logging.warning(f"No probability rasters found under {prob_output_dir}; skipping threshold pass.")
        return

    # Iterate through every found probability file
    for prob_path in prob_files:
        # Convert the path to a Path object for easier manipulation
        prob_path = Path(prob_path)
        # Calculate the relative path to maintain directory structure
        rel = prob_path.relative_to(prob_output_dir)
        # Define the matching filename for the binary mask
        mask_name = rel.name.replace("_probability.tif", "_predicted_mask.tif")
        # Define the destination path for the mask
        mask_path = output_dir / rel.parent / mask_name
        # Ensure the subdirectories for the mask exist
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        # Open the probability raster
        with rasterio.open(prob_path) as src:
            # Read the first band (the probability values)
            prob = src.read(1)
            # Copy the metadata from the source file
            meta = src.meta.copy()

        # Create a binary mask where values > threshold are 1, otherwise 0
        mask = (prob > threshold).astype(np.uint8)
        # Update metadata for the output (1 band, uint8, use LZW compression)
        meta.update(count=1, dtype="uint8", nodata=255, compress="lzw")
        
        # Write the binary mask to the destination path
        with rasterio.open(mask_path, "w", **meta) as dst:
            dst.write(mask, 1)
        # Log the successful re-thresholding of the file
        logging.info(f"Re-thresholded {mask_path} (threshold={threshold})")


def _build_parser() -> argparse.ArgumentParser:
    """Constructs the command-line argument parser."""
    # Load default values from the Training configuration
    defaults = PredictionConfig.from_training(TrainingConfig())

    # Initialize the argument parser
    parser = argparse.ArgumentParser(
        description="End-to-end wildfire prediction from raw Sentinel-2 data."
    )

    # Add arguments for input/output directories
    parser.add_argument("--raw_dir", default="Sentinel2_Raw", help="Raw Sentinel-2 inputs")
    parser.add_argument("--prepared_dir", default=None, help="Extracted JP2 folders")
    parser.add_argument("--extracted_dir", default=None, help="Unzipped archives")
    parser.add_argument("--stacked_dir", default="Raster_Classified", help="Resampled GeoTIFFs")
    parser.add_argument("--scl_dir", default="SCL_Classified", help="SCL rasters")
    parser.add_argument("--cloud_masked_dir", default="Raster_Classified_Cloud_Mask", help="Prediction input")
    parser.add_argument("--log_dir", default="logs", help="Directory where log files will be saved")

    # Add arguments for model and prediction settings
    parser.add_argument("--model_path", default=str(defaults.model_path), help="Path to trained model")
    parser.add_argument("--output_dir", default=str(defaults.output_dir), help="Binary mask output")
    parser.add_argument("--prob_output_dir", default=str(defaults.prob_output_dir), help="Prob map output")
    parser.add_argument("--tile_size", type=int, default=defaults.tile_size, help="Tile edge size")
    parser.add_argument("--target_size", type=int, default=defaults.target_size[0], help="Network input size")
    parser.add_argument("--overlap", type=int, default=defaults.overlap, help="Tile overlap")
    parser.add_argument("--use_indices", action="store_true", help="Compute spectral indices")
    parser.add_argument("--band_layout", type=str, default=None, help="JSON band layout")
    parser.add_argument("--reflectance_scale", type=float, default=defaults.reflectance_scale, help="Scale factor")
    parser.add_argument("--threshold", type=_threshold_arg, default=0.7, help="Binary threshold [0.1, 0.9]")
    parser.add_argument("--visualize", action="store_true", help="Show plot")
    parser.add_argument("--recursive", action="store_true", default=True, help="Search subdirs")
    parser.add_argument("--preserve_structure", action="store_true", help="Mirror directory structure")

    # Add skip flags to bypass specific stages of the pipeline
    parser.add_argument("--skip_archive_prepare", action="store_true", help="Skip step 1")
    parser.add_argument("--skip_preprocessing", action="store_true", help="Skip step 2")
    parser.add_argument("--skip_cloud_mask", action="store_true", help="Skip step 3")
    parser.add_argument("--skip_prediction", action="store_true", help="Skip step 4")
    parser.add_argument("--skip_polygon", action="store_true", help="Skip step 5")
    
    return parser


def main() -> None:
    """Main execution logic with timing and logging orchestration."""
    # Parse command line arguments
    args = _build_parser().parse_args()
    # Start the stopwatch to measure total runtime
    start_time = time.perf_counter()

    # Initialize the logging system with the user-specified directory
    _setup_logging(Path(args.log_dir))
    # Log the start of the execution
    logging.info("--- Wildfire Prediction Pipeline Started ---")

    # Parse the JSON band layout string
    band_layout = parse_band_layout(args.band_layout)
    
    # Validate requirements for using indices
    if args.use_indices and band_layout is None:
        logging.error("--use_indices requires --band_layout configuration.")
        raise ValueError("--use_indices requires --band_layout")
    
    if band_layout is not None and not args.use_indices:
        logging.info("Note: --band_layout supplied; enabling spectral-index computation.")

    # Convert the raw directory string to a Path object
    raw_dir: Path = Path(args.raw_dir)
    
    # --- STEP 1: PREPARATION ---
    if not args.skip_archive_prepare:
        logging.info("[1/5] Preparing Sentinel-2 raw inputs...")
        raw_dir = prepare_sentinel2_raw(
            raw_input=raw_dir,
            staging_dir=args.prepared_dir,
            extracted_dir=args.extracted_dir,
        )

    # --- STEP 2: PREPROCESSING ---
    if not args.skip_preprocessing:
        logging.info("[2/5] Resampling, stacking, and index computation...")
        run_image_processing(
            root_folder=raw_dir,
            output_folder=args.stacked_dir,
            scl_output_folder=args.scl_dir,
        )

    # --- STEP 3: CLOUD MASKING ---
    if not args.skip_cloud_mask:
        logging.info("[3/5] Cloud / shadow / water masking via SCL...")
        run_cloud_masking(
            scl_dir=args.scl_dir,
            band_dir=args.stacked_dir,
            output_dir=args.cloud_masked_dir,
        )

    # --- STEP 4: INFERENCE ---
    if not args.skip_prediction:
        logging.info("[4/5] U-Net wildfire inference...")
        # Initialize the prediction configuration object
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
        # Run the actual U-Net inference
        run_prediction(config)
        # Apply the user-defined threshold to convert probabilities to binary masks
        _apply_threshold(Path(args.prob_output_dir), Path(args.output_dir), args.threshold)

    # --- STEP 5: VECTORIZATION ---
    if not args.skip_polygon:
        logging.info("[5/5] Vectorizing burn masks into polygons...")
        run_polygon_extraction()

    # Calculate the end time for the stopwatch
    end_time = time.perf_counter()
    # Calculate the total elapsed time in seconds
    total_elapsed = end_time - start_time
    
    # Log the total time taken (formatted to 2 decimal places)
    logging.info(f"--- Workflow Completed Successfully ---")
    logging.info(f"Total execution time: {total_elapsed:.2f} seconds")

if __name__ == "__main__":
    # Entry point of the script
    main()
