"""Prediction configuration for the U-Net wildfire pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from unet_wildfire_training import TrainingConfig


@dataclass
class PredictionConfig:
    """Hyperparameters and I/O paths for running U-Net inference.

    Attributes:
        model_path: Path to the trained ``.pth`` state-dict.
        image_path: GeoTIFF file or directory of GeoTIFFs to predict on.
        output_dir: Destination directory for binary mask rasters.
        prob_output_dir: Destination directory for probability rasters.
        tile_size: Raster-window tile edge in pixels (must match training).
        target_size: Network-input tile size in pixels (must match training).
        overlap: Pixel overlap between adjacent raster tiles during stitching.
        normalization_percentiles: ``(low, high)`` percentile clip applied per
            band before min-max scaling. Matches the training pipeline.
        band_layout: Optional 0-indexed channel positions for Sentinel-2 bands.
            When provided, NDVI/NDWI/SAVI/BAIS2 are appended to the input stack.
        reflectance_scale: Divisor applied to raw bands before index computation
            (10000 for Sentinel-2 L2A uint16, 1.0 if already in ``[0, 1]``).
        recursive: Whether to search subdirectories of ``image_path``.
        preserve_structure: Whether to mirror the input directory structure in
            the output directories.
        visualize: Display a matplotlib figure of each prediction.
    """

    model_path: Path = Path("Export_Model") / "unet_wildfire.pth"
    image_path: Path = Path("Raster_Classified_Cloud_Mask")
    output_dir: Path = Path("Predicted_Mask")
    prob_output_dir: Path = Path("Predicted_Probability")

    tile_size: int = 512
    target_size: Tuple[int, int] = (256, 256)
    overlap: int = 64

    normalization_percentiles: Tuple[float, float] = (2.0, 98.0)

    band_layout: Optional[Dict[str, int]] = None
    reflectance_scale: float = 10000.0

    recursive: bool = True
    preserve_structure: bool = False
    visualize: bool = False

    @classmethod
    def from_training(cls, training: TrainingConfig) -> "PredictionConfig":
        """Build a prediction config whose tiling/normalization match training."""
        return cls(
            model_path=training.model_path(),
            tile_size=training.tile_size,
            target_size=training.target_size,
            normalization_percentiles=training.normalization_percentiles,
        )

    def to_training_config(self) -> TrainingConfig:
        """Adapter for reusing the training-side data helpers during inference."""
        cfg = TrainingConfig()
        cfg.tile_size = self.tile_size
        cfg.target_size = self.target_size
        cfg.normalization_percentiles = self.normalization_percentiles
        return cfg
