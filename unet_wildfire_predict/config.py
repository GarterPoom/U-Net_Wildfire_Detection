"""Prediction configuration for the U‑Net wildfire pipeline.

The :class:`PredictionConfig` dataclass centralises all hyperparameters and I/O paths required
to run inference.  It mirrors the training‑side configuration (tiling, normalisation) so that
the same preprocessing steps are applied to both phases.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

# Import the training‑time configuration to enable easy adaptation.
from unet_wildfire_training import TrainingConfig


@dataclass
class PredictionConfig:
    """Hyperparameters and I/O paths for running U‑Net inference.

    Attributes:
        model_path: Path to the trained ``.pth`` state‑dict file.
        image_path: Directory or single GeoTIFF file containing input rasters.
        output_dir: Destination folder for binary mask rasters (e.g., predicted burn areas).
        prob_output_dir: Destination folder for probability map rasters.
        tile_size: Size of the sliding window used during inference (must match training).
        target_size: Spatial dimensions (height, width) that the network expects as input.
        overlap: Number of pixels shared between adjacent tiles; helps seamless stitching.
        normalization_percentiles: Tuple ``(low, high)`` indicating percentile clip before min‑max scaling.
        band_layout: Optional dictionary mapping Sentinel‑2 band names to 0‑indexed channel positions.
            When provided, indices (NDVI, NDWI, SAVI, BAIS2) are appended to the input stack.
        reflectance_scale: Divisor applied to raw band values before index computation
            (10000 for Sentinel‑2 L2A uint16; 1.0 if bands are already in ``[0, 1]``).
        recursive: If True, search sub‑directories of ``image_path`` for GeoTIFFs.
        preserve_structure: When True, replicate the input directory hierarchy inside the output folders.
        visualize: Show a Matplotlib figure for each processed raster (useful for debugging).
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
        """Create a :class:`PredictionConfig` whose tiling and normalisation match a training config.

        Args:
            training: Configuration used during model training.

        Returns:
            A new ``PredictionConfig`` instance with matching ``tile_size``,
            ``target_size``, and ``normalization_percentiles``.
        """
        return cls(
            model_path=training.model_path(),
            tile_size=training.tile_size,
            target_size=training.target_size,
            normalization_percentiles=training.normalization_percentiles,
        )

    def to_training_config(self) -> TrainingConfig:
        """Adapt this config for reuse of training‑side data helpers (e.g., dataloaders)."""
        cfg = TrainingConfig()
        cfg.tile_size = self.tile_size
        cfg.target_size = self.target_size
        cfg.normalization_percentiles = self.normalization_percentiles
        return cfg
