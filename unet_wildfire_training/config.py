"""Training configuration for the U-Net wildfire pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class TrainingConfig:
    """Hyperparameters and I/O paths for training U-Net on wildfire imagery.

    Attributes:
        image_dir: Directory containing the training rasters (recursively scanned).
        label_dir: Directory containing wildfire polygon shapefiles.
        export_dir: Destination directory for the saved model weights.
        evaluation_dir: Destination directory for metric plots and CSV reports.
        model_filename: Filename of the exported state-dict.
        tile_size: Edge length (in pixels) of raster tiles extracted from each image.
        target_size: Spatial size of tiles fed to the network after resizing.
        batch_size: Mini-batch size used by both DataLoaders.
        num_epochs: Number of training epochs.
        learning_rate: Adam learning rate.
        num_workers: DataLoader worker processes. Keep at 0 on Windows unless the
            entry script uses the multiprocessing-safe ``if __name__ == "__main__"``
            guard and you've validated the worker startup behaviour.
        pin_memory: Whether DataLoader should pin host memory for faster GPU copy.
        val_split: Fraction of tiles held out for validation.
        random_seed: Seed for the stratified train/val split.
        normalization_percentiles: (low, high) percentile clip applied per band
            before min-max scaling to ``[0, 1]``. Matches the percentile clipping
            used by the prediction script for consistency.
    """

    image_dir: Path = Path("Raster_Train")
    label_dir: Path = Path("Wildfire_Polygon_Train")
    export_dir: Path = Path("Export_Model")
    evaluation_dir: Path = Path("Model_Evaluation")
    model_filename: str = "unet_wildfire.pth"

    tile_size: int = 512
    target_size: Tuple[int, int] = (256, 256)

    batch_size: int = 4
    num_epochs: int = 10
    learning_rate: float = 1e-4
    num_workers: int = 0
    pin_memory: bool = False

    val_split: float = 0.2
    random_seed: int = 42

    normalization_percentiles: Tuple[float, float] = (2.0, 98.0)

    def model_path(self) -> Path:
        return self.export_dir / self.model_filename
