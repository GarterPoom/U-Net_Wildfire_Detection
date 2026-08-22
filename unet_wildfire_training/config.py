"""Training configuration for the U‑Net wildfire pipeline.

The :class:`TrainingConfig` dataclass centralises all hyperparameters and I/O paths required
to train a U‑Net model on wildfire imagery.
"""

from dataclasses import dataclass, field  # Dataclass decorator and field marker for default values
from pathlib import Path  # Convenient path handling using OS‑agnostic objects
from typing import Tuple  # Tuple type hint

@dataclass
class TrainingConfig:
    """Hyperparameters and I/O paths for training U‑Net on wildfire imagery.

    Attributes:
        image_dir: Directory containing the training rasters (recursively scanned).
        label_dir: Directory containing wildfire polygon shapefiles.
        export_dir: Destination directory for the saved model weights.
        evaluation_dir: Destination directory for metric plots and CSV reports.
        model_filename: Filename of the exported state‑dict.
        tile_size: Edge length (in pixels) of raster tiles extracted from each image.
        target_size: Spatial size of tiles fed to the network after resizing.
        batch_size: Mini‑batch size used by both DataLoaders.
        num_epochs: Number of training epochs.
        learning_rate: Adam learning rate.
        num_workers: DataLoader worker processes. Keep at 0 on Windows unless
            the entry script uses a ``if __name__ == "__main__"`` guard and you've validated worker startup.
        pin_memory: Whether DataLoader should pin host memory for faster GPU copy.
        val_split: Fraction of tiles held out for validation.
        random_seed: Seed for the stratified train/val split.
        normalization_percentiles: (low, high) percentile clip applied per band
            before min‑max scaling to ``[0, 1]``. Matches the clipping used by the prediction script.
    """
    image_dir: Path = Path("Raster_Train")  # Directory that holds training raster files
    label_dir: Path = Path("Wildfire_Polygon_Train")  # Directory that holds shapefile labels
    export_dir: Path = Path("Export_Model")  # Where the trained model checkpoint is stored
    evaluation_dir: Path = Path("Model_Evaluation")  # Folder for plots, CSVs and reports
    model_filename: str = "unet_wildfire.pth"  # Name of the saved state‑dict file

    tile_size: int = 512  # Size (edge length) of each raster tile extracted from images
    target_size: Tuple[int, int] = (256, 256)  # Desired spatial dimensions after resizing

    batch_size: int = 4  # Number of samples processed in one training step
    num_epochs: int = 10  # Total number of passes over the entire training set
    learning_rate: float = 1e-4  # Step size for Adam optimizer
    num_workers: int = 0  # Parallel workers for DataLoader; Windows needs special handling
    pin_memory: bool = False  # Pin memory to speed up host‑to‑device copies

    val_split: float = 0.2  # Portion of tiles reserved for validation
    random_seed: int = 42  # Seed used for stratified train/val split reproducibility
    normalization_percentiles: Tuple[float, float] = (2.0, 98.0)  # Percentile range for per‑band clipping

    def model_path(self) -> Path:
        """Return the full path to the exported model checkpoint."""
        return self.export_dir / self.model_filename
