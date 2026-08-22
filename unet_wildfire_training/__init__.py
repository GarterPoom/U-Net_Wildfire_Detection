"""U‑Net wildfire segmentation training and prediction package.

Importing this package configures the OpenMP and PROJ environment variables
before any heavy native libraries (PyTorch, rasterio, pyproj) are loaded.
"""

import os  # Configure environment to allow duplicate MKL libraries (prevents crashes)
import sys  # Needed to locate the PROJ library directory

# Set default for KMP_DUPLICATE_LIB_OK to avoid Intel MKL duplicate‑library errors
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Locate the PROJ data directory inside the package prefix and expose it via PROJ_LIB if present
_proj_path = os.path.join(sys.prefix, "Library", "share", "proj")
if os.path.isdir(_proj_path):
    os.environ.setdefault("PROJ_LIB", _proj_path)

# Import objects that should be publicly accessible when the package is imported.
from unet_wildfire_training.config import TrainingConfig  # Training configuration dataclass
from unet_wildfire_training.model import UNet  # Core U‑Net model definition
from unet_wildfire_training.losses import DownsampledBCEWithLogitsLoss  # Custom loss function

# Define symbols that are exported by ``from unet_wildfire_training import *``.
__all__ = ["TrainingConfig", "UNet", "DownsampledBCEWithLogitsLoss"]
