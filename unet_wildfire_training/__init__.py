"""U-Net wildfire segmentation training and prediction package.

Importing this package configures the OpenMP and PROJ environment variables
before any heavy native libraries (PyTorch, rasterio, pyproj) are loaded.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_proj_path = os.path.join(sys.prefix, "Library", "share", "proj")
if os.path.isdir(_proj_path):
    os.environ.setdefault("PROJ_LIB", _proj_path)

from unet_wildfire_training.config import TrainingConfig
from unet_wildfire_training.model import UNet
from unet_wildfire_training.losses import DownsampledBCEWithLogitsLoss

__all__ = ["TrainingConfig", "UNet", "DownsampledBCEWithLogitsLoss"]
