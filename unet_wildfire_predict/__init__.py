"""U-Net wildfire segmentation prediction package.

Importing this package configures the OpenMP and PROJ environment variables
before any heavy native libraries (PyTorch, rasterio, pyproj) are loaded so
that prediction code behaves identically to the training-time pipeline.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_proj_path = os.path.join(sys.prefix, "Library", "share", "proj")
if os.path.isdir(_proj_path):
    os.environ.setdefault("PROJ_LIB", _proj_path)

from unet_wildfire_predict.config import PredictionConfig
from unet_wildfire_predict.inference import predict_on_new_image
from unet_wildfire_predict.prediction import run_prediction

__all__ = ["PredictionConfig", "predict_on_new_image", "run_prediction"]
