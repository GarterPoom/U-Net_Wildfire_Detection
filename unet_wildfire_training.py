"""Entry script for training the U-Net wildfire segmentation model.

All logic lives in the ``unet_wildfire`` package; this file just builds a
configuration object and hands it to ``train_model``.
"""

from unet_wildfire_training import TrainingConfig
from unet_wildfire_training.training import train_model


def main() -> None:
    train_model(TrainingConfig())
    

if __name__ == "__main__":
    main() 