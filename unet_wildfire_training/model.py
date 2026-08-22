"""U-Net architecture for binary semantic segmentation.

The implementation follows the classic encoder‑decoder design:
* Down‑sampling (max‑pool + double conv) reduces spatial resolution.
* Up‑sampling (transposed conv) restores resolution, and skip connections concatenate
  the corresponding encoder features.
* The final 1×1 convolution maps the feature vector to a single logit per pixel,
  which is later turned into a probability with ``torch.sigmoid``.

The class is deliberately **generic** – you can change the number of input
channels (e.g. multispectral imagery) and the number of output classes.
For binary segmentation set ``n_classes=1`` and use ``BCEWithLogitsLoss``.
"""
import torch                                      # Core deep‑learning library
import torch.nn as nn                             # Base neural‑network building blocks
import torch.nn.functional as F                   # Functional API (e.g. padding, activations)

# --------------------------------------------------------------------------- #
# Reusable building block – two conv layers with batch‑norm and ReLU.
# --------------------------------------------------------------------------- #

class DoubleConv(nn.Module):
    """
    Two consecutive 3×3 convolutions, each followed by BatchNorm and ReLU.

    The spatial dimensions stay unchanged because ``padding=1``.
    This module is the fundamental “feature extractor” used at every
    resolution level of the U‑Net.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Parameters
        ----------
        in_channels : int
            Number of channels received from the previous layer (or input image).
        out_channels : int
            Number of channels produced by this block.
        """
        super().__init__()
        # ``nn.Sequential`` makes the code concise and keeps the forward pass clean.
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 3×3 conv
            nn.BatchNorm2d(out_channels),                                      # Normalise activations
            nn.ReLU(inplace=True),                                            # Non‑linearity (in‑place to save memory)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # Second 3×3 conv
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass the input tensor through the two convolution‑batch‑norm‑ReLU stages.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape ``(B, C_in, H, W)``.

        Returns
        -------
        torch.Tensor
            Output feature map of shape ``(B, C_out, H, W)``.
        """
        return self.double_conv(x)


# --------------------------------------------------------------------------- #
# Encoder block – max‑pool followed by ``DoubleConv`` (down‑sampling)
# --------------------------------------------------------------------------- #

class Down(nn.Module):
    """
    Encoder block: 2×2 max‑pooling reduces the spatial size by a factor of two,
    after which ``DoubleConv`` extracts features at the lower resolution.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Parameters
        ----------
        in_channels : int
            Number of channels entering the block (coming from previous layer).
        out_channels : int
            Number of channels produced after down‑sampling.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),                     # 2×2 max‑pool → halve H and W
            DoubleConv(in_channels, out_channels)   # Extract features at the new resolution
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply max‑pool then the double convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, C_in, H, W)``.

        Returns
        -------
        torch.Tensor
            Down‑sampled feature map of shape ``(B, C_out, H/2, W/2)``.
        """
        return self.maxpool_conv(x)


# --------------------------------------------------------------------------- #
# Decoder block – transposed convolution + skip connection concat → DoubleConv
# --------------------------------------------------------------------------- #

class Up(nn.Module):
    """
    Decoder block: upsample the input (using a transposed conv), then concatenate
    it with the corresponding encoder feature map (skip connection) and finally
    apply ``DoubleConv`` to fuse the information.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Parameters
        ----------
        in_channels : int
            Number of channels coming from the up‑sampling step (typically
            ``C_{up} = C_{down} / 2``).
        out_channels : int
            Number of channels after the DoubleConv (usually equal to the
            number of channels from the corresponding encoder block).
        """
        super().__init__()
        # The transposed convolution performs nearest‑neighbor up‑sampling by a factor of 2.
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                    kernel_size=2, stride=2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Upsample ``x1`` and concatenate it with the skip‑connection ``x2``.

        Parameters
        ----------
        x1 : torch.Tensor
            Feature map from the decoder (to be up‑sampled). Shape ``(B, C_up, H, W)``.
        x2 : torch.Tensor
            Corresponding encoder feature map (same spatial size as the upsampled
            version of ``x1``). Shape ``(B, C_skip, H, W)``.

        Returns
        -------
        torch.Tensor
            Output of ``DoubleConv`` that fuses both sources.
        """
        x1 = self.up(x1)                                 # Upsample to match spatial size of x2

        # Pad ``x1`` so that its dimensions exactly match those of ``x2``.
        # The padding is symmetric (left/right, top/bottom) except for a possible
        # 1‑pixel difference when the sizes are odd.
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1,
                   [diff_x // 2, diff_x - diff_x // 2,   # left, right
                    diff_y // 2, diff_y - diff_y // 2])  # top, bottom

        # Concatenate along the channel dimension and run the double convolution.
        return self.conv(torch.cat([x2, x1], dim=1))


# --------------------------------------------------------------------------- #
# Output convolution – maps final feature map to per‑pixel logits (C = 1 for binary)
# --------------------------------------------------------------------------- #

class OutConv(nn.Module):
    """
    1×1 convolution that converts the final feature vector into logits for each class.
    For binary segmentation we set ``out_channels=1`` and later apply ``sigmoid``.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Parameters
        ----------
        in_channels : int
            Number of channels coming from the decoder (typically 64 in this architecture).
        out_channels : int
            Number of output classes. Use ``1`` for binary segmentation.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the 1×1 convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, C_in, H, W)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, out_channels, H, W)``.
        """
        return self.conv(x)


# --------------------------------------------------------------------------- #
# Full U‑Net model – symmetric encoder/decoder with 5 levels of abstraction.
# --------------------------------------------------------------------------- #

class UNet(nn.Module):
    """
    Classic U‑Net for binary semantic segmentation.

    The architecture follows the design described in the original paper
    (Ronneberger et al., 2015) but uses a **64 → 1024** channel progression:

    * Encoder (down‑sampling) stages: 64 → 128 → 256 → 512 → 1024
    * Decoder (up‑sampling) stages: 1024 → 512 → 256 → 128 → 64
    * Final 1×1 convolution maps to a single logit per pixel.

    Parameters
    ----------
    n_channels : int
        Number of input image channels (e.g. 3 for RGB, 4 for RGBA, or more for
        multispectral data).
    n_classes : int
        Number of output classes. For binary segmentation set ``n_classes=1``.
    """

    def __init__(self, n_channels: int, n_classes: int):
        """
        The constructor builds the whole network from elementary blocks.

        The numeric layout is:
            inc   – initial double conv (64 channels)
            down1 – 64 → 128
            down2 – 128 → 256
            down3 – 256 → 512
            down4 – 512 → 1024   (bottleneck)
            up1   – 1024 → 512
            up2   – 512 → 256
            up3   – 256 → 128
            up4   – 128 → 64
            outc  – final 1×1 conv (n_classes)
        """
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)          # Initial feature extraction
        self.down1 = Down(64, 128)                     # 64 → 128
        self.down2 = Down(128, 256)                    # 128 → 256
        self.down3 = Down(256, 512)                    # 256 → 512
        self.down4 = Down(512, 1024)                   # 512 → 1024 (bottleneck)

        self.up1 = Up(1024, 512)                       # 1024 → 512
        self.up2 = Up(512, 256)                        # 512 → 256
        self.up3 = Up(256, 128)                        # 256 → 128
        self.up4 = Up(128, 64)                         # 128 → 64

        self.outc = OutConv(64, n_classes)             # Map to logits (C = n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the entire U‑Net.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape ``(B, C_in, H, W)``.

        Returns
        -------
        torch.Tensor
            Logits (or probabilities if you apply ``sigmoid`` later) with shape
            ``(B, n_classes, H, W)`` – spatial dimensions are identical to the input.
        """
        # -------------------- Encoder (down‑sampling) --------------------
        x1 = self.inc(x)                # Low‑level features (64 channels)
        x2 = self.down1(x1)             # 128 channels, quarter size
        x3 = self.down2(x2)             # 256 channels, eighth size
        x4 = self.down3(x3)             # 512 channels, sixteenth size
        x5 = self.down4(x4)             # 1024 channels, thirty‑second size (bottleneck)

        # -------------------- Decoder (up‑sampling) --------------------
        x = self.up1(x5, x4)            # 1024 → 512, restore spatial resolution
        x = self.up2(x, x3)             # 512 → 256
        x = self.up3(x, x2)             # 256 → 128
        x = self.up4(x, x1)             # 128 → 64

        # -------------------- Output --------------------
        logits = self.outc(x)           # Final 1×1 conv → (B, n_classes, H, W)
        return logits
