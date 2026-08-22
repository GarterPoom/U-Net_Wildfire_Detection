"""Custom loss functions for class‑imbalanced binary segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsampledBCEWithLogitsLoss(nn.Module):
    """BCE loss with per‑batch random downsampling of the majority pixel class.

    Each forward pass, whichever class (burned vs. unburned) has more pixels in
    the batch is randomly downsampled without replacement to match the minority
    class count. Mean BCE is then computed over the resulting ``2 * min(P, N)``
    pixels, ensuring a class‑balanced gradient step.

    If only one class is present (degenerate case), falls back to standard mean BCE.
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Flatten tensors to 1‑D for element‑wise loss computation
        logits_flat = logits.reshape(-1)
        targets_flat = targets.reshape(-1)

        # Compute raw BCE loss (no reduction) for each pixel
        bce = F.binary_cross_entropy_with_logits(logits_flat, targets_flat, reduction="none")

        # Identify indices of positive (burned) and negative (unburned) pixels
        pos_idx = (targets_flat == 1).nonzero(as_tuple=False).squeeze(1)
        neg_idx = (targets_flat == 0).nonzero(as_tuple=False).squeeze(1)

        n_pos = pos_idx.numel()
        n_neg = neg_idx.numel()

        # If either class is absent, just use the overall mean loss
        if n_pos == 0 or n_neg == 0:
            return bce.mean()

        # Determine the smaller class count to downsample the larger one
        k = min(n_pos, n_neg)

        # Randomly select ``k`` indices from each class without replacement
        if n_pos > k:
            perm = torch.randperm(n_pos, device=logits_flat.device)[:k]
            pos_idx = pos_idx[perm]
        if n_neg > k:
            perm = torch.randperm(n_neg, device=logits_flat.device)[:k]
            neg_idx = neg_idx[perm]

        # Concatenate selected indices and compute mean loss over the balanced set
        selected = torch.cat([pos_idx, neg_idx], dim=0)
        return bce[selected].mean()
