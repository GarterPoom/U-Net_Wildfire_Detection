"""Custom loss functions for class-imbalanced binary segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsampledBCEWithLogitsLoss(nn.Module):
    """BCE loss with per-batch random downsampling of the majority pixel class.

    Each forward pass, whichever class (burned vs. unburned) has more pixels in
    the batch is randomly downsampled without replacement to match the minority
    class count. Mean BCE is then computed over the resulting ``2 * min(P, N)``
    pixels, so every gradient step is driven by an exactly class-balanced
    pixel set.

    Across epochs the model still sees every pixel — a fresh majority-class
    subsample is drawn on each forward pass.

    If only one class is present in the batch (degenerate case), falls back to
    mean BCE over the available pixels.
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits_flat = logits.reshape(-1)
        targets_flat = targets.reshape(-1)

        bce = F.binary_cross_entropy_with_logits(logits_flat, targets_flat, reduction="none")
        pos_idx = (targets_flat == 1).nonzero(as_tuple=False).squeeze(1)
        neg_idx = (targets_flat == 0).nonzero(as_tuple=False).squeeze(1)

        n_pos = pos_idx.numel()
        n_neg = neg_idx.numel()

        if n_pos == 0 or n_neg == 0:
            return bce.mean()

        k = min(n_pos, n_neg)
        if n_pos > k:
            perm = torch.randperm(n_pos, device=logits_flat.device)[:k]
            pos_idx = pos_idx[perm]
        if n_neg > k:
            perm = torch.randperm(n_neg, device=logits_flat.device)[:k]
            neg_idx = neg_idx[perm]

        selected = torch.cat([pos_idx, neg_idx], dim=0)
        return bce[selected].mean()