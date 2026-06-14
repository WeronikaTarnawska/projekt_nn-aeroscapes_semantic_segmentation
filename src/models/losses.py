"""Loss functions for semantic segmentation.

Plain-PyTorch implementations (no extra dependencies) so they drop straight into
the existing Fiddle configs as ``fdl.Config(DiceCELoss, ...)``. They accept
``class_weights`` as a list of floats (Fiddle-friendly) and convert it to a
registered buffer internally, mirroring how ``SegmentationModel`` handles its
own weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multiclass soft Dice loss, ``1 - mean(per-class Dice)``.

    Dice measures region overlap — the same quantity IoU rewards — but it is
    normalized *per class*, so every class contributes equally regardless of how
    many pixels it covers. That normalization is exactly what makes it robust to
    class imbalance: a rare class (bike, obstacle) cannot be drowned out by
    vegetation/road the way it is under pixel-summed cross-entropy.

    Args:
        num_classes:  Number of classes ``C`` in the logits.
        ignore_index: Class index excluded from the loss (e.g. background).
        smooth:       Laplace smoothing added to numerator and denominator. Also
                      defines the score for a class absent from the batch
                      (Dice = 1, i.e. zero loss) instead of dividing by zero.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int | None = None,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        if self.ignore_index is not None:
            # Neutralize ignored pixels: zero them out of BOTH probs and target
            # one-hot so they add nothing to intersection or cardinality.
            valid = target != self.ignore_index
            target_clamped = target.clone()
            target_clamped[~valid] = 0
            one_hot = F.one_hot(target_clamped, self.num_classes)
            one_hot = one_hot.permute(0, 3, 1, 2).float()
            keep_mask = valid.unsqueeze(1).float()
            probs = probs * keep_mask
            one_hot = one_hot * keep_mask
        else:
            one_hot = F.one_hot(target, self.num_classes)
            one_hot = one_hot.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)  # sum over batch + spatial, keep the per-class axis
        intersection = (probs * one_hot).sum(dims)
        cardinality = probs.sum(dims) + one_hot.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        if self.ignore_index is not None:
            # Drop the ignored class from the mean so its Dice=1 can't mask the
            # foreground score we actually care about.
            keep = [c for c in range(self.num_classes) if c != self.ignore_index]
            dice = dice[keep]

        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    """Weighted sum of cross-entropy and soft Dice loss.

    Cross-entropy supplies dense, well-behaved per-pixel gradients that keep
    early training stable; Dice directly optimizes overlap and rebalances the
    rare classes. Summing them keeps CE's optimization stability while letting
    Dice pull up the small, under-represented classes. It is a standard,
    low-fuss remedy for segmentation class imbalance — no resampling pipeline,
    no extra hyperparameters beyond the two term weights.

    Args:
        num_classes:   Number of classes.
        ignore_index:  Class excluded from BOTH terms (e.g. background).
        class_weights: Per-class CE weights (e.g. median-frequency from EDA).
                       List/tuple of floats, converted to a buffer internally;
                       ``None`` disables CE weighting.
        ce_weight:     Scalar multiplier on the cross-entropy term.
        dice_weight:   Scalar multiplier on the Dice term.
        smooth:        Dice smoothing (see ``DiceLoss``).
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int | None = None,
        class_weights: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        # -100 is torch's CE sentinel meaning "no class is ignored".
        self._ce_ignore = -100 if ignore_index is None else ignore_index
        self.dice = DiceLoss(num_classes, ignore_index=ignore_index, smooth=smooth)

        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            # Buffer so the weights follow .to(device) and roundtrip checkpoints.
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            ignore_index=self._ce_ignore,
        )
        dice = self.dice(logits, target)
        return self.ce_weight * ce + self.dice_weight * dice
