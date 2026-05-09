"""Global-bottleneck MLP baseline for semantic segmentation.

Pools the image to a fixed shape matching ``input_shape``, runs it through
:class:`~src.models.architectures.mlp.MLPBackbone` (same interface as ``mlp.py``),
reshapes the embedding to coarse logits, then upsamples to the input resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.architectures.mlp import MLPBackbone


class MLPSegmentationBaseline(nn.Module):
    """Coarse grid + :class:`MLPBackbone` + bilinear upsampling of logits.

    Args:
        num_classes: Number of segmentation classes (including background).
        input_shape: Shape of the pooled tensor per sample, ``(C, H_b, W_b)``.
                     Use ``C == 3`` for RGB; ``H_b``, ``W_b`` set the bottleneck grid.
        hidden_dims: Sizes of the hidden layers (same as :class:`MLPBackbone`).
        dropout:     Dropout probability (same as :class:`MLPBackbone`).
    """

    def __init__(
        self,
        num_classes: int,
        input_shape: tuple[int, ...],
        hidden_dims: list[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape
        _, hb, wb = input_shape
        output_dim = num_classes * hb * wb
        self.network = MLPBackbone(
            input_shape=input_shape,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        b, _, h, w = x.shape
        pooled = F.adaptive_avg_pool2d(x, output_size=self.input_shape[1:])
        logits_coarse = self.network(pooled).view(
            b, self.num_classes, self.input_shape[1], self.input_shape[2]
        )
        return F.interpolate(
            logits_coarse, size=(h, w), mode="bilinear", align_corners=False
        )
