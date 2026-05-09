import torch
import torch.nn as nn
from torchvision.models import segmentation as tv_seg

_VALID_NAMES = (
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
    "fcn_resnet50",
    "fcn_resnet101",
    "lraspp_mobilenet_v3_large",
)


class TorchVisionSegmentation(nn.Module):
    """Pretrained torchvision segmentation model adapted to `num_classes`.

    Args:
        name:            Architecture key — one of {deeplabv3_resnet50/101,
                         fcn_resnet50/101, lraspp_mobilenet_v3_large}.
        num_classes:     Number of segmentation classes for the new head.
        weights:         "DEFAULT" loads full COCO-pretrained weights and surgically
                         swaps the final classifier conv(s). None gives a fresh head
                         on top of an ImageNet-pretrained backbone — the usual choice
                         when the target label set differs from COCO.
        freeze_backbone: Freeze backbone parameters; only the head trains.
                         Use for the frozen-feature baseline.

    Forward returns a plain (B, num_classes, H, W) tensor — the torchvision
    {"out": ...} dict is unwrapped so SegmentationModel sees a uniform interface.
    """

    def __init__(
        self,
        name: str,
        num_classes: int,
        weights: str | None = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        if name not in _VALID_NAMES:
            raise ValueError(f"Unknown model {name!r}. Valid: {_VALID_NAMES}")
        factory = getattr(tv_seg, name)

        if weights is None:
            # Backbone gets ImageNet pretrain, head built fresh at the right width.
            # aux_loss=False drops the auxiliary head — we don't use it during training.
            self.model = factory(
                weights=None,
                weights_backbone="DEFAULT",
                num_classes=num_classes,
                aux_loss=False,
            )
        else:
            # COCO-pretrained head (21 classes) — load first, then replace final conv(s).
            self.model = factory(weights=weights)
            _replace_classifier_head(self.model, num_classes)

        if freeze_backbone:
            for p in self.model.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]


def _replace_classifier_head(model: nn.Module, num_classes: int) -> None:
    for head_attr in ("classifier", "aux_classifier"):
        head = getattr(model, head_attr, None)
        if head is None:
            continue
        if hasattr(head, "low_classifier") and hasattr(head, "high_classifier"):
            # LRASPP has two parallel 1x1 convs fused at the output.
            head.low_classifier = _swap_out_channels(head.low_classifier, num_classes)
            head.high_classifier = _swap_out_channels(head.high_classifier, num_classes)
        else:
            # DeepLabV3 / FCN: classifier is Sequential, last layer is the per-class conv.
            head[-1] = _swap_out_channels(head[-1], num_classes)


def _swap_out_channels(conv: nn.Conv2d, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(
        conv.in_channels,
        out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
