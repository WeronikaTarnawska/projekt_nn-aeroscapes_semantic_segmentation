import torch
import torch.nn as nn

from src.models.modules.conv_block import ConvBlock


class UNet(nn.Module):
    """Vanilla U-Net for semantic segmentation.

    Symmetric encoder/decoder with skip connections. Each encoder stage doubles
    channels and halves spatial dims; each decoder stage mirrors with transposed
    convolution + concat-skip + double conv.

    Input H and W must be divisible by 2 ** (len(channel_dims) - 1) so the skip
    connections align with the upsampled features.

    Args:
        in_channels:  Input image channels (3 for RGB).
        num_classes:  Output channels — one logit per class per pixel.
        channel_dims: Channel widths per encoder level. The last entry is the
                      bottleneck. Standard U-Net uses [64, 128, 256, 512, 1024].
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channel_dims: list[int],
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleList()
        prev = in_channels
        for c in channel_dims:
            self.encoders.append(
                nn.Sequential(
                    ConvBlock(prev, c, kernel_size=3, padding=1, bias=False),
                    ConvBlock(c, c, kernel_size=3, padding=1, bias=False),
                )
            )
            prev = c
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder mirrors encoders, skipping the bottleneck (last entry).
        decoder_dims = list(reversed(channel_dims[:-1]))
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = channel_dims[-1]
        for c in decoder_dims:
            self.upsamples.append(nn.ConvTranspose2d(prev, c, kernel_size=2, stride=2))
            # After concat with skip (also c channels), input width is 2 * c.
            self.decoders.append(
                nn.Sequential(
                    ConvBlock(2 * c, c, kernel_size=3, padding=1, bias=False),
                    ConvBlock(c, c, kernel_size=3, padding=1, bias=False),
                )
            )
            prev = c

        self.head = nn.Conv2d(prev, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            if i < len(self.encoders) - 1:
                skips.append(x)
                x = self.pool(x)
        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return self.head(x)
