import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger

import wandb
from src.config.aeroscapes_constants import IMAGENET_MEAN, IMAGENET_STD


class LogPredictionsCallback(L.Callback):
    """Log N segmentation predictions to W&B at the end of each validation epoch.

    Each panel shows the input image with two W&B mask overlays — ground truth
    and prediction — togglable in the run dashboard. The same images are reused
    across epochs (cached on first val batch) so progress is visually comparable.

    Args:
        num_samples:    Number of validation images to log per epoch.
        class_names:    Optional class label list — gives the W&B mask a legend.
        denorm_mean:    Per-channel mean used by the dataset's Normalize step.
                        Default = ImageNet, matching AeroScapesDataModule.
        denorm_std:     Per-channel std used by the dataset's Normalize step.
    """

    def __init__(
        self,
        num_samples: int = 8,
        class_names: list[str] | None = None,
        denorm_mean: tuple[float, float, float] = IMAGENET_MEAN,
        denorm_std: tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.class_names = class_names
        # Reshape for broadcast against (B, C, H, W).
        self._mean = torch.tensor(denorm_mean).view(1, 3, 1, 1)
        self._std = torch.tensor(denorm_std).view(1, 3, 1, 1)
        self._cached: tuple[torch.Tensor, torch.Tensor] | None = None

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if self._cached is not None or batch_idx != 0:
            return
        images, attributes = batch
        self._cached = (
            images[: self.num_samples].detach().cpu(),
            attributes["mask"][: self.num_samples].detach().cpu(),
        )

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        wandb_logger = next(
            (lg for lg in trainer.loggers if isinstance(lg, WandbLogger)), None
        )
        if wandb_logger is None or self._cached is None:
            return

        images, gt_masks = self._cached
        with torch.inference_mode():
            preds = pl_module(images.to(pl_module.device)).argmax(dim=1).cpu()

        # Undo the dataset's Normalize(mean, std) so the panel shows real RGB.
        vis = (images * self._std + self._mean).clamp(0, 1).mul(255).byte()
        # W&B expects HWC uint8 numpy.
        vis = vis.permute(0, 2, 3, 1).numpy()

        class_labels = (
            {i: name for i, name in enumerate(self.class_names)}
            if self.class_names
            else None
        )
        wandb_images = [
            wandb.Image(
                img,
                masks={
                    "ground_truth": {
                        "mask_data": gt.numpy(),
                        "class_labels": class_labels,
                    },
                    "prediction": {
                        "mask_data": pred.numpy(),
                        "class_labels": class_labels,
                    },
                },
            )
            for img, gt, pred in zip(vis, gt_masks, preds)
        ]
        wandb_logger.experiment.log(
            {"val/predictions": wandb_images, "epoch": trainer.current_epoch}
        )
