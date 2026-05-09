from typing import Callable

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex


class SegmentationModel(L.LightningModule):
    """Lightning module for multi-class semantic segmentation.

    The `model` argument is anything that maps (B, 3, H, W) -> (B, num_classes, H, W),
    keeping this module agnostic to whether the network is a pretrained TorchVision
    head, a custom UNet, or anything else producing per-pixel logits.

    Args:
        model:             nn.Module producing per-pixel logits.
        num_classes:       Number of target classes.
        class_weights:     Per-class CE weights (e.g. median-frequency from EDA).
                           Accepts list/tuple of floats (Fiddle-friendly) or a
                           torch.Tensor; None disables weighting.
        ignore_index:      Class index excluded from loss and metrics (e.g. background).
        lr:                Learning rate.
        weight_decay:      Optimizer weight decay.
        optimizer_cls:     Torch optimizer class.
        scheduler_partial: functools.partial returning an LRScheduler given an
                           optimizer; None disables scheduling.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        class_weights: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        ignore_index: int | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.AdamW,
        scheduler_partial: Callable[..., torch.optim.lr_scheduler.LRScheduler]
        | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_cls = optimizer_cls
        self.scheduler_partial = scheduler_partial
        # register_buffer so the weights move with .to(device) and roundtrip
        # checkpoints. Accept list/tuple from configs (Fiddle-friendly: no need
        # to wrap torch.tensor, which has no inspectable signature).
        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None
        # -100 is torch's CE sentinel meaning "no class is ignored".
        self._ce_ignore = -100 if ignore_index is None else ignore_index

        metric_kwargs = {"num_classes": num_classes, "ignore_index": ignore_index}
        base = MetricCollection(
            {
                "miou": MulticlassJaccardIndex(**metric_kwargs),
                "pixel_acc": MulticlassAccuracy(**metric_kwargs, average="micro"),
            }
        )
        self.train_metrics = base.clone(prefix="train/")
        self.val_metrics = base.clone(prefix="val/")
        self.test_metrics = base.clone(prefix="test/")
        # Per-class IoU only on val — gives the imbalance picture without doubling test cost.
        self.val_iou_per_class = MulticlassJaccardIndex(**metric_kwargs, average="none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
        images, attributes = batch
        masks = attributes["mask"]
        logits = self(images)
        loss = F.cross_entropy(
            logits, masks, weight=self.class_weights, ignore_index=self._ce_ignore
        )
        preds = logits.argmax(dim=1)

        metrics: MetricCollection = getattr(self, f"{stage}_metrics")
        metrics.update(preds, masks)
        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log_dict(metrics, on_step=False, on_epoch=True)

        if stage == "val":
            self.val_iou_per_class.update(preds, masks)

        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def on_validation_epoch_end(self) -> None:
        per_class = self.val_iou_per_class.compute()
        self.log_dict(
            {f"val/iou_class_{i}": iou for i, iou in enumerate(per_class)},
            sync_dist=True,
        )
        self.val_iou_per_class.reset()

    def configure_optimizers(self):
        # filter() so frozen-backbone setups don't pass dead params to the optimizer.
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = self.optimizer_cls(
            params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.scheduler_partial is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": self.scheduler_partial(optimizer),
        }
