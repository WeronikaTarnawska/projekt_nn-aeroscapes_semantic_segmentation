"""Best-mIoU experiment: DeepLabV3-ResNet101, COCO-pretrained + Dice+CE.

Stacks the ingredients that move overall mIoU the most on this dataset:

  * a deeper backbone (ResNet101 vs ResNet50),
  * full COCO-pretrained weights (`weights="DEFAULT"`) rather than only an
    ImageNet backbone — the segmentation head starts from real segmentation
    features and the final conv is surgically resized to 12 classes,
  * the Dice + weighted-CE loss (also helps the rare classes), and
  * a cosine LR schedule for a cleaner late-training descent.

Heavier than the baseline: ResNet101 at crop 512 fits ~12 GB at batch 4. Raise
`crop_size` / `batch_size` if you have more VRAM — higher resolution is the
single biggest lever left for the small classes.
"""

import fiddle as fdl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config.aeroscapes_constants import (
    CLASS_NAMES,
    IGNORE_INDEX_BACKGROUND,
    MEDIAN_FREQ_WEIGHTS,
    NUM_CLASSES,
)
from src.config.constants import WANDB_ENTITY, WANDB_PROJECT
from src.config.schemas import ExperimentConfig, TrainingConfig
from src.datasets.aeroscapes import AeroScapesDataModule
from src.models.architectures.torchvision_segmentation import TorchVisionSegmentation
from src.models.losses import DiceCELoss
from src.models.segmentation_model import SegmentationModel
from src.utils.callbacks import LogPredictionsCallback


def build_config() -> fdl.Config[ExperimentConfig]:
    max_epochs = 40

    # weights="DEFAULT": full COCO-pretrained DeepLabV3; the classifier head is
    # resized from 21 -> NUM_CLASSES by TorchVisionSegmentation.
    architecture = fdl.Config(
        TorchVisionSegmentation,
        name="deeplabv3_resnet101",
        num_classes=NUM_CLASSES,
        weights="DEFAULT",
        freeze_backbone=False,
    )

    # crop 512 @ batch 4 keeps ResNet101 within ~12 GB. Validation at native
    # 1280x720 (eval_size=None) for paper-comparable mIoU.
    data_module = fdl.Config(
        AeroScapesDataModule,
        "data/aeroscapes",
        batch_size=4,
        val_batch_size=1,
        crop_size=512,
        # RTX 3060 box has ~4 usable CPU cores; 8 workers oversubscribe and can
        # stall the loader. 4 matches the hardware.
        num_workers=4,
    )

    criterion = fdl.Config(
        DiceCELoss,
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX_BACKGROUND,
        class_weights=list(MEDIAN_FREQ_WEIGHTS),
        ce_weight=1.0,
        dice_weight=1.0,
    )

    # Cosine anneal over the full run; SegmentationModel calls
    # scheduler_partial(optimizer) and Lightning steps it per epoch.
    scheduler = fdl.Partial(CosineAnnealingLR, T_max=max_epochs)

    model = fdl.Config(
        SegmentationModel,
        model=architecture,
        num_classes=NUM_CLASSES,
        class_weights=None,
        ignore_index=IGNORE_INDEX_BACKGROUND,
        criterion=criterion,
        lr=1e-4,
        weight_decay=1e-4,
        scheduler_partial=scheduler,
    )

    wandb_logger = fdl.Partial(
        WandbLogger,
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
    )

    checkpoints_callback = fdl.Partial(
        ModelCheckpoint,
        monitor="val/miou",
        mode="max",
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
    )

    return fdl.Config(
        ExperimentConfig,
        "deeplabv3_resnet101_best_aeroscapes",
        model,
        data_module,
        training_cfg=fdl.Config(
            TrainingConfig,
            wandb_logger,
            checkpoints_callback,
            max_epochs,
            callbacks=[
                fdl.Config(
                    LogPredictionsCallback,
                    num_samples=8,
                    class_names=CLASS_NAMES,
                ),
            ],
        ),
    )
