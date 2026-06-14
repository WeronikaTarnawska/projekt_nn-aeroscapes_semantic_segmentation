"""Class-imbalance experiment: DeepLabV3-ResNet50 + Dice+CE loss.

Identical to `deeplabv3_aeroscapes.py` EXCEPT the loss: this swaps plain weighted
cross-entropy for a Dice + weighted-CE combination (`DiceCELoss`). Keeping the
architecture, data, LR and schedule fixed makes this a controlled A/B test that
isolates the effect of the loss on the hard, under-represented classes
(obstacle / bike / animal).
"""

import fiddle as fdl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

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
    max_epochs = 30

    # ImageNet-pretrained backbone + fresh head, same as the CE baseline.
    architecture = fdl.Config(
        TorchVisionSegmentation,
        name="deeplabv3_resnet50",
        num_classes=NUM_CLASSES,
        weights=None,
        freeze_backbone=False,
    )

    data_module = fdl.Config(
        AeroScapesDataModule,
        "data/aeroscapes",
        batch_size=8,
        val_batch_size=2,
        crop_size=512,
    )

    # The class-imbalance fix. Dice is normalized per class, so rare classes
    # carry the same weight as common ones; weighted CE keeps gradients stable
    # early. ce_weight == dice_weight == 1.0 is a sensible, untuned default.
    # The criterion OWNS the class weights here, so SegmentationModel.class_weights
    # is left None to avoid double-counting (ignore_index stays on the model for
    # the metrics).
    criterion = fdl.Config(
        DiceCELoss,
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX_BACKGROUND,
        class_weights=list(MEDIAN_FREQ_WEIGHTS),
        ce_weight=1.0,
        dice_weight=1.0,
    )

    model = fdl.Config(
        SegmentationModel,
        model=architecture,
        num_classes=NUM_CLASSES,
        class_weights=None,
        ignore_index=IGNORE_INDEX_BACKGROUND,
        criterion=criterion,
        lr=1e-4,
        weight_decay=1e-4,
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
        "deeplabv3_resnet50_dice_aeroscapes",
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
