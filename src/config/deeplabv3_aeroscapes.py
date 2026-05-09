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
from src.models.segmentation_model import SegmentationModel
from src.utils.callbacks import LogPredictionsCallback


def build_config() -> fdl.Config[ExperimentConfig]:
    max_epochs = 30

    # weights=None: ImageNet-pretrained backbone + fresh head sized to NUM_CLASSES.
    # Switch to weights="DEFAULT" to start from full COCO-pretrained model
    # (final conv gets surgically swapped). Flip freeze_backbone for the
    # frozen-feature baseline used in the assignment.
    architecture = fdl.Config(
        TorchVisionSegmentation,
        name="deeplabv3_resnet50",
        num_classes=NUM_CLASSES,
        weights=None,
        freeze_backbone=False,
    )

    # crop_size=512: square random crops keep small classes (person/bike/obstacle)
    # at native pixel scale while still allowing batch>1. Validation runs at
    # native 1280x720 (eval_size=None) for paper-comparable mIoU.
    data_module = fdl.Config(
        AeroScapesDataModule,
        "data/aeroscapes",
        batch_size=8,
        val_batch_size=2,
        crop_size=512,
    )

    # class_weights: median-frequency from EDA (notebooks/eda.ipynb) — keeps the
    # loss from being dominated by vegetation/road/background. Plain list of
    # floats: SegmentationModel converts to tensor internally (Fiddle-friendly,
    # avoids wrapping torch.tensor which has no inspectable signature).
    # ignore_index=0: background is ~23% of pixels but carries no semantic info,
    # so exclude it from loss AND metrics rather than just downweighting it.
    model = fdl.Config(
        SegmentationModel,
        model=architecture,
        num_classes=NUM_CLASSES,
        class_weights=list(MEDIAN_FREQ_WEIGHTS),
        ignore_index=IGNORE_INDEX_BACKGROUND,
        lr=1e-4,
        weight_decay=1e-4,
    )

    wandb_logger = fdl.Partial(
        WandbLogger,
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
    )

    # save_last=True is required by train_model.py's resume mechanism (it looks
    # for last.ckpt first). monitor="val/miou" picks the best checkpoint by mIoU.
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
        "deeplabv3_resnet50_aeroscapes",
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
