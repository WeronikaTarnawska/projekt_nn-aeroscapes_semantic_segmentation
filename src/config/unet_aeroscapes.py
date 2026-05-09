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
from src.models.architectures.unet import UNet
from src.models.segmentation_model import SegmentationModel
from src.utils.callbacks import LogPredictionsCallback


def build_config() -> fdl.Config[ExperimentConfig]:
    max_epochs = 50

    # 4 levels → spatial dim must divide by 2**3 = 8. crop_size=512 is fine.
    architecture = fdl.Config(
        UNet,
        in_channels=3,
        num_classes=NUM_CLASSES,
        channel_dims=[64, 128, 256, 512],
    )

    # Train on 512 random crops; evaluate at native 1280x720 (which also divides
    # cleanly by 8) for paper-comparable mIoU.
    data_module = fdl.Config(
        AeroScapesDataModule,
        "data/aeroscapes",
        batch_size=8,
        val_batch_size=2,
        crop_size=512,
    )

    # Higher LR than the fine-tuning config — training from scratch needs more push.
    # class_weights: median-frequency from EDA. Plain list (Fiddle-friendly);
    # SegmentationModel converts to tensor internally.
    # ignore_index=0 drops the semantically-empty background from loss & metrics.
    model = fdl.Config(
        SegmentationModel,
        model=architecture,
        num_classes=NUM_CLASSES,
        class_weights=list(MEDIAN_FREQ_WEIGHTS),
        ignore_index=IGNORE_INDEX_BACKGROUND,
        lr=1e-3,
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
        "unet_aeroscapes_scratch",
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
