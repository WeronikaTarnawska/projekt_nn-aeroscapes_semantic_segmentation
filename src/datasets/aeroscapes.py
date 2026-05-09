from pathlib import Path
from typing import Literal

import lightning as L
import numpy as np
import torch
import torchvision.transforms.v2 as v2
import torchvision.transforms.v2.functional as v2F
import torchvision.tv_tensors as tv_tensors
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.config.aeroscapes_constants import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
)

# Official Aeroscapes split files. EDA confirmed an exact 80/20 train/val ratio
# (2621/648), matching the paper's reported split. We do NOT use random_split:
# the paper warns that frames from the same video sequence must not straddle the
# train/test boundary, and the official files already enforce this.
_SPLIT_FILES = {"train": "trn.txt", "val": "val.txt"}
_NATIVE_SIZE = (SOURCE_HEIGHT, SOURCE_WIDTH)  # (H, W) — Resize accepts this order


class AeroScapesDataset(Dataset):
    """Aeroscapes semantic segmentation, official `ImageSets/{trn,val}.txt` splits.

    Returns ``(image, {"mask": mask})`` to match the contract expected by
    ``SegmentationModel`` (which destructures ``attributes["mask"]``). Image is a
    normalized float tensor (C, H, W); mask is an int64 tensor (H, W) of class
    indices in 0..11.

    Joint geometric transforms (image AND mask) rely on torchvision v2 +
    tv_tensors so that `Resize`, `RandomCrop`, `RandomHorizontalFlip` etc.
    dispatch the right interpolation per type (NEAREST for `Mask`, BILINEAR for
    `Image`) and apply the same random parameters to both.

    Args:
        data_dir:   Path to the unpacked `aeroscapes/` directory (containing
                    `JPEGImages/`, `SegmentationClass/`, `ImageSets/`).
        split:      Which official split to load — `"train"` or `"val"`.
        transforms: Joint v2 transform pipeline. See `build_train_transforms` /
                    `build_eval_transforms` for ready-made defaults.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: Literal["train", "val"] = "train",
        transforms: v2.Transform | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / "JPEGImages"
        self.mask_dir = self.data_dir / "SegmentationClass"
        self.transforms = transforms

        split_file = self.data_dir / "ImageSets" / _SPLIT_FILES[split]
        self.stems = split_file.read_text().split()

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self.stems[idx]
        pil_img = Image.open(self.img_dir / f"{stem}.jpg").convert("RGB")
        # Mask mode is 'L' and pixel values ARE class indices 0..11 (verified in EDA).
        # No color->class decoding, no 255 ignore label to mask out.
        pil_mask = Image.open(self.mask_dir / f"{stem}.png")

        # pil_to_tensor produces CHW (3, H, W) uint8 — the layout v2 transforms
        # require. Going through np.array would yield HWC and crash on Normalize
        # / RandomResizedCrop's channel-axis assumptions.
        img = tv_tensors.Image(v2F.pil_to_tensor(pil_img))
        mask = tv_tensors.Mask(np.array(pil_mask))

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, {"mask": mask.long()}


def build_train_transforms(crop_size: int | None = None) -> v2.Transform:
    """Train pipeline: optional random scaled crop, geometric & photometric
    augmentation, then ImageNet normalization.

    `crop_size=None` follows the paper (train at native 1280x720, batch=1). Set
    an int to enable random crops + larger batches — EDA showed median 6 classes
    per image, so crops down to ~256 still see meaningful class variety.

    `RandomResizedCrop` (rather than plain `RandomCrop`) adds scale jitter, which
    matters here: drone altitude varies 5–50m, so the same object can appear at
    very different pixel scales across images. Without this the model would only
    see each object at its native pixel size and generalize worse across
    altitudes.

    `hue=0.0` deliberately — for aerial scenes, hue shifts can blur class
    boundaries that depend strongly on color (sky/vegetation/water). Brightness
    and contrast still help discourage backdrop shortcuts flagged by the
    co-occurrence analysis (boat↔water, drone↔sky, animal↔construction).
    """
    steps: list = []
    if crop_size is not None:
        # scale fraction of input AREA — torchvision requires max ≤ 1.0.
        # (0.4, 1.0) ≈ 1.6× linear zoom range, supplementing the natural
        # altitude-driven scale variation already present in the dataset.
        steps.append(v2.RandomResizedCrop(crop_size, scale=(0.4, 1.0), antialias=True))
    steps += [
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return v2.Compose(steps)


def build_eval_transforms(size: tuple[int, int] | None = None) -> v2.Transform:
    """Validation/test pipeline: deterministic resize + ImageNet normalize.

    `size=None` keeps native 1280x720. v2 dispatches per type — BILINEAR for the
    image, NEAREST for the mask — so class indices are never blended.
    """
    steps: list = []
    if size is not None and tuple(size) != _NATIVE_SIZE:
        steps.append(v2.Resize(size))
    steps += [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return v2.Compose(steps)


class AeroScapesDataModule(L.LightningDataModule):
    """Lightning DataModule wrapping the official Aeroscapes train/val splits.

    Defaults follow the paper: native 1280x720, no crop. Set `crop_size` for
    larger training batches at the cost of some context.

    Aeroscapes ships only `trn.txt` and `val.txt` — there is no canonical test
    split, so `test_dataloader` is intentionally not provided. Add a hold-out
    from `val.txt` later if needed.

    Args:
        data_dir:        Path to the unpacked `aeroscapes/` directory.
        batch_size:      Training batch size.
        val_batch_size:  Validation batch size; defaults to `batch_size`. Useful
                         to set lower when evaluating at native resolution.
        crop_size:       If set, train on random `crop_size x crop_size` crops.
                         If None, train on native 1280x720 (matches the paper).
        eval_size:       Validation resize target as `(H, W)`. None = native.
        num_workers:     DataLoader workers per process.
    """

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 4,
        val_batch_size: int | None = None,
        crop_size: int | None = None,
        eval_size: tuple[int, int] | None = None,
        num_workers: int = 8,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.val_batch_size = (
            val_batch_size if val_batch_size is not None else batch_size
        )
        self.num_workers = num_workers
        self._train_tf = build_train_transforms(crop_size=crop_size)
        self._eval_tf = build_eval_transforms(size=eval_size)
        self.train_dataset: AeroScapesDataset | None = None
        self.val_dataset: AeroScapesDataset | None = None

    def prepare_data(self) -> None:
        # Single-process check that the dataset is on disk. Download itself is
        # handled out-of-band by scripts/download_data.py — this just fails fast
        # with a clearer message than a missing-file error deep in __getitem__.
        if not (self.data_dir / "ImageSets" / "trn.txt").exists():
            raise FileNotFoundError(
                f"Aeroscapes split file not found under {self.data_dir}. "
                "Run scripts/download_data.py first."
            )

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit", "validate"):
            self.train_dataset = AeroScapesDataset(
                self.data_dir, "train", self._train_tf
            )
            self.val_dataset = AeroScapesDataset(self.data_dir, "val", self._eval_tf)

    def _loader(
        self, dataset: Dataset, *, batch_size: int, shuffle: bool
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, "call setup() before train_dataloader()"
        return self._loader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None, "call setup() before val_dataloader()"
        return self._loader(
            self.val_dataset, batch_size=self.val_batch_size, shuffle=False
        )
