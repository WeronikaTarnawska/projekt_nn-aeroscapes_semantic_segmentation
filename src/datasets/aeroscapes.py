from pathlib import Path
from typing import Callable

import lightning as L
import torch
import torchvision.transforms.functional as TF
import torchvision.tv_tensors as tv_tensors
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


class AeroScapesDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        size: tuple[int, int] = (512, 512),
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.size = size
        self.transform = transform

        self.img_dir = self.data_dir / "JPEGImages"
        self.mask_dir = self.data_dir / "SegmentationClass"
        
        self._image_list = sorted(list(self.img_dir.glob("*.jpg")))

    def __len__(self) -> int:
        return len(self._image_list)

    @staticmethod
    def _normalize(img: torch.Tensor) -> torch.Tensor:
        return img.float() / 127.5 - 1.0

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        img_path = self._image_list[idx]
        mask_path = self.mask_dir / img_path.with_suffix(".png").name

        pil_img = Image.open(img_path).convert("RGB")
        pil_mask = Image.open(mask_path)

        img = tv_tensors.Image(TF.pil_to_tensor(pil_img))
        mask = tv_tensors.Mask(TF.pil_to_tensor(pil_mask))

        # Resizing
        # BILINEAR - when resizing an image, the library calculates the average of the surrounding pixels
        img = TF.resize(img, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        # NEAREST - when resizing a mask, the library takes the value of the nearest pixel without averaging, for keeping class IDs intact
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)

        if self.transform is not None:
            img = self.transform(img)
            # If you're applying an effect, you must apply it to the mask as well!

        # Squeeze removes the unnecessary channel dimension from the mask
        return self._normalize(img), {"mask": mask.squeeze().long()}


class AeroScapesDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 8, # a smaller batch due to the large images
        size: tuple[int, int] = (512, 512),
        val_split: float = 0.1,
        test_split: float = 0.2,
    ):
        super().__init__()
        self.batch_size = batch_size

        full_dataset = AeroScapesDataset(data_dir, size=size)
        n = len(full_dataset)
        n_test = int(n * test_split)
        n_val = int(n * val_split)
        n_train = n - n_val - n_test
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42) # for reproducibility
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    