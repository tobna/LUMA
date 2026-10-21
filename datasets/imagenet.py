# ---------------------------------------------------------------
# ImageNet data module for linear-probe training.
# Returns uint8 tensors [3, H, W] in [0, 255] — same convention as
# the segmentation datasets so the encoder's normalization path
# (pixel_mean / pixel_std) is shared without modification.
# ---------------------------------------------------------------

from pathlib import Path
from typing import Union

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from datasets.lightning_data_module import LightningDataModule


class ImageNetDataModule(LightningDataModule):
    """ImageNet data module for linear-probe experiments.

    Expects the dataset to be laid out as::

        <path>/
            train/
                n01440764/  (one folder per class, synset-ID or arbitrary name)
                    *.JPEG
                ...
            val/
                n01440764/
                    *.JPEG
                ...

    Images are returned as **uint8 tensors [3, H, W] in [0, 255]** — the same
    convention as the segmentation datasets — so the LinearProbeImageNet module
    can apply the encoder's own ``pixel_mean`` / ``pixel_std`` normalisation
    without any extra plumbing.

    Args:
        path:        Root directory containing ``train/`` and ``val/``.
        batch_size:  Per-GPU batch size.
        num_workers: DataLoader workers per process.
        img_size:    Crop size used for both training and evaluation.
                     Auto-linked from here to ``encoder.init_args.img_size``
                     by ``main_linear_probe.py``, so the encoder's positional
                     embeddings are resized correctly.  Default 224 works for
                     all standard ViT / Swin / ResNet backbones.
        num_classes: Number of ImageNet classes.  1000 for standard ImageNet,
                     21841 for ImageNet-21k if you probe on the full dataset.
        scale_range: (min, max) random crop scale for training augmentation.
    """

    def __init__(
        self,
        path: str,
        batch_size: int = 256,
        num_workers: int = 8,
        img_size: tuple[int, int] = (224, 224),
        num_classes: int = 1000,
        scale_range: tuple[float, float] = (0.08, 1.0),
    ) -> None:
        # ignore_idx not used for classification; set a dummy value
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            img_size=img_size,
            num_classes=num_classes,
            check_empty_targets=False,
        )
        self.save_hyperparameters(ignore=["_class_path"])
        self.scale_range = scale_range

        crop_size = img_size[0]  # assume square crop
        # Validation resize: scale the shorter side to 256 / 224 * crop_size,
        # then centre-crop to crop_size.
        resize_size = round(256 / 224 * crop_size)

        self._train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    crop_size,
                    scale=scale_range,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                # PILToTensor returns uint8 [C, H, W] in [0, 255] — no normalisation.
                transforms.PILToTensor(),
            ]
        )

        self._val_transforms = transforms.Compose(
            [
                transforms.Resize(
                    resize_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(crop_size),
                transforms.PILToTensor(),
            ]
        )

    def setup(self, stage: Union[str, None] = None) -> "ImageNetDataModule":
        self.train_dataset = ImageFolder(
            Path(self.path) / "train",
            transform=self._train_transforms,
        )
        self.val_dataset = ImageFolder(
            Path(self.path) / "val",
            transform=self._val_transforms,
        )
        return self

    # ------------------------------------------------------------------
    # Collate helpers
    # ------------------------------------------------------------------

    @staticmethod
    def train_collate(batch):
        imgs = torch.stack([img for img, _ in batch])  # [B, 3, H, W] uint8
        labels = torch.tensor([lbl for _, lbl in batch])  # [B] int64
        return imgs, labels

    @staticmethod
    def eval_collate(batch):
        imgs = torch.stack([img for img, _ in batch])
        labels = torch.tensor([lbl for _, lbl in batch])
        return imgs, labels

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
