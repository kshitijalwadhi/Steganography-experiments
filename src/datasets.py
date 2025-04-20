import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.utils import (  # Assuming prepare_secret handles single instance generation
    prepare_secret,
)


def get_transforms(size):
    """Get standard transformations for images."""
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            # Add any other augmentations if needed
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Example
        ]
    )


class PairedDataset(Dataset):
    """A dataset that pairs cover images with secret data (binary or image)."""

    def __init__(self, cover_dataset, secret_type, size, secret_image_dataset=None):
        self.cover_dataset = cover_dataset
        self.secret_type = secret_type
        self.size = size
        self.secret_image_dataset = secret_image_dataset

        if self.secret_type == "image" and self.secret_image_dataset is None:
            raise ValueError(
                "secret_image_dataset must be provided for secret_type='image'"
            )
        if self.secret_type == "image" and len(self.cover_dataset) != len(
            self.secret_image_dataset
        ):
            print(
                f"Warning: Cover dataset size ({len(self.cover_dataset)}) and Secret image dataset size ({len(self.secret_image_dataset)}) differ. Pairing might be uneven."
            )
            # Consider repeating the smaller dataset or truncating the larger one if strict pairing is needed.

    def __len__(self):
        return len(self.cover_dataset)

    def __getitem__(self, idx):
        cover_img, _ = self.cover_dataset[
            idx
        ]  # Assuming cover_dataset returns (img, label)

        if self.secret_type == "binary":
            # Generate random binary secret on the fly
            secret = torch.randint(0, 2, (1, self.size, self.size), dtype=torch.float32)
        elif self.secret_type == "image":
            # Use corresponding secret image, handle index wrapping if datasets differ in size
            secret_idx = idx % len(self.secret_image_dataset)
            secret_img, _ = self.secret_image_dataset[secret_idx]
            secret = secret_img
        else:
            raise ValueError(f"Unknown secret_type: {self.secret_type}")

        # Ensure cover is also a tensor (CIFAR10/ImageFolder should already handle this)
        # If cover_dataset yields PIL images, apply transform here:
        # if not isinstance(cover_img, torch.Tensor):
        #    transform = get_transforms(self.size)
        #    cover_img = transform(cover_img)

        return cover_img, secret


def get_dataloaders(cfg):
    """Creates train and validation dataloaders based on config."""
    data_cfg = cfg.data
    train_cfg = cfg.training
    size = data_cfg.image_size
    transform = get_transforms(size)

    # --- Setup Cover Dataset ---
    if data_cfg.cover_dataset == "CIFAR10":
        # Use CIFAR10 train+test for cover images
        # Split manually later if needed, or use standard splits
        try:
            # Attempt download only if needed
            cover_train_set = datasets.CIFAR10(
                root=data_cfg.root_dir, train=True, download=False, transform=transform
            )
            cover_val_set = datasets.CIFAR10(
                root=data_cfg.root_dir, train=False, download=False, transform=transform
            )
        except RuntimeError:
            print(f"CIFAR10 not found in {data_cfg.root_dir}. Downloading...")
            cover_train_set = datasets.CIFAR10(
                root=data_cfg.root_dir, train=True, download=True, transform=transform
            )
            cover_val_set = datasets.CIFAR10(
                root=data_cfg.root_dir, train=False, download=True, transform=transform
            )
        print(
            f"Using CIFAR10: {len(cover_train_set)} train, {len(cover_val_set)} val images."
        )

    elif data_cfg.cover_dataset == "ImageFolder":
        if (
            not data_cfg.image_folder_cover
            or not Path(data_cfg.image_folder_cover).is_dir()
        ):
            raise FileNotFoundError(
                f"Cover image folder not found or invalid: {data_cfg.image_folder_cover}"
            )

        full_dataset = datasets.ImageFolder(
            root=data_cfg.image_folder_cover, transform=transform
        )
        # Simple 80/20 split for ImageFolder
        train_len = int(0.8 * len(full_dataset))
        val_len = len(full_dataset) - train_len
        cover_train_set, cover_val_set = torch.utils.data.random_split(
            full_dataset, [train_len, val_len]
        )
        print(
            f"Using ImageFolder {data_cfg.image_folder_cover}: {len(cover_train_set)} train, {len(cover_val_set)} val images."
        )
    else:
        raise ValueError(f"Unknown cover_dataset type: {data_cfg.cover_dataset}")

    # --- Setup Secret Dataset (if secret_type is 'image') ---
    secret_train_set = None
    secret_val_set = None
    if data_cfg.secret_type == "image":
        if (
            not data_cfg.image_folder_secret
            or not Path(data_cfg.image_folder_secret).is_dir()
        ):
            raise FileNotFoundError(
                f"Secret image folder not found or invalid: {data_cfg.image_folder_secret}"
            )

        # Use the same split ratio for secrets as covers for simplicity
        # This assumes secret images correspond to cover images if split this way
        # A more robust approach might require pre-defined train/val splits for secret images
        full_secret_dataset = datasets.ImageFolder(
            root=data_cfg.image_folder_secret, transform=transform
        )
        if len(full_secret_dataset) != len(cover_train_set) + len(cover_val_set):
            print(
                f"Warning: Total cover images ({len(cover_train_set) + len(cover_val_set)}) != total secret images ({len(full_secret_dataset)}). Ensure pairing logic is intended."
            )

        # Use the same indices from cover split if possible, otherwise re-split (less ideal)
        if isinstance(cover_train_set, torch.utils.data.Subset):
            secret_train_set = torch.utils.data.Subset(
                full_secret_dataset, cover_train_set.indices
            )
            secret_val_set = torch.utils.data.Subset(
                full_secret_dataset, cover_val_set.indices
            )
        else:  # CIFAR10 case, need a separate split strategy for secrets
            train_len = int(0.8 * len(full_secret_dataset))
            val_len = len(full_secret_dataset) - train_len
            secret_train_set, secret_val_set = torch.utils.data.random_split(
                full_secret_dataset, [train_len, val_len]
            )
        print(
            f"Using Secret ImageFolder {data_cfg.image_folder_secret}: {len(secret_train_set)} train, {len(secret_val_set)} val images."
        )

    # --- Create Paired Datasets ---
    train_dataset = PairedDataset(
        cover_train_set, data_cfg.secret_type, size, secret_train_set
    )
    val_dataset = PairedDataset(
        cover_val_set, data_cfg.secret_type, size, secret_val_set
    )

    # --- Create DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=True,  # Helps speed up CPU->GPU transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.evaluation.batch_size,  # Use evaluation batch size for val
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
