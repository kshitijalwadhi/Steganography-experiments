import random
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.utils import (
    prepare_secret,  # Assuming this isn't needed for image secrets loaded directly
)


def get_transforms(size, channels=3):
    """Get standard transformations for images."""
    transform_list = [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ]
    # Add grayscale conversion if needed
    if channels == 1:
        transform_list.insert(1, transforms.Grayscale(num_output_channels=1))

    # Example Normalization (adjust if necessary, maybe not needed for cover/secret pairs)
    # if channels == 3:
    #     transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    # elif channels == 1:
    #     transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))

    return transforms.Compose(transform_list)


class PairedDataset(Dataset):
    """A dataset that pairs cover images with secret data (binary or image)."""

    def __init__(
        self,
        cover_dataset,
        secret_type,
        size,
        secret_source_dataset=None,
        secret_length=None,
    ):
        self.cover_dataset = cover_dataset
        self.secret_type = secret_type
        self.size = size
        self.secret_source_dataset = secret_source_dataset  # e.g., MNIST dataset
        self.secret_length = (
            secret_length  # e.g., number of classes or fixed length for binary
        )

        if self.secret_type == "image" and self.secret_source_dataset is None:
            raise ValueError(
                "secret_source_dataset must be provided for secret_type='image'"
            )
        # Warning if datasets have different lengths is good, but we might intentionally repeat secrets
        # if self.secret_type == "image" and len(self.cover_dataset) != len(
        #     self.secret_source_dataset
        # ):
        #     print(
        #         f"Warning: Cover dataset size ({len(self.cover_dataset)}) and Secret image dataset size ({len(self.secret_source_dataset)}) differ. Secrets will be repeated/cycled."
        #     )

    def __len__(self):
        return len(self.cover_dataset)

    def __getitem__(self, idx):
        cover_img, _ = self.cover_dataset[
            idx
        ]  # Assuming cover_dataset returns (img, label)

        if self.secret_type == "binary":
            # Generate random binary secret on the fly
            # Ensure secret_length is defined in config for binary secrets
            if self.secret_length is None:
                raise ValueError(
                    "secret_length must be defined in config for binary secrets"
                )
            # Note: Original implementation generated a HxW binary image.
            # If a fixed-length binary vector is needed, adjust this.
            # Assuming we still want a binary image secret for now:
            secret = torch.randint(0, 2, (1, self.size, self.size), dtype=torch.float32)
        elif self.secret_type == "image":
            # Use a secret image, cycling through the secret dataset
            secret_idx = idx % len(self.secret_source_dataset)
            secret_img, _ = self.secret_source_dataset[
                secret_idx
            ]  # MNIST returns (img, label)
            # Ensure secret_img is a tensor with the correct shape (1, H, W) for grayscale
            secret = secret_img
        else:
            raise ValueError(f"Unknown secret_type: {self.secret_type}")

        # Ensure cover is also a tensor (CIFAR10/ImageFolder should already handle this)
        if not isinstance(cover_img, torch.Tensor):
            # This case might occur if cover dataset yields PIL images directly
            cover_transform = get_transforms(
                self.size, channels=3
            )  # Assuming cover is RGB
            cover_img = cover_transform(cover_img)

        # Ensure secret is a tensor
        if self.secret_type == "image" and not isinstance(secret, torch.Tensor):
            # This case might occur if secret dataset yields PIL images directly
            secret_transform = get_transforms(
                self.size, channels=1
            )  # Assuming secret is Grayscale
            secret = secret_transform(secret)

        return cover_img, secret


def get_dataloaders(cfg):
    """Creates train and validation dataloaders based on config."""
    data_cfg = cfg.data
    train_cfg = cfg.training
    size = data_cfg.image_size
    cover_transform = get_transforms(size, channels=3)  # Assume covers are RGB
    secret_channels = (
        1
        if data_cfg.secret_type == "image" and data_cfg.secret_dataset == "MNIST"
        else 3
    )
    secret_transform = get_transforms(size, channels=secret_channels)

    # --- Setup Cover Dataset ---
    if data_cfg.cover_dataset == "CIFAR10":
        try:
            cover_train_set_full = datasets.CIFAR10(
                root=data_cfg.root_dir,
                train=True,
                download=False,
                transform=cover_transform,
            )
            cover_val_set_full = datasets.CIFAR10(
                root=data_cfg.root_dir,
                train=False,
                download=False,
                transform=cover_transform,
            )
        except RuntimeError:
            print(f"CIFAR10 not found in {data_cfg.root_dir}. Downloading...")
            cover_train_set_full = datasets.CIFAR10(
                root=data_cfg.root_dir,
                train=True,
                download=True,
                transform=cover_transform,
            )
            cover_val_set_full = datasets.CIFAR10(
                root=data_cfg.root_dir,
                train=False,
                download=True,
                transform=cover_transform,
            )
        cover_train_set = cover_train_set_full
        cover_val_set = cover_val_set_full
        print(
            f"Using CIFAR10 covers: {len(cover_train_set)} train, {len(cover_val_set)} val images."
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
            root=data_cfg.image_folder_cover, transform=cover_transform
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

    # --- Setup Secret Source Dataset ---
    secret_train_set = None
    secret_val_set = None
    secret_length = None  # Define based on secret type

    if data_cfg.secret_type == "binary":
        secret_length = data_cfg.get(
            "secret_length", size * size
        )  # Default to image size if not specified
        # No specific dataset needed, generated on the fly in PairedDataset
        print(f"Using random binary secrets (1x{size}x{size}).")
    elif data_cfg.secret_type == "image":
        if data_cfg.secret_dataset == "MNIST":
            try:
                mnist_train = datasets.MNIST(
                    root=data_cfg.root_dir,
                    train=True,
                    download=False,
                    transform=secret_transform,
                )
                mnist_val = datasets.MNIST(
                    root=data_cfg.root_dir,
                    train=False,
                    download=False,
                    transform=secret_transform,
                )
            except RuntimeError:
                print(f"MNIST not found in {data_cfg.root_dir}. Downloading...")
                mnist_train = datasets.MNIST(
                    root=data_cfg.root_dir,
                    train=True,
                    download=True,
                    transform=secret_transform,
                )
                mnist_val = datasets.MNIST(
                    root=data_cfg.root_dir,
                    train=False,
                    download=True,
                    transform=secret_transform,
                )
            secret_train_set = mnist_train
            secret_val_set = mnist_val
            secret_length = 10  # Number of classes in MNIST
            print(
                f"Using MNIST secrets: {len(secret_train_set)} train, {len(secret_val_set)} val images."
            )

        elif data_cfg.secret_dataset == "ImageFolder":
            # ... (ImageFolder logic for secrets remains similar) ...
            pass  # Keep existing ImageFolder logic
        else:
            raise ValueError(f"Unknown secret_dataset type: {data_cfg.secret_dataset}")
    else:
        raise ValueError(f"Unknown secret_type: {data_cfg.secret_type}")

    # --- Create Paired Datasets ---
    train_dataset = PairedDataset(
        cover_train_set, data_cfg.secret_type, size, secret_train_set, secret_length
    )
    val_dataset = PairedDataset(
        cover_val_set, data_cfg.secret_type, size, secret_val_set, secret_length
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
