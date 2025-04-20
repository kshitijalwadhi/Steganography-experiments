import logging
import math
import os
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

import wandb


# --- Configuration ---
def load_config(config_path="configs/default.yaml"):
    """Loads configuration from a YAML file."""
    if not Path(config_path).is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    conf = OmegaConf.load(config_path)
    # TODO: Add schema validation if needed
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(conf))
    print("---------------------")
    return conf


# --- Device Handling ---
def get_device(requested_device="auto"):
    """Gets the appropriate torch device."""
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested_device
    print(f"Using device: {device}")
    return torch.device(device)


# --- Metrics ---
def psnr(mse: torch.Tensor):
    """Calculate Peak Signal-to-Noise Ratio from MSE."""
    if mse == 0:
        return float("inf")
    return 20 * math.log10(
        1.0 / math.sqrt(mse.clamp(min=1e-10).item())
    )  # Use 20log10(MAX_I) - 10log10(MSE) where MAX_I=1


def bit_accuracy(output: torch.Tensor, target: torch.Tensor, threshold=0.5):
    """Calculate bit accuracy between predictions and targets.

    Handles both logits (binary case) and direct float values (image case) by thresholding.
    Assumes target is in [0, 1] range for images or is {0, 1} for binary.
    """
    assert output.shape == target.shape

    # Determine if output is likely logits (check range, but heuristic)
    # A safer approach is to know based on the model architecture.
    # Assuming RotationInvariantRevealNet outputs raw values (not logits/sigmoid)
    # If output values are outside [0,1], apply sigmoid first
    # For now, assume output is roughly in [0,1] as it is for MNIST target
    # If it were logits, we'd use: preds = (torch.sigmoid(output) > threshold).float()
    preds = (output > threshold).float()

    # Threshold the target as well to ensure it's binary for comparison
    target_binary = (target > threshold).float()

    return (preds == target_binary).float().mean().item()


# Placeholder for SSIM/LPIPS if needed later
# def calculate_ssim(img1, img2): ...
# def calculate_lpips(img1, img2, lpips_model): ...


# --- Image Handling ---
def load_image_tensor(image_path, size, device):
    """Loads an image, transforms it, and moves it to the device."""
    pil_img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
        ]
    )
    return transform(pil_img).unsqueeze(0).to(device)


def save_tensor_image(tensor: torch.Tensor, output_path):
    """Saves a tensor (typically B=1, C, H, W) as an image file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(tensor.squeeze(0).cpu()).save(output_path)
    print(f"Saved image to {output_path}")


# --- Checkpointing ---
def save_checkpoint(state, filepath):
    """Saves model and optimizer state."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None, device="cpu"):
    """Loads model and optionally optimizer state."""
    if not Path(filepath).is_file():
        print(f"Checkpoint not found at {filepath}, starting from scratch.")
        return 0  # Start epoch
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    start_epoch = checkpoint.get("epoch", 0) + 1  # Start from next epoch
    print(f"Loaded checkpoint from {filepath}, resuming from epoch {start_epoch}")
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Loaded optimizer state.")
    # Load other states like scheduler if needed
    return start_epoch


# --- Logging ---
def setup_wandb(cfg, job_type="train"):
    """Initializes WandB run."""
    if cfg.training.wandb.enabled:
        run_name = (
            cfg.training.wandb.run_name
            or f"{cfg.data.cover_dataset}_{cfg.data.secret_type}_ep{cfg.training.epochs}_bs{cfg.training.batch_size}"
        )
        wandb.init(
            project=cfg.training.wandb.project,
            entity=cfg.training.wandb.entity,  # Set this in your env or config
            config=OmegaConf.to_container(cfg, resolve=True),  # Log hyperparams
            name=run_name,
            job_type=job_type,
            reinit=True,  # Allows re-running in Jupyter/scripts
        )
        print(f"WandB run '{run_name}' initialized.")
    else:
        print("WandB logging disabled.")
        # You might want a dummy wandb object here if you call wandb.log unconditionally
        # class DummyWandb:
        #     def log(self, *args, **kwargs): pass
        #     def finish(self, *args, **kwargs): pass
        # return DummyWandb()
        wandb.init(mode="disabled")  # Use wandb's disabled mode


# --- Embedding/Extraction Specific Utils ---


def prepare_secret(secret_type, size, batch_size, device, secret_path=None):
    """Prepares the secret tensor based on type and config."""
    if secret_type == "binary":
        return torch.randint(0, 2, (batch_size, 1, size, size), device=device).float()
    elif secret_type == "image":
        if secret_path is None:
            raise ValueError(
                "Secret image path must be provided for secret_type='image'."
            )
        secret_tensor = load_image_tensor(secret_path, size, device)
        # Assume we use the same secret image for the whole batch for embedding
        return secret_tensor.repeat(batch_size, 1, 1, 1)
    else:
        raise ValueError(f"Unknown secret_type: {secret_type}")


# --- Placeholder for main embed/extract logic ---
# These could live here or in dedicated embed.py / extract.py files


def run_embedding(cfg, model, device):
    """Embeds a secret into a cover image."""
    print("--- Running Embedding ---")
    cover_path = cfg.embed.cover_image
    secret_path = cfg.embed.secret_image  # May be None for binary
    output_path = cfg.embed.output_image
    ckpt_path = cfg.embed.checkpoint_path
    size = cfg.data.image_size
    secret_type = cfg.data.secret_type

    if not cover_path or not Path(cover_path).is_file():
        raise FileNotFoundError(f"Cover image not found: {cover_path}")
    if ckpt_path:
        load_checkpoint(ckpt_path, model, device=device)
    else:
        print("Warning: No checkpoint specified for embedding. Using untrained model.")

    model.to(device).eval()

    cover_tensor = load_image_tensor(cover_path, size, device)
    # Prepare secret for a single image (batch size 1)
    secret_tensor = prepare_secret(secret_type, size, 1, device, secret_path)

    with torch.no_grad():
        stego_tensor, recovered_secret_logits = model(cover_tensor, secret_tensor)
        hide_loss = torch.mean((stego_tensor - cover_tensor) ** 2)  # MSE

        if secret_type == "binary":
            # Reveal loss needs sigmoid and BCE for binary
            reveal_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                recovered_secret_logits, secret_tensor
            )
            acc = bit_accuracy(recovered_secret_logits, secret_tensor)
            print(
                f"Embedding Results: PSNR={psnr(hide_loss):.2f} dB | Bit Acc={acc*100:.1f}%"
            )
        else:  # secret_type == "image"
            # Reveal loss is MSE for image secrets
            reveal_loss = torch.mean(
                (torch.sigmoid(recovered_secret_logits) - secret_tensor) ** 2
            )  # Assuming sigmoid activation for image
            print(
                f"Embedding Results: Cover PSNR={psnr(hide_loss):.2f} dB | Secret PSNR={psnr(reveal_loss):.2f} dB"
            )

    save_tensor_image(stego_tensor, output_path)
    print("-----------------------")


def run_extraction(cfg, model, device):
    """Extracts a secret from a stego image."""
    print("--- Running Extraction ---")
    stego_path = cfg.extract.stego_image
    output_path = cfg.extract.output_secret
    ckpt_path = cfg.extract.checkpoint_path
    size = cfg.data.image_size
    secret_type = cfg.data.secret_type

    if not stego_path or not Path(stego_path).is_file():
        raise FileNotFoundError(f"Stego image not found: {stego_path}")
    if ckpt_path:
        load_checkpoint(ckpt_path, model, device=device)
    else:
        print("Warning: No checkpoint specified for extraction. Using untrained model.")

    model.to(device).eval()

    stego_tensor = load_image_tensor(stego_path, size, device)

    with torch.no_grad():
        revealed_secret = model.reveal(stego_tensor)  # Get output from reveal net

    # Post-process revealed secret
    if secret_type == "binary":
        # Convert logits to binary image
        revealed_tensor = (torch.sigmoid(revealed_secret) > 0.5).float()
    else:  # secret_type == "image"
        # Apply sigmoid (assuming RevealNet outputs logits for image too)
        # Or potentially tanh if the range is [-1, 1]
        revealed_tensor = torch.sigmoid(revealed_secret)

    save_tensor_image(revealed_tensor, output_path)
    print("------------------------")
