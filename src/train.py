import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

import wandb
from src.datasets import get_dataloaders
from src.models import SteganoModel
from src.transforms import (apply_affine_transform,
                            apply_inverse_affine_transform, get_affine_params)
from src.utils import (bit_accuracy, get_device, load_checkpoint, psnr,
                       save_checkpoint, setup_wandb)


def validate_epoch(model, val_loader, criterion_hide, criterion_reveal, device, cfg):
    """Runs validation on the model for one epoch, reporting metrics primarily for the identity transform."""
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0.0
    total_val_loss_hide = 0.0
    total_val_loss_reveal = 0.0  # Loss for identity transform (no aug)
    total_val_psnr = 0.0
    total_val_secret_mse = 0.0  # MSE for identity transform
    total_val_secret_acc = 0.0  # Accuracy for identity transform
    num_batches = len(val_loader)

    # Determine interpolation based on secret type - needed for alignment
    interp_mode = (
        InterpolationMode.NEAREST
        if cfg.data.secret_type == "binary"
        else InterpolationMode.BILINEAR
    )

    with torch.no_grad():  # Disable gradient calculations
        for cover, secret in tqdm(val_loader, desc="Validating", leave=False):
            cover, secret = cover.to(device), secret.to(device)

            # Generate stego image
            stego = model.hide_secret(cover, secret)

            # --- Calculate Metrics for Identity Transform --- #
            # No transformation applied to stego
            rec_secret_no_transform = model.reveal_secret(stego)
            # No inverse transform needed as none was applied
            rec_secret_aligned_identity = rec_secret_no_transform

            # Calculate hiding metrics (cover vs stego)
            loss_hide = criterion_hide(stego, cover)
            total_val_loss_hide += loss_hide.item()
            total_val_psnr += psnr(loss_hide)

            # Calculate reveal metrics for the identity case
            loss_reveal_identity = criterion_reveal(rec_secret_aligned_identity, secret)
            secret_mse_identity = loss_reveal_identity.item()
            secret_acc_identity = bit_accuracy(rec_secret_aligned_identity, secret)

            total_val_loss_reveal += (
                secret_mse_identity  # Use identity reveal loss for main metric
            )
            total_val_secret_mse += secret_mse_identity
            total_val_secret_acc += secret_acc_identity

            # Overall validation loss is based on identity transform
            loss = (
                cfg.training.lam_hide * loss_hide
                + cfg.training.lam_reveal * loss_reveal_identity
            )
            total_val_loss += loss.item()

            # --- Optional: Test specific fixed transformations (e.g., 90-degree rotation) --- #
            # This section could be added later if specific robustness checks are needed during validation
            # Example: angle=90, scale=1.0, translate=[0,0]
            # transformed_stego = apply_affine_transform(stego, 90, 1.0, [0,0], interp_mode)
            # rec_secret_transformed = model.reveal_secret(transformed_stego)
            # rec_secret_aligned = apply_inverse_affine_transform(rec_secret_transformed, 90, 1.0, [0,0], interp_mode)
            # loss_reveal_rot90 = criterion_reveal(rec_secret_aligned, secret)
            # ... store rot90 metrics ...

    # Calculate averages for standard metrics (based on identity transform results)
    avg_val_loss = total_val_loss / num_batches
    avg_val_loss_hide = total_val_loss_hide / num_batches
    avg_val_loss_reveal = total_val_loss_reveal / num_batches
    avg_val_psnr = total_val_psnr / num_batches
    avg_val_secret_mse = total_val_secret_mse / num_batches
    avg_val_secret_acc = total_val_secret_acc / num_batches

    # Remove the old rotation_metrics dictionary return
    return (
        avg_val_loss,
        avg_val_loss_hide,
        avg_val_loss_reveal,
        avg_val_psnr,
        avg_val_secret_mse,
        avg_val_secret_acc,
    )


def train(cfg):
    """Main training loop with affine transformation augmentation and inverse transform for loss."""
    device = get_device(cfg.training.device)
    setup_wandb(cfg, job_type="train")

    # --- Data ---
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(cfg)
    print("Data loaded.")
    image_size = cfg.data.image_size  # Get image size for transforms

    # --- Model ---
    print("Initializing model...")
    model = SteganoModel(cfg).to(device)
    print(model)
    print("Model initialized.")

    # --- Optimizer & Loss ---
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)
    criterion_hide = nn.MSELoss()
    # Determine reveal loss and interpolation based on secret type
    if cfg.data.secret_type == "binary":
        criterion_reveal = nn.BCEWithLogitsLoss()
        interp_mode = InterpolationMode.NEAREST
    elif cfg.data.secret_type == "image":
        criterion_reveal = nn.MSELoss()
        interp_mode = InterpolationMode.BILINEAR
    else:
        raise ValueError(f"Unsupported secret_type: {cfg.data.secret_type}")

    # --- Checkpointing ---
    start_epoch = 0
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_path = ckpt_dir / "latest_checkpoint.pth"
    best_ckpt_path = ckpt_dir / "best_checkpoint.pth"
    if latest_ckpt_path.is_file():
        print(f"Resuming from latest checkpoint: {latest_ckpt_path}")
        start_epoch = load_checkpoint(latest_ckpt_path, model, optimizer, device)
    elif best_ckpt_path.is_file():  # Fallback to best if latest doesn't exist
        print(f"Resuming from best checkpoint: {best_ckpt_path}")
        # Don't load optimizer state from best to potentially adapt LR
        start_epoch = load_checkpoint(
            best_ckpt_path, model, optimizer=None, device=device
        )

    # --- Setup Transformation Parameters from Config --- #
    print("Transformation Augmentation during Training:")
    print(f"- Rotation: {cfg.get('rotation', {}).get('enabled', False)}")
    print(f"- Scaling: {cfg.get('scaling', {}).get('enabled', False)}")
    print(f"- Translation: {cfg.get('translation', {}).get('enabled', False)}")

    # --- Training Loop ---
    print(f"Starting training from epoch {start_epoch}...")
    # Track best validation metric (higher accuracy / lower MSE is better)
    best_val_metric = (
        -float("inf") if cfg.data.secret_type == "binary" else float("inf")
    )
    # Define the metric to track for saving best model (accuracy for binary, mse for image)
    comparison_metric = "acc" if cfg.data.secret_type == "binary" else "mse"

    for epoch in range(start_epoch, cfg.training.epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_loss_hide = 0.0
        running_loss_reveal = 0.0
        running_psnr = 0.0
        running_secret_mse = 0.0  # Tracks MSE of aligned secret
        running_secret_acc = 0.0  # Tracks Accuracy of aligned secret

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.training.epochs}")
        for i, (cover, secret) in enumerate(pbar):
            cover, secret = cover.to(device), secret.to(device)

            optimizer.zero_grad()

            # Generate Stego Image
            stego = model.hide_secret(cover, secret)

            # --- Apply random affine transformation --- #
            angle, scale, translate_pixels = get_affine_params(cfg, image_size, device)
            transformed_stego = apply_affine_transform(
                stego, angle, scale, translate_pixels, interpolation=interp_mode
            )

            # Reveal from potentially transformed stego image
            rec_secret_transformed = model.reveal_secret(transformed_stego)

            # Apply INVERSE affine transform to the recovered secret before calculating loss
            rec_secret_aligned = apply_inverse_affine_transform(
                rec_secret_transformed,
                angle,
                scale,
                translate_pixels,
                interpolation=interp_mode,
            )

            # Calculate losses
            loss_hide = criterion_hide(stego, cover)  # Compare original cover and stego
            loss_reveal = criterion_reveal(
                rec_secret_aligned, secret
            )  # Compare aligned recovered and original secret

            # Calculate metrics for monitoring training progress (based on aligned secret)
            current_secret_mse = (
                loss_reveal.item()
                if cfg.data.secret_type == "image"
                else F.mse_loss(torch.sigmoid(rec_secret_aligned), secret).item()
            )
            current_secret_acc = bit_accuracy(rec_secret_aligned, secret)

            # Combined loss for backpropagation
            loss = (
                cfg.training.lam_hide * loss_hide
                + cfg.training.lam_reveal * loss_reveal
            )

            loss.backward()
            optimizer.step()

            # --- Logging --- #
            current_loss = loss.item()
            current_loss_hide = loss_hide.item()
            current_loss_reveal = (
                loss_reveal.item()
            )  # This is reveal loss on aligned secret
            current_psnr = psnr(loss_hide)

            running_loss += current_loss
            running_loss_hide += current_loss_hide
            running_loss_reveal += current_loss_reveal
            running_psnr += current_psnr
            running_secret_mse += current_secret_mse
            running_secret_acc += current_secret_acc

            # Log step metrics to wandb
            if i % 100 == 0:  # Log every 100 steps
                log_data = {
                    "train/step_loss": current_loss,
                    "train/step_loss_hide": current_loss_hide,
                    "train/step_loss_reveal_aligned": current_loss_reveal,
                    "train/step_psnr": current_psnr,
                    "train/step_secret_mse_aligned": current_secret_mse,
                    "train/step_secret_acc_aligned": current_secret_acc,
                    "train/applied_angle": angle,
                    "train/applied_scale": scale,
                    "train/applied_translate_x": translate_pixels[0],
                    "train/applied_translate_y": translate_pixels[1],
                    "epoch": epoch,
                    "step": epoch * len(train_loader) + i,
                }
                wandb.log(log_data)

            # Update progress bar
            pbar.set_postfix(
                {
                    "Loss": f"{current_loss:.4f}",
                    "Hide": f"{current_loss_hide:.4f}",
                    "RevAln": f"{current_loss_reveal:.4f}",
                    "PSNR": f"{current_psnr:.2f}",
                    "MSEAln": f"{current_secret_mse:.4f}",
                    "AccAln": f"{current_secret_acc*100:.1f}%",
                    "Angle": f"{angle:.1f}",
                    "Scale": f"{scale:.2f}",
                    "Trans": f"({translate_pixels[0]},{translate_pixels[1]})",
                }
            )

        # --- Epoch End --- #
        avg_train_loss = running_loss / len(train_loader)
        avg_train_loss_hide = running_loss_hide / len(train_loader)
        avg_train_loss_reveal = running_loss_reveal / len(train_loader)
        avg_train_psnr = running_psnr / len(train_loader)
        avg_train_secret_mse = running_secret_mse / len(train_loader)
        avg_train_secret_acc = running_secret_acc / len(train_loader)

        print(
            f"Epoch {epoch} Train Summary: Loss={avg_train_loss:.4f}, Hide={avg_train_loss_hide:.4f}, Reveal(Aln)={avg_train_loss_reveal:.4f}, PSNR={avg_train_psnr:.2f}, SecMSE(Aln)={avg_train_secret_mse:.4f}, SecAcc(Aln)={avg_train_secret_acc*100:.1f}%"
        )

        # Log training epoch metrics
        train_log_dict = {
            "epoch": epoch,
            "train/epoch_loss": avg_train_loss,
            "train/epoch_loss_hide": avg_train_loss_hide,
            "train/epoch_loss_reveal_aligned": avg_train_loss_reveal,
            "train/epoch_psnr": avg_train_psnr,
            "train/epoch_secret_mse_aligned": avg_train_secret_mse,
            "train/epoch_secret_acc_aligned": avg_train_secret_acc,
        }

        # --- Validation --- #
        if (epoch + 1) % cfg.training.val_freq == 0:
            val_results = validate_epoch(
                model, val_loader, criterion_hide, criterion_reveal, device, cfg
            )
            (
                val_loss,
                val_loss_hide,
                val_loss_reveal,
                val_psnr,
                val_secret_mse,  # Validation MSE (identity transform)
                val_secret_acc,  # Validation Acc (identity transform)
            ) = val_results

            # Print validation summary (based on identity transform)
            print(
                f"Epoch {epoch} Validation Summary: Loss={val_loss:.4f}, Hide={val_loss_hide:.4f}, Reveal(Ident)={val_loss_reveal:.4f}, PSNR={val_psnr:.2f}, SecMSE(Ident)={val_secret_mse:.4f}, SecAcc(Ident)={val_secret_acc*100:.1f}%"
            )

            # Log validation metrics to wandb
            val_log_dict = {
                "val/epoch_loss": val_loss,
                "val/epoch_loss_hide": val_loss_hide,
                "val/epoch_loss_reveal_identity": val_loss_reveal,
                "val/epoch_psnr": val_psnr,
                "val/epoch_secret_mse_identity": val_secret_mse,
                "val/epoch_secret_acc_identity": val_secret_acc,
            }
            wandb.log({**train_log_dict, **val_log_dict})  # Combine train and val logs

            # --- Checkpointing: Save based on BEST validation metric (identity transform) --- #
            current_metric = (
                val_secret_acc if comparison_metric == "acc" else val_secret_mse
            )
            # Check if current metric is better than best metric seen so far
            is_best = False
            if comparison_metric == "acc":  # Higher is better
                if current_metric > best_val_metric:
                    is_best = True
                    best_val_metric = current_metric
            else:  # Lower is better (MSE)
                if current_metric < best_val_metric:
                    is_best = True
                    best_val_metric = current_metric

            # Save latest checkpoint
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "best_val_metric": best_val_metric,  # Save best metric value
                },
                latest_ckpt_path,
            )

            if is_best:
                print(
                    f"New best validation {comparison_metric.upper()}: {best_val_metric:.4f}"
                )
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),  # Save optimizer with best model too
                        "config": OmegaConf.to_container(cfg, resolve=True),
                        "best_val_metric": best_val_metric,
                    },
                    best_ckpt_path,  # Save to best_checkpoint.pth
                )
        else:
            # Log only training epoch metrics if not validating this epoch
            wandb.log(train_log_dict)

    print("Training finished.")
    wandb.finish()


# Example usage (typically called from main.py):
# if __name__ == '__main__':
#     from src.utils import load_config
#     config = load_config()
#     train(config)
