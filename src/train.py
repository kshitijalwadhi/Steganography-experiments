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
from tqdm import tqdm

import wandb
from src.datasets import get_dataloaders
from src.models import SteganoModel
from src.utils import (
    bit_accuracy,
    get_device,
    load_checkpoint,
    psnr,
    save_checkpoint,
    setup_wandb,
)


def apply_rotation(img, angle):
    """Rotate image tensor by specified angle in degrees.

    Args:
        img: Image tensor of shape (B, C, H, W)
        angle: Rotation angle in degrees (0, 90, 180, or 270)

    Returns:
        Rotated image tensor of shape (B, C, H, W)
    """
    # Angles should be multiples of 90 for this TF function
    # Ensure angle is float for TF.rotate
    angle = float(angle)
    if angle % 90 != 0:
        print(
            f"Warning: Rotation angle {angle} is not a multiple of 90. Result might be unexpected."
        )

    if angle == 0:
        return img
    else:
        # Interpolation mode might matter, NEAREST might be better for preserving edges?
        # Default is BILINEAR
        return TF.rotate(img, angle)


def get_inverse_rotation_angle(angle):
    """Calculate the inverse rotation angle for 0, 90, 180, 270 degrees."""
    if angle == 0:
        return 0
    elif angle == 90:
        return 270  # -90 degrees
    elif angle == 180:
        return 180  # -180 degrees
    elif angle == 270:
        return 90  # -270 degrees
    else:
        # Fallback for non-cardinal angles (might not be perfect inverse)
        return -angle


def validate_epoch(model, val_loader, criterion_hide, criterion_reveal, device, cfg):
    """Runs validation on the model for one epoch with rotation testing and inverse transform."""
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0.0
    total_val_loss_hide = 0.0
    total_val_loss_reveal = 0.0  # For 0-degree rotation
    total_val_psnr = 0.0
    total_val_secret_mse = 0.0  # For 0-degree rotation after inverse transform
    total_val_secret_acc = 0.0  # For 0-degree rotation after inverse transform
    num_batches = len(val_loader)

    # For tracking rotation-specific metrics
    rotation_angles = (
        cfg.rotation.rotation_angles
        if hasattr(cfg.rotation, "rotation_angles")
        else [0, 90, 180, 270]
    )
    rotation_metrics = {
        angle: {"secret_mse": [], "secret_acc": []} for angle in rotation_angles
    }

    with torch.no_grad():  # Disable gradient calculations
        for cover, secret in tqdm(val_loader, desc="Validating", leave=False):
            cover, secret = cover.to(device), secret.to(device)

            # Generate stego image once per batch
            stego = model.hide_secret(cover, secret)

            # Calculate hiding metrics (cover vs stego)
            loss_hide = criterion_hide(stego, cover)
            total_val_loss_hide += loss_hide.item()
            total_val_psnr += psnr(loss_hide)

            # Test each rotation angle
            for angle in rotation_angles:
                # Apply rotation to stego image
                rotated_stego = apply_rotation(stego, angle)

                # Recover secret from rotated stego image
                rec_secret_rotated = model.reveal_secret(rotated_stego)

                # Apply INVERSE rotation to the recovered secret
                inv_angle = get_inverse_rotation_angle(angle)
                rec_secret_aligned = apply_rotation(rec_secret_rotated, inv_angle)

                # Calculate reveal metrics (MSE) between aligned recovered secret and original secret
                loss_reveal = criterion_reveal(rec_secret_aligned, secret)
                secret_mse = loss_reveal.item()

                # Calculate Bit Accuracy
                secret_acc = bit_accuracy(rec_secret_aligned, secret)

                # Track angle-specific metrics
                rotation_metrics[angle]["secret_mse"].append(secret_mse)
                rotation_metrics[angle]["secret_acc"].append(secret_acc)

                # For 0-degree rotation, include in overall metrics (loss_reveal is already calculated)
                if angle == 0:
                    total_val_loss_reveal += loss_reveal.item()
                    total_val_secret_mse += secret_mse
                    total_val_secret_acc += secret_acc
                    # Overall loss uses the 0-degree reveal loss (after identity inverse rotation)
                    loss = (
                        cfg.training.lam_hide * loss_hide
                        + cfg.training.lam_reveal * loss_reveal
                    )
                    total_val_loss += loss.item()

    # Calculate averages for standard metrics (based on 0-degree rotation results)
    avg_val_loss = total_val_loss / num_batches
    avg_val_loss_hide = total_val_loss_hide / num_batches
    avg_val_loss_reveal = total_val_loss_reveal / num_batches
    avg_val_psnr = total_val_psnr / num_batches
    avg_val_secret_mse = total_val_secret_mse / num_batches
    avg_val_secret_acc = total_val_secret_acc / num_batches
    avg_val_acc = 0  # Not applicable for image secrets

    # Calculate averages for rotation-specific metrics
    for angle in rotation_angles:
        if rotation_metrics[angle]["secret_mse"]:
            rotation_metrics[angle]["avg_secret_mse"] = np.mean(
                rotation_metrics[angle]["secret_mse"]
            )
        else:
            rotation_metrics[angle]["avg_secret_mse"] = float("nan")
        if rotation_metrics[angle]["secret_acc"]:
            rotation_metrics[angle]["avg_secret_acc"] = np.mean(
                rotation_metrics[angle]["secret_acc"]
            )
        else:
            rotation_metrics[angle]["avg_secret_acc"] = float("nan")

    return (
        avg_val_loss,
        avg_val_loss_hide,
        avg_val_loss_reveal,  # 0-degree reveal loss
        avg_val_psnr,
        avg_val_secret_mse,  # 0-degree secret mse
        avg_val_secret_acc,  # 0-degree secret acc
        rotation_metrics,  # Dict with avg_secret_mse and avg_secret_acc per angle
    )


def train(cfg):
    """Main training loop with rotation augmentation and inverse transform for loss."""
    device = get_device(cfg.training.device)
    setup_wandb(cfg, job_type="train")

    # --- Data ---
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(cfg)
    print("Data loaded.")

    # --- Model ---
    print("Initializing model...")
    model = SteganoModel(cfg).to(device)
    print(model)
    print("Model initialized.")

    # --- Optimizer & Loss ---
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)
    criterion_hide = nn.MSELoss()
    # Use BCEWithLogitsLoss for binary secrets, as RevealNet outputs raw values (logits)
    criterion_reveal = nn.BCEWithLogitsLoss()

    # --- Checkpointing ---
    start_epoch = 0
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_path = ckpt_dir / "latest_checkpoint.pth"
    if latest_ckpt_path.is_file():
        print(f"Found latest checkpoint: {latest_ckpt_path}")
        start_epoch = load_checkpoint(latest_ckpt_path, model, optimizer, device)

    # --- Setup Rotation Parameters ---
    use_rotation = hasattr(cfg, "rotation") and cfg.rotation.get("enabled", False)
    rotation_angles = cfg.rotation.rotation_angles if use_rotation else [0]
    rotation_probs = cfg.rotation.probs if use_rotation else [1.0]
    print(f"Rotation Augmentation during Training: {use_rotation}")
    if use_rotation:
        print(f"Angles: {rotation_angles}, Probs: {rotation_probs}")

    # --- Training Loop ---
    print(f"Starting training from epoch {start_epoch}...")
    # Track best average rotation accuracy (higher is better)
    best_val_metric = -float("inf")  # Initialize for maximization

    for epoch in range(start_epoch, cfg.training.epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_loss_hide = 0.0
        running_loss_reveal = 0.0
        running_psnr = 0.0
        running_secret_mse = 0.0
        running_secret_acc = 0.0  # Track training accuracy too

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.training.epochs}")
        for i, (cover, secret) in enumerate(pbar):
            cover, secret = cover.to(device), secret.to(device)

            optimizer.zero_grad()

            # Generate Stego Image
            stego = model.hide_secret(cover, secret)

            # Apply random rotation to stego image
            angle = 0
            if use_rotation:
                angle_idx = random.choices(
                    range(len(rotation_angles)), weights=rotation_probs, k=1
                )[0]
                angle = rotation_angles[angle_idx]
                rotated_stego = apply_rotation(stego, angle)
            else:
                rotated_stego = stego  # No rotation

            # Reveal from potentially rotated stego image
            rec_secret_rotated = model.reveal_secret(rotated_stego)

            # Apply INVERSE rotation to the recovered secret before calculating loss
            inv_angle = get_inverse_rotation_angle(angle)
            rec_secret_aligned = apply_rotation(rec_secret_rotated, inv_angle)

            # Calculate losses
            loss_hide = criterion_hide(stego, cover)  # Compare original cover and stego
            loss_reveal = criterion_reveal(
                rec_secret_aligned, secret
            )  # Compare aligned recovered and original secret
            current_secret_mse = loss_reveal.item()
            # Calculate accuracy for monitoring training progress
            current_secret_acc = bit_accuracy(rec_secret_aligned, secret)

            # Combined loss for backpropagation
            loss = (
                cfg.training.lam_hide * loss_hide
                + cfg.training.lam_reveal * loss_reveal
            )

            loss.backward()
            optimizer.step()

            # --- Logging ---
            current_loss = loss.item()
            current_loss_hide = loss_hide.item()
            # Note: loss_reveal is already calculated on the aligned secret
            current_loss_reveal = loss_reveal.item()
            current_psnr = psnr(loss_hide)

            running_loss += current_loss
            running_loss_hide += current_loss_hide
            running_loss_reveal += current_loss_reveal
            running_psnr += current_psnr
            running_secret_mse += current_secret_mse
            running_secret_acc += current_secret_acc

            # Log metrics to wandb
            if i % 100 == 0:  # Log every 100 steps
                wandb.log(
                    {
                        "train/step_loss": current_loss,
                        "train/step_loss_hide": current_loss_hide,
                        "train/step_loss_reveal_aligned": current_loss_reveal,
                        "train/step_psnr": current_psnr,
                        "train/step_secret_mse_aligned": current_secret_mse,
                        "train/step_secret_acc_aligned": current_secret_acc,
                        "train/rotation_angle_applied": angle,
                        "epoch": epoch,
                        "step": epoch * len(train_loader) + i,
                    }
                )

            # Update progress bar
            pbar.set_postfix(
                {
                    "Loss": f"{current_loss:.4f}",
                    "Hide": f"{current_loss_hide:.4f}",
                    "RevAln": f"{current_loss_reveal:.4f}",  # Reveal Loss (Aligned)
                    "PSNR": f"{current_psnr:.2f}",
                    "MSEAln": f"{current_secret_mse:.4f}",  # Secret MSE (Aligned)
                    "AccAln": f"{current_secret_acc*100:.1f}%",  # Show aligned accuracy
                }
            )

        # --- Epoch End ---
        avg_train_loss = running_loss / len(train_loader)
        avg_train_loss_hide = running_loss_hide / len(train_loader)
        avg_train_loss_reveal = running_loss_reveal / len(train_loader)
        avg_train_psnr = running_psnr / len(train_loader)
        avg_train_secret_mse = running_secret_mse / len(train_loader)
        avg_train_secret_acc = running_secret_acc / len(train_loader)

        print(
            f"Epoch {epoch} Train Summary: Loss={avg_train_loss:.4f}, Hide={avg_train_loss_hide:.4f}, Reveal(Aln)={avg_train_loss_reveal:.4f}, PSNR={avg_train_psnr:.2f}, SecMSE(Aln)={avg_train_secret_mse:.4f}, SecAcc(Aln)={avg_train_secret_acc*100:.1f}%"
        )

        # --- Validation ---
        if (epoch + 1) % cfg.training.val_freq == 0:
            val_results = validate_epoch(
                model, val_loader, criterion_hide, criterion_reveal, device, cfg
            )
            (
                val_loss,  # Based on 0-deg reveal loss
                val_loss_hide,
                val_loss_reveal,  # 0-deg reveal loss (aligned)
                val_psnr,
                val_secret_mse,  # 0-deg secret mse (aligned)
                val_secret_acc,  # 0-deg secret acc (aligned)
                rotation_metrics,  # Dict with avg_secret_mse and avg_secret_acc per angle
            ) = val_results

            # Print validation summary (based on 0-degree metrics)
            print(
                f"Epoch {epoch} Validation Summary: Loss={val_loss:.4f}, Hide={val_loss_hide:.4f}, Reveal(Aln)={val_loss_reveal:.4f}, PSNR={val_psnr:.2f}, SecMSE(Aln)={val_secret_mse:.4f}, SecAcc(Aln)={val_secret_acc*100:.1f}%"
            )

            # Calculate and print average reveal MSE across all validation rotations
            avg_rot_secret_mse = 0
            avg_rot_secret_acc = 0
            valid_angles = 0
            print(
                "Rotation-specific validation metrics (Aligned Secret MSE and Accuracy):"
            )
            for angle in rotation_angles:
                avg_mse = rotation_metrics[angle].get("avg_secret_mse", float("nan"))
                avg_acc = rotation_metrics[angle].get("avg_secret_acc", float("nan"))
                print(
                    f"  Angle {angle}°: Secret MSE={avg_mse:.6f}, Secret Acc={avg_acc*100:.1f}%"
                )
                if not np.isnan(avg_mse):
                    avg_rot_secret_mse += avg_mse
                    avg_rot_secret_acc += avg_acc
                    valid_angles += 1
            if valid_angles > 0:
                avg_rot_secret_mse /= valid_angles
                avg_rot_secret_acc /= valid_angles
            print(
                f"  Average across rotations: Secret MSE={avg_rot_secret_mse:.6f}, Secret Acc={avg_rot_secret_acc*100:.1f}%"
            )

            # Log to wandb
            log_dict = {
                "epoch": epoch,
                "train/epoch_loss": avg_train_loss,
                "train/epoch_loss_hide": avg_train_loss_hide,
                "train/epoch_loss_reveal_aligned": avg_train_loss_reveal,
                "train/epoch_psnr": avg_train_psnr,
                "train/epoch_secret_mse_aligned": avg_train_secret_mse,
                "train/epoch_secret_acc_aligned": avg_train_secret_acc,
                "val/epoch_loss": val_loss,
                "val/epoch_loss_hide": val_loss_hide,
                "val/epoch_loss_reveal_aligned": val_loss_reveal,  # 0-degree aligned reveal loss
                "val/epoch_psnr": val_psnr,
                "val/epoch_secret_mse_aligned": val_secret_mse,  # 0-degree aligned secret mse
                "val/epoch_secret_acc_aligned": val_secret_acc,  # 0-degree aligned secret acc
                "val/avg_rot_secret_mse_aligned": avg_rot_secret_mse,  # Avg MSE across rotations
                "val/avg_rot_secret_acc_aligned": avg_rot_secret_acc,  # Avg Acc across rotations
            }

            # Add rotation-specific metrics to wandb
            for angle in rotation_angles:
                if "avg_secret_mse" in rotation_metrics[angle]:
                    log_dict[f"val/secret_mse_aligned_rot{angle}"] = rotation_metrics[
                        angle
                    ]["avg_secret_mse"]
                if "avg_secret_acc" in rotation_metrics[angle]:
                    log_dict[f"val/secret_acc_aligned_rot{angle}"] = rotation_metrics[
                        angle
                    ]["avg_secret_acc"]

            wandb.log(log_dict)

            # --- Checkpointing: Save based on BEST average rotation ACCURACY ---
            current_metric = avg_rot_secret_acc  # Higher is better
            is_best = current_metric > best_val_metric
            if is_best:
                best_val_metric = current_metric
                print(f"New best average rotation accuracy: {best_val_metric*100:.2f}%")
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": OmegaConf.to_container(cfg, resolve=True),
                    },
                    ckpt_dir / "best_checkpoint.pth",
                )

            # Save latest checkpoint periodically or always
            if (epoch + 1) % cfg.training.checkpoint_freq == 0 or is_best:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": OmegaConf.to_container(cfg, resolve=True),
                    },
                    latest_ckpt_path,
                )
        else:
            # Log training epoch metrics even if not validating this epoch
            wandb.log(
                {
                    "epoch": epoch,
                    "train/epoch_loss": avg_train_loss,
                    "train/epoch_loss_hide": avg_train_loss_hide,
                    "train/epoch_loss_reveal_aligned": avg_train_loss_reveal,
                    "train/epoch_psnr": avg_train_psnr,
                    "train/epoch_secret_mse_aligned": avg_train_secret_mse,
                    "train/epoch_secret_acc_aligned": avg_train_secret_acc,
                }
            )

    print("Training finished.")
    wandb.finish()


# Example usage (typically called from main.py):
# if __name__ == '__main__':
#     from src.utils import load_config
#     config = load_config()
#     train(config)
