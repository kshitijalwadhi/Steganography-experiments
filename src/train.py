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
    """Rotate image tensor by specified angle in degrees using interpolation.

    Uses NEAREST for robustness testing where appropriate, but could be adaptive.
    """
    angle = float(angle)
    # Note: Non-90 degree rotations will use interpolation (default BILINEAR)
    # This might affect results differently compared to rotating stego later.

    if angle == 0:
        return img
    else:
        # Using default bilinear interpolation for now.
        # Consider NEAREST if edges/binary data are critical and visual quality less so.
        return TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)


def get_inverse_rotation_angle(angle):
    """Calculate the inverse rotation angle."""
    # This works generally, including for non-cardinal angles.
    return -angle


def validate_epoch(model, val_loader, criterion_hide, criterion_reveal, device, cfg):
    """Runs validation on the model for one epoch with rotation testing and inverse transform."""
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0.0
    total_val_loss_hide = 0.0
    total_val_loss_reveal = 0.0  # For 0-degree rotation equivalent
    total_val_psnr = 0.0
    total_val_secret_mse = 0.0  # For 0-degree rotation equivalent after inverse transform
    total_val_secret_acc = 0.0  # For 0-degree rotation equivalent after inverse transform
    num_batches = len(val_loader)

    # Check rotation config
    use_rotation = hasattr(cfg, "rotation") and cfg.rotation.get("enabled", False)
    apply_to = cfg.rotation.get("apply_to", "stego") if use_rotation else "stego"
    rotation_angles = (
        cfg.rotation.rotation_angles
        if use_rotation
        else [0] # Only test 0 degrees if rotation is disabled
    )
    print(f"Validation: Rotation enabled={use_rotation}, Apply to={apply_to}, Angles={rotation_angles}")


    rotation_metrics = {
        angle: {"secret_mse": [], "secret_acc": []} for angle in rotation_angles
    }

    with torch.no_grad():  # Disable gradient calculations
        for cover, secret in tqdm(val_loader, desc="Validating", leave=False):
            cover, secret = cover.to(device), secret.to(device)

            # --- Calculate Hiding Loss (ALWAYS use original cover for consistency) ---
            # Generate stego image from original cover for hide loss calculation consistency
            stego_for_hideloss = model.hide_secret(cover, secret)
            loss_hide = criterion_hide(stego_for_hideloss, cover)
            total_val_loss_hide += loss_hide.item()
            total_val_psnr += psnr(loss_hide)

            # Test each rotation angle
            for angle in rotation_angles:
                # --- Prepare input for RevealNet based on rotation mode ---
                if apply_to == "cover":
                    # Rotate cover, then embed
                    rotated_cover = apply_rotation(cover, angle)
                    # Embed secret into the *rotated* cover (HideNet now expects polar coords internally)
                    stego_input = model.hide_secret(rotated_cover, secret)
                else: # apply_to == "stego" (default/original behavior)
                    # Embed into original cover first (already done as stego_for_hideloss)
                    stego_base = stego_for_hideloss # Reuse the stego generated for hide loss
                    # Rotate the resulting stego image
                    stego_input = apply_rotation(stego_base, angle)

                # --- Reveal Secret ---
                # Recover secret from the prepared input image
                rec_secret_rotated = model.reveal_secret(stego_input)

                # --- Align Revealed Secret ---
                # Apply INVERSE rotation to the recovered secret to align with original
                inv_angle = get_inverse_rotation_angle(angle)
                rec_secret_aligned = apply_rotation(rec_secret_rotated, inv_angle)

                # --- Calculate Reveal Metrics ---
                # Compare aligned recovered secret and original secret
                loss_reveal = criterion_reveal(rec_secret_aligned, secret)
                # Use MSE for logging comparison regardless of loss type used for training
                secret_mse = F.mse_loss(torch.sigmoid(rec_secret_aligned) if isinstance(criterion_reveal, nn.BCEWithLogitsLoss) else rec_secret_aligned, secret).item()
                secret_acc = bit_accuracy(rec_secret_aligned, secret)

                # Track angle-specific metrics
                rotation_metrics[angle]["secret_mse"].append(secret_mse)
                rotation_metrics[angle]["secret_acc"].append(secret_acc)

                # For 0-degree rotation, update overall validation metrics
                if angle == 0:
                    # Use the specific loss_reveal calculated for angle=0
                    loss_reveal_0_deg = criterion_reveal(rec_secret_aligned, secret)
                    total_val_loss_reveal += loss_reveal_0_deg.item()
                    total_val_secret_mse += secret_mse # Already calculated using angle 0
                    total_val_secret_acc += secret_acc # Already calculated using angle 0
                    # Overall loss uses the 0-degree reveal loss + the consistent hide loss
                    loss = (
                        cfg.training.lam_hide * loss_hide # Use consistent hide loss
                        + cfg.training.lam_reveal * loss_reveal_0_deg # Use 0-degree reveal loss
                    )
                    total_val_loss += loss.item()

    # --- Calculate Averages ---
    # Average standard metrics (based on 0-degree rotation results / consistent hide loss)
    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0
    avg_val_loss_hide = total_val_loss_hide / num_batches if num_batches > 0 else 0
    avg_val_loss_reveal = total_val_loss_reveal / num_batches if num_batches > 0 else 0
    avg_val_psnr = total_val_psnr / num_batches if num_batches > 0 else 0
    avg_val_secret_mse = total_val_secret_mse / num_batches if num_batches > 0 else 0
    avg_val_secret_acc = total_val_secret_acc / num_batches if num_batches > 0 else 0


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
        avg_val_loss_reveal,  # 0-degree reveal loss (aligned)
        avg_val_psnr,
        avg_val_secret_mse,  # 0-degree secret mse (aligned)
        avg_val_secret_acc,  # 0-degree secret acc (aligned)
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
    # Use correct reveal loss based on config (expecting binary for rotation tests)
    if cfg.data.secret_type == "binary":
        criterion_reveal = nn.BCEWithLogitsLoss()
        print("Using BCEWithLogitsLoss for reveal loss.")
    else: # Assume MSE for image secrets or others
        criterion_reveal = nn.MSELoss()
        print("Using MSELoss for reveal loss.")


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
    apply_to = cfg.rotation.get("apply_to", "stego") if use_rotation else "stego" # Default to rotating stego
    rotation_angles = cfg.rotation.rotation_angles if use_rotation else [0]
    rotation_probs = cfg.rotation.probs if use_rotation else [1.0]
    print(f"Rotation Augmentation during Training: {use_rotation}")
    if use_rotation:
        print(f"Apply rotation to: {apply_to}")
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

            # --- Apply Rotation based on config ---
            angle = 0
            if use_rotation:
                angle_idx = random.choices(
                    range(len(rotation_angles)), weights=rotation_probs, k=1
                )[0]
                angle = rotation_angles[angle_idx]

            # --- Prepare inputs and Calculate Hide Loss ---
            if use_rotation and apply_to == "cover":
                # Rotate cover *before* embedding
                rotated_cover = apply_rotation(cover, angle)
                # Generate stego for reveal path using rotated cover
                stego_input_for_reveal = model.hide_secret(rotated_cover, secret)
                # Generate stego for hide loss path using ORIGINAL cover
                # This requires a second pass through the hider but ensures hide loss is consistent.
                with torch.no_grad(): # Don't need gradients for this stego used only for loss
                    stego_for_hideloss = model.hide_secret(cover, secret)
                loss_hide = criterion_hide(stego_for_hideloss, cover) # Compare stego from orig cover to orig cover
            else:
                # Standard path (no rotation) or rotate stego *after* embedding
                stego = model.hide_secret(cover, secret) # Embed in original cover
                loss_hide = criterion_hide(stego, cover) # Hide loss vs original cover
                if use_rotation: # and apply_to == "stego"
                    stego_input_for_reveal = apply_rotation(stego, angle)
                else: # No rotation
                    stego_input_for_reveal = stego

            # --- Reveal and Align ---
            # Reveal from the prepared input (potentially rotated stego or stego from rotated cover)
            rec_secret_rotated = model.reveal_secret(stego_input_for_reveal)

            # Apply INVERSE rotation to the recovered secret before calculating reveal loss
            inv_angle = get_inverse_rotation_angle(angle)
            rec_secret_aligned = apply_rotation(rec_secret_rotated, inv_angle)

            # --- Calculate Reveal Loss ---
            loss_reveal = criterion_reveal(
                rec_secret_aligned, secret
            )  # Compare aligned recovered and original secret

            # --- Combine Losses for Backpropagation ---
            # Note: loss_hide was calculated above based on the mode, always vs original cover if apply_to==cover
            loss = (
                cfg.training.lam_hide * loss_hide
                + cfg.training.lam_reveal * loss_reveal
            )

            loss.backward()
            optimizer.step()

            # --- Logging Metrics ---
            current_loss = loss.item()
            current_loss_hide = loss_hide.item()
            current_loss_reveal = loss_reveal.item()
            current_psnr = psnr(loss_hide) # PSNR is based on the calculated loss_hide
            # Use MSE for comparable logging metric even if using BCE loss
            current_secret_mse = F.mse_loss(torch.sigmoid(rec_secret_aligned) if isinstance(criterion_reveal, nn.BCEWithLogitsLoss) else rec_secret_aligned, secret).item()
            current_secret_acc = bit_accuracy(rec_secret_aligned, secret)

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
                        "train/step_secret_mse_aligned": current_secret_mse, # Log comparable MSE
                        "train/step_secret_acc_aligned": current_secret_acc,
                        "train/rotation_angle_applied": angle,
                        "train/rotation_applied_to": apply_to if use_rotation else "none",
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
                    "RotAng": f"{angle:.0f}",
                    "RotTo": apply_to if use_rotation else "N/A"
                }
            )

        # --- Epoch End ---
        num_batches = len(train_loader)
        avg_train_loss = running_loss / num_batches if num_batches > 0 else 0
        avg_train_loss_hide = running_loss_hide / num_batches if num_batches > 0 else 0
        avg_train_loss_reveal = running_loss_reveal / num_batches if num_batches > 0 else 0
        avg_train_psnr = running_psnr / num_batches if num_batches > 0 else 0
        avg_train_secret_mse = running_secret_mse / num_batches if num_batches > 0 else 0
        avg_train_secret_acc = running_secret_acc / num_batches if num_batches > 0 else 0


        print(
            f"Epoch {epoch} Train Summary: Loss={avg_train_loss:.4f}, Hide={avg_train_loss_hide:.4f}, Reveal(Aln)={avg_train_loss_reveal:.4f}, PSNR={avg_train_psnr:.2f}, SecMSE(Aln)={avg_train_secret_mse:.4f}, SecAcc(Aln)={avg_train_secret_acc*100:.1f}%"
        )

        # --- Validation ---
        if (epoch + 1) % cfg.training.val_freq == 0:
            val_results = validate_epoch(
                model, val_loader, criterion_hide, criterion_reveal, device, cfg
            )
            (
                val_loss,           # Based on 0-deg reveal loss + consistent hide loss
                val_loss_hide,      # Consistent hide loss (orig cover vs stego from orig cover)
                val_loss_reveal,    # 0-deg reveal loss (aligned)
                val_psnr,           # Based on consistent hide loss
                val_secret_mse,     # 0-deg secret mse (aligned)
                val_secret_acc,     # 0-deg secret acc (aligned)
                rotation_metrics,   # Dict with avg_secret_mse and avg_secret_acc per angle
            ) = val_results

            # Print validation summary (based on 0-degree metrics / consistent hide loss)
            print(
                f"Epoch {epoch} Validation Summary: Loss={val_loss:.4f}, Hide={val_loss_hide:.4f}, Reveal(Aln)={val_loss_reveal:.4f}, PSNR={val_psnr:.2f}, SecMSE(Aln)={val_secret_mse:.4f}, SecAcc(Aln)={val_secret_acc*100:.1f}%"
            )

            # Calculate and print average reveal MSE/Acc across all validation rotations
            avg_rot_secret_mse = 0
            avg_rot_secret_acc = 0
            valid_angles = 0
            print(
                "Rotation-specific validation metrics (Aligned Secret MSE and Accuracy):"
            )
            if rotation_metrics:
                current_rotation_angles = list(rotation_metrics.keys())
                for angle in current_rotation_angles:
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
                "train/epoch_secret_mse_aligned": avg_train_secret_mse, # Log comparable MSE
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
            if rotation_metrics:
                current_rotation_angles = list(rotation_metrics.keys())
                for angle in current_rotation_angles:
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
                    "train/epoch_secret_mse_aligned": avg_train_secret_mse, # Log comparable MSE
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
