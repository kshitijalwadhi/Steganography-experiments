import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
from src.experiment_rotation import rotate_image


def validate_epoch(model, val_loader, criterion_hide, criterion_reveal, device, cfg):
    """Runs validation on the model for one epoch."""
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0.0
    total_val_loss_hide = 0.0
    total_val_loss_reveal = 0.0
    total_val_psnr = 0.0
    total_val_acc = 0.0
    total_val_loss_rotate = 0.0
    num_batches = len(val_loader)

    with torch.no_grad():  # Disable gradient calculations
        for cover, secret in tqdm(val_loader, desc="Validating", leave=False):
            cover, secret = cover.to(device), secret.to(device)

            stego, rec_secret = model(cover, secret)

            loss_hide = criterion_hide(stego, cover)
            if cfg.data.secret_type == "binary":
                loss_reveal = criterion_reveal(rec_secret, secret)
                acc = bit_accuracy(rec_secret, secret)
            else:  # Image secret
                # Assuming reveal net outputs logits, apply sigmoid before MSE
                loss_reveal = criterion_reveal(torch.sigmoid(rec_secret), secret)
                acc = 0  # Accuracy doesn't make sense for image secrets directly

            # --- Rotation Loss Calculation (Validation) ---
            loss_rotate = 0.0
            if cfg.training.rotation_robustness.enabled:
                rotation_angles = [90, 180, 270]
                for angle in rotation_angles:
                    rotated_stego = rotate_image(stego, angle)
                    rec_secret_rotated = model.reveal(rotated_stego)
                    if cfg.data.secret_type == "binary":
                        loss_reveal_rotated = criterion_reveal(rec_secret_rotated, secret)
                    else:
                        loss_reveal_rotated = criterion_reveal(torch.sigmoid(rec_secret_rotated), secret)
                    loss_rotate += loss_reveal_rotated.item()
                if rotation_angles:
                    loss_rotate /= len(rotation_angles)
            # -----------------------------------------------

            # Calculate primary validation loss (Hide + Reveal)
            loss = (
                cfg.training.lam_hide * loss_hide
                + cfg.training.lam_reveal * loss_reveal
            )

            total_val_loss += loss.item() # This loss is now ONLY hide+reveal
            total_val_loss_hide += loss_hide.item()
            total_val_loss_reveal += loss_reveal.item()
            total_val_psnr += psnr(loss_hide)
            total_val_acc += acc
            total_val_loss_rotate += loss_rotate

    avg_val_loss = total_val_loss / num_batches
    avg_val_loss_hide = total_val_loss_hide / num_batches
    avg_val_loss_reveal = total_val_loss_reveal / num_batches
    avg_val_psnr = total_val_psnr / num_batches
    avg_val_acc = total_val_acc / num_batches
    avg_val_loss_rotate = total_val_loss_rotate / num_batches

    return (
        avg_val_loss,
        avg_val_loss_hide,
        avg_val_loss_reveal,
        avg_val_psnr,
        avg_val_acc,
        avg_val_loss_rotate,
    )


def train(cfg):
    """Main training loop."""
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
    if cfg.data.secret_type == "binary":
        criterion_reveal = nn.BCEWithLogitsLoss()
    else:  # Image secret
        criterion_reveal = nn.MSELoss()

    # --- Checkpointing ---
    start_epoch = 0
    # Try loading latest checkpoint by default
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    latest_ckpt_path = ckpt_dir / "latest_checkpoint.pth"
    if latest_ckpt_path.is_file():
        print(f"Found latest checkpoint: {latest_ckpt_path}")
        start_epoch = load_checkpoint(latest_ckpt_path, model, optimizer, device)

    # --- Training Loop ---
    print(f"Starting training from epoch {start_epoch}...")
    best_val_loss = float("inf")

    for epoch in range(start_epoch, cfg.training.epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_loss_hide = 0.0
        running_loss_reveal = 0.0
        running_psnr = 0.0
        running_acc = 0.0
        running_loss_rotate = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.training.epochs}")
        for i, (cover, secret) in enumerate(pbar):
            cover, secret = cover.to(device), secret.to(device)

            optimizer.zero_grad()

            stego, rec_secret = model(cover, secret)

            loss_hide = criterion_hide(stego, cover)
            if cfg.data.secret_type == "binary":
                loss_reveal = criterion_reveal(rec_secret, secret)
                acc = bit_accuracy(rec_secret, secret)
            else:  # Image secret
                loss_reveal = criterion_reveal(torch.sigmoid(rec_secret), secret)
                acc = 0  # Accuracy not applicable directly

            # --- Rotation Loss Calculation (Training) ---
            loss_rotate = torch.tensor(0.0, device=device)
            if cfg.training.rotation_robustness.enabled:
                rotation_angles = [90, 180, 270]
                batch_loss_rotate = 0.0
                for angle in rotation_angles:
                    # Detach stego so that Hider is not trained by rotation loss
                    rotated_stego = rotate_image(stego.detach(), angle)
                    rec_secret_rotated = model.reveal(rotated_stego)
                    if cfg.data.secret_type == "binary":
                        loss_reveal_rotated = criterion_reveal(rec_secret_rotated, secret)
                    else:
                        loss_reveal_rotated = criterion_reveal(torch.sigmoid(rec_secret_rotated), secret)
                    batch_loss_rotate = batch_loss_rotate + loss_reveal_rotated

                if rotation_angles: # Avoid division by zero if list is empty
                    loss_rotate = batch_loss_rotate / len(rotation_angles)
            # -----------------------------------------------

            # Combine losses
            loss = (
                cfg.training.lam_hide * loss_hide
                + cfg.training.lam_reveal * loss_reveal
            )
            if cfg.training.rotation_robustness.enabled:
                loss = loss + cfg.training.rotation_robustness.lam_rotate * loss_rotate

            loss.backward()
            optimizer.step()

            # --- Logging ---
            current_loss = loss.item()
            current_loss_hide = loss_hide.item()
            current_loss_reveal = loss_reveal.item()
            current_psnr = psnr(loss_hide)
            current_acc = acc
            current_loss_rotate = loss_rotate.item()

            running_loss += current_loss
            running_loss_hide += current_loss_hide
            running_loss_reveal += current_loss_reveal
            running_psnr += current_psnr
            running_acc += current_acc
            running_loss_rotate += current_loss_rotate

            # Log metrics to wandb (maybe less frequently than every step)
            if i % 100 == 0:  # Log every 100 steps
                log_dict = {
                    "train/step_loss": current_loss,
                    "train/step_loss_hide": current_loss_hide,
                    "train/step_loss_reveal": current_loss_reveal,
                    "train/step_psnr": current_psnr,
                    "train/step_accuracy": (
                        current_acc * 100 if cfg.data.secret_type == "binary" else 0
                    ),
                    "epoch": epoch,
                    "step": epoch * len(train_loader) + i,
                }
                if cfg.training.rotation_robustness.enabled:
                    log_dict["train/step_loss_rotate"] = current_loss_rotate
                wandb.log(log_dict)

            # Update progress bar
            postfix_dict = {
                "Loss": f"{current_loss:.4f}",
                "Hide": f"{current_loss_hide:.4f}",
                "Reveal": f"{current_loss_reveal:.4f}",
                "PSNR": f"{current_psnr:.2f}",
                "Acc": (
                    f"{current_acc*100:.1f}%"
                    if cfg.data.secret_type == "binary"
                    else "N/A"
                ),
            }
            if cfg.training.rotation_robustness.enabled:
                postfix_dict["RotLoss"] = f"{current_loss_rotate:.4f}"
            pbar.set_postfix(postfix_dict)

        # --- Epoch End ---
        avg_train_loss = running_loss / len(train_loader)
        avg_train_loss_hide = running_loss_hide / len(train_loader)
        avg_train_loss_reveal = running_loss_reveal / len(train_loader)
        avg_train_psnr = running_psnr / len(train_loader)
        avg_train_acc = running_acc / len(train_loader)
        avg_train_loss_rotate = running_loss_rotate / len(train_loader)

        # Print train summary (conditionally include rotation loss)
        train_summary_base = (
            f"Epoch {epoch} Train Summary: Loss={avg_train_loss:.4f}, Hide={avg_train_loss_hide:.4f}, "
            f"Reveal={avg_train_loss_reveal:.4f}, PSNR={avg_train_psnr:.2f}"
        )
        if cfg.training.rotation_robustness.enabled:
            train_summary_base += f", RotLoss={avg_train_loss_rotate:.4f}"
        if cfg.data.secret_type == "binary":
            train_summary_base += f", Acc={avg_train_acc*100:.1f}%"
        print(train_summary_base)

        # --- Validation ---
        if (epoch + 1) % cfg.training.val_freq == 0:
            val_loss, val_loss_hide, val_loss_reveal, val_psnr, val_acc, val_loss_rotate = (
                validate_epoch(
                    model, val_loader, criterion_hide, criterion_reveal, device, cfg
                )
            )
            # Print val summary (conditionally include rotation loss)
            val_summary_base = (
                f"Epoch {epoch} Validation Summary: Loss={val_loss:.4f}, Hide={val_loss_hide:.4f}, "
                f"Reveal={val_loss_reveal:.4f}, PSNR={val_psnr:.2f}"
            )
            if cfg.training.rotation_robustness.enabled:
                val_summary_base += f", RotLoss={val_loss_rotate:.4f}"
            if cfg.data.secret_type == "binary":
                val_summary_base += f", Acc={val_acc*100:.1f}%"
            print(val_summary_base)

            # Log val metrics to wandb (conditionally include rotation loss)
            val_log_dict = {
                "epoch": epoch,
                "train/epoch_loss": avg_train_loss,
                "train/epoch_loss_hide": avg_train_loss_hide,
                "train/epoch_loss_reveal": avg_train_loss_reveal,
                "train/epoch_psnr": avg_train_psnr,
                "train/epoch_accuracy": (
                    avg_train_acc * 100 if cfg.data.secret_type == "binary" else 0
                ),
                "val/epoch_loss": val_loss,
                "val/epoch_loss_hide": val_loss_hide,
                "val/epoch_loss_reveal": val_loss_reveal,
                "val/epoch_psnr": val_psnr,
                "val/epoch_accuracy": (
                    val_acc * 100 if cfg.data.secret_type == "binary" else 0
                ),
            }
            if cfg.training.rotation_robustness.enabled:
                val_log_dict["train/epoch_loss_rotate"] = avg_train_loss_rotate
                val_log_dict["val/epoch_loss_rotate"] = val_loss_rotate
            wandb.log(val_log_dict)

            # --- Checkpointing ---
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                print(f"New best validation loss: {best_val_loss:.4f}")
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": OmegaConf.to_container(
                            cfg, resolve=True
                        ),  # Save config with checkpoint
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
                    "train/epoch_loss_reveal": avg_train_loss_reveal,
                    "train/epoch_psnr": avg_train_psnr,
                    "train/epoch_accuracy": (
                        avg_train_acc * 100 if cfg.data.secret_type == "binary" else 0
                    ),
                }
            )

    print("Training finished.")
    wandb.finish()


# Example usage (typically called from main.py):
# if __name__ == '__main__':
#     from src.utils import load_config
#     config = load_config()
#     train(config)
