import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image
import torchvision.utils as vutils # <<< Import for saving images
import wandb # Added wandb import

from src.models import SteganoModel
from src.utils import load_config, get_device, load_checkpoint, psnr, bit_accuracy
from src.experiment_rotation import rotate_image
from src.datasets import get_dataloaders # To get transforms and maybe a default cover


def verify_secret(model, cover_image, secret_tensor, device, cfg):
    """Verifies the robustness of the derived secret for a given cover."""
    model.eval()
    # print("\n--- Verifying Derived Secret ---") # Verbosity reduced for logging
    criterion_reveal = nn.BCEWithLogitsLoss() if cfg.data.secret_type == "binary" else nn.MSELoss()
    accuracies = {}

    with torch.no_grad():
        # Ensure cover and secret have batch dim and are on the correct device
        cover = cover_image.unsqueeze(0).to(device)
        secret = secret_tensor.unsqueeze(0).to(device) # Should be binary 0/1 already

        prepared_secret = model.prep(secret)
        stego = model.hider(cover, prepared_secret)

        for angle in [0, 90, 180, 270]:
            rotated_stego = rotate_image(stego, angle)
            revealed_rotated = model.reveal(rotated_stego)

            if cfg.data.secret_type == "binary":
                acc = bit_accuracy(revealed_rotated, secret)
                accuracies[angle] = acc * 100 # Store as percentage (acc is already float)
                # print(f"  Angle {angle}: Bit Accuracy = {accuracies[angle]:.2f}%") # Verbosity reduced
            else:
                # TODO: Implement image secret verification (e.g., PSNR)
                # print(f"  Angle {angle}: Verification for image secrets not implemented yet.") # Verbosity reduced
                accuracies[angle] = -1 # Placeholder

    return accuracies


def derive_secret(cfg, cover_image, secret_output_path, model, device, n_steps=500, lr=1e-2, save_visualization=True, log_interval=10, secret_index=0):
    """Derives a rotation-robust secret via optimization. Cover image is passed directly."""
    # Model is now passed in, already loaded and on device
    model.eval() # Ensure model is in eval mode
    for param in model.parameters(): # Ensure parameters are still frozen
        param.requires_grad = False

    # --- Loss ---
    criterion_reveal = nn.BCEWithLogitsLoss() if cfg.data.secret_type == "binary" else nn.MSELoss()

    # Cover image is already a tensor on the correct device

    # --- Initialize Learnable Secret ---
    if cfg.data.secret_type == "binary":
        secret_size = cfg.data.image_size
        secret_learnable = torch.randn(1, 1, secret_size, secret_size, device=device, requires_grad=True)
    elif cfg.data.secret_type == "image":
        secret_size = cfg.data.image_size
        secret_learnable = torch.randn(1, 3, secret_size, secret_size, device=device, requires_grad=True)
    else:
        raise ValueError(f"Unsupported secret type: {cfg.data.secret_type}")

    # --- Optimizer ---
    optimizer = optim.Adam([secret_learnable], lr=lr)

    # --- Optimization Loop ---
    pbar_desc = f"Optimizing Secret {secret_index} (Output: {Path(secret_output_path).name})"
    pbar = tqdm(range(n_steps), desc=pbar_desc, leave=False)

    # Calculate a global step offset for logging to keep steps sequential across secrets
    global_step_offset = secret_index * n_steps
    final_stego = None # Variable to store the last stego image

    for step in pbar:
        optimizer.zero_grad()
        target_secret = secret_learnable.sigmoid()
        prepared_secret = model.prep(target_secret)
        stego = model.hider(cover_image, prepared_secret) # Use passed cover_image tensor

        total_rotation_loss = torch.tensor(0.0, device=device)
        angle_losses = {}
        angles = [0, 90, 180, 270]
        for angle in angles:
            rotated_stego = rotate_image(stego, angle)
            revealed_rotated = model.reveal(rotated_stego)
            loss_angle = criterion_reveal(revealed_rotated, target_secret)
            angle_losses[angle] = loss_angle.item() # Store individual loss
            total_rotation_loss = total_rotation_loss + loss_angle

        avg_rotation_loss = total_rotation_loss / len(angles)
        avg_rotation_loss.backward(retain_graph=True)
        optimizer.step()

        # Store the last stego image for PSNR calculation later
        if step == n_steps - 1:
            final_stego = stego.detach()

        # Log loss and PSNR to wandb more frequently
        current_global_step = global_step_offset + step
        if step % log_interval == 0 or step == n_steps - 1:
            log_data = {f"secret_{secret_index}/avg_rotation_loss": avg_rotation_loss.item()}
            # Add individual angle losses to log data
            for angle, loss in angle_losses.items():
                log_data[f"secret_{secret_index}/loss_angle_{angle}"] = loss

            # Calculate and log PSNR at log_interval
            with torch.no_grad(): # Ensure PSNR calc doesn't affect gradients
                current_mse = torch.mean((cover_image - stego.detach()) ** 2)
                current_psnr = psnr(current_mse)
                log_data[f"secret_{secret_index}/stego_psnr"] = current_psnr

                # Calculate and log Verification Accuracy at log_interval
                current_target_secret = secret_learnable.sigmoid().detach()
                if cfg.data.secret_type == "binary":
                    current_verify_secret = (current_target_secret > 0.5).float()
                else: # Image (Verification for image secrets might not be fully implemented in verify_secret)
                    current_verify_secret = current_target_secret.clamp(0.0, 1.0)

                # verify_secret expects secret without batch dim and on CPU
                current_verification_results = verify_secret(
                    model, cover_image[0], current_verify_secret[0].cpu(), device, cfg
                )
                for angle, acc in current_verification_results.items():
                    log_data[f"secret_{secret_index}/verify_acc_{angle}"] = acc

            wandb.log(log_data, step=current_global_step) # Use global step for x-axis
            # Add Acc (e.g., angle 0) to progress bar if desired
            acc_0 = current_verification_results.get(0, -1)
            pbar.set_postfix({
                "Loss": f"{avg_rotation_loss.item():.4f}",
                "PSNR": f"{current_psnr:.2f}",
                "Acc0": f"{acc_0:.1f}%" # Show Acc for angle 0
            })

    # --- Final Secret Generation ---
    final_secret_continuous = secret_learnable.sigmoid().detach()

    # Log histogram of continuous secret values
    wandb.log({f"secret_{secret_index}/final_secret_value_distribution": wandb.Histogram(final_secret_continuous.cpu().numpy())}, step=current_global_step)

    if cfg.data.secret_type == "binary":
        final_secret = (final_secret_continuous > 0.5).float()
    else: # Image
        final_secret = final_secret_continuous.clamp(0.0, 1.0)

    # --- Save Secret Tensor ---
    output_path = Path(secret_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_secret.cpu(), output_path)

    # --- Save and Log Visualization (Optional) ---
    vis_path = None
    if save_visualization:
        vis_path = output_path.with_suffix('.png')
        vutils.save_image(final_secret.cpu(), vis_path, normalize=False)
        # Log visualization to wandb, associate with the final step for this secret
        wandb.log({f"secret_{secret_index}/derived_secret_visualization": wandb.Image(str(vis_path))}, step=current_global_step)

    # Return the path, final secret tensor, and FINAL verification results (re-calculate for consistency? or return last calculated? Returning last calculated for now)
    return str(output_path), final_secret, current_verification_results # Return last calculated results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Derive Multiple Rotation-Robust Secrets")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs/derived_secrets", help="Directory to save secrets and visualizations"
    )
    parser.add_argument(
        "--num_secrets", type=int, default=10, help="Number of secrets to derive"
    )
    parser.add_argument(
        "--steps", type=int, default=500, help="Optimization steps per secret"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2, help="Optimization learning rate"
    )
    parser.add_argument(
        "--log_interval", type=int, default=10, help="Frequency (steps) for logging loss during optimization"
    )
    # Wandb specific arguments
    parser.add_argument("--wandb_project", type=str, default="robust-stegano-secret-derivation", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity (username or team)")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name (defaults to auto-generated)")

    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.training.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Initialize Wandb ---
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        config={ # Log hyperparameters
            "args": vars(args),
            "config": cfg,
        }
    )

    # --- Load Model Once ---
    ckpt_path = cfg.embed.checkpoint_path
    if not ckpt_path or not Path(ckpt_path).is_file():
        # Log error and exit if checkpoint not found
        wandb.log({"error": f"Checkpoint path '{ckpt_path}' not found or not specified."})
        wandb.finish(exit_code=1)
        raise ValueError(f"Checkpoint path '{ckpt_path}' not found or not specified in config.")
    print(f"Loading model from {ckpt_path}...")
    model = SteganoModel(cfg, use_equivariant_reveal=True).to(device)
    _ = load_checkpoint(ckpt_path, model, device=device)
    model.eval()
    for param in model.parameters(): # Ensure frozen
        param.requires_grad = False
    print("Model loaded and frozen.")

    # --- Get Cover Images --- #
    print("Loading validation data to get cover images...")
    _, val_loader = get_dataloaders(cfg)
    val_iter = iter(val_loader)
    cover_images = []
    try:
        for i in range(args.num_secrets):
             # Get next batch, take first image, move to device
             cover_batch, _ = next(val_iter)
             cover_images.append(cover_batch[0:1].to(device)) # Keep batch dimension
    except StopIteration:
         warning_msg = f"WARNING: Validation set only has {len(cover_images)} images, but requested {args.num_secrets}. Deriving only {len(cover_images)} secrets."
         print(warning_msg)
         wandb.log({"warning": warning_msg})
         args.num_secrets = len(cover_images)

    if not cover_images:
        error_msg = "ERROR: Could not load any cover images from validation set."
        print(error_msg)
        wandb.log({"error": error_msg})
        wandb.finish(exit_code=1)
        exit()

    # --- Derive Secrets Loop ---
    print(f"\n--- Starting Derivation for {args.num_secrets} Secrets ---")
    derived_secret_paths = []
    all_verification_results = {} # Still useful for overall summary
    successful_derivations = 0
    for i in tqdm(range(args.num_secrets), desc="Overall Progress"):
        cover_image = cover_images[i]
        secret_output_name = f"derived_secret_{i}.pth"
        secret_output_path = output_dir / secret_output_name

        # Call the derivation function (pass model and device)
        # Note: verification_results returned are the ones from the *last* logged step now
        secret_path, derived_tensor, last_step_verification_results = derive_secret(
            cfg=cfg,
            cover_image=cover_image,
            secret_output_path=str(secret_output_path),
            model=model, # Pass loaded model
            device=device, # Pass device
            n_steps=args.steps,
            lr=args.lr,
            save_visualization=True,
            log_interval=args.log_interval, # Pass log interval from args
            secret_index=i # Pass index for logging prefix
        )
        derived_secret_paths.append(secret_path)
        all_verification_results[f"secret_{i}"] = last_step_verification_results # Store last results for summary

        # Log the derived secret tensor as a wandb artifact
        artifact = wandb.Artifact(f'derived_secret_{i}', type='secret_tensor')
        artifact.add_file(secret_path)
        wandb.log_artifact(artifact)

        successful_derivations += 1 # Assume success if it finishes

    print(f"\n--- Derivation Complete --- ")
    print(f"Successfully derived and saved {successful_derivations}/{args.num_secrets} secrets to: {output_dir}")
    print(f"Secret tensors (.pth) and visualizations (.png) saved.")

    # Log summary statistics (using the last step's verification results for each secret)
    overall_avg_acc = {}
    if successful_derivations > 0:
        for angle in [0, 90, 180, 270]:
             angle_accs = [results[angle] for results in all_verification_results.values() if angle in results and results[angle] != -1] # Handle -1 placeholder
             if angle_accs:
                 overall_avg_acc[f"overall_avg_acc_{angle}"] = sum(angle_accs) / len(angle_accs)
             else:
                 overall_avg_acc[f"overall_avg_acc_{angle}"] = -1 # No valid results for this angle
        wandb.log(overall_avg_acc)
        wandb.log({"successful_derivations": successful_derivations, "requested_secrets": args.num_secrets})


    wandb.finish() # End the wandb run
    print("Wandb logging finished.") 