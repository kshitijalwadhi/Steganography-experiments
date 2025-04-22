import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image
import torchvision.utils as vutils # <<< Import for saving images

from src.models import SteganoModel
from src.utils import load_config, get_device, load_checkpoint, psnr, bit_accuracy
from src.experiment_rotation import rotate_image
from src.datasets import get_dataloaders # To get transforms and maybe a default cover


def verify_secret(model, cover_image, secret_tensor, device, cfg):
    """Verifies the robustness of the derived secret for a given cover."""
    model.eval()
    print("\n--- Verifying Derived Secret ---")
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
                accuracies[angle] = acc * 100 # Store as percentage
                print(f"  Angle {angle}: Bit Accuracy = {acc * 100:.2f}%")
            else:
                # TODO: Implement image secret verification (e.g., PSNR)
                print(f"  Angle {angle}: Verification for image secrets not implemented yet.")
                accuracies[angle] = -1 # Placeholder

    return accuracies


def derive_secret(cfg, cover_image, secret_output_path, n_steps=500, lr=1e-2, save_visualization=True):
    """Derives a rotation-robust secret via optimization. Cover image is passed directly."""
    # Note: Removed print statements about loading model/cover, assuming they are handled by the calling loop.
    device = get_device(cfg.training.device)
    ckpt_path = cfg.embed.checkpoint_path

    # --- Model --- (Assume model is passed in or loaded once outside the loop? For now, load each time)
    model = SteganoModel(cfg, use_equivariant_reveal=True).to(device) # Ensure equivariant is used
    _ = load_checkpoint(ckpt_path, model, device=device)
    model.eval()
    for param in model.parameters():
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
    # Can potentially reduce verbosity here if called in a loop
    pbar_desc = f"Optimizing Secret (Output: {Path(secret_output_path).name})"
    pbar = tqdm(range(n_steps), desc=pbar_desc, leave=False) # Use leave=False in loop

    for step in pbar:
        optimizer.zero_grad()
        target_secret = secret_learnable.sigmoid()
        prepared_secret = model.prep(target_secret)
        stego = model.hider(cover_image, prepared_secret) # Use passed cover_image tensor

        total_rotation_loss = torch.tensor(0.0, device=device)
        angles = [0, 90, 180, 270]
        for angle in angles:
            rotated_stego = rotate_image(stego, angle)
            revealed_rotated = model.reveal(rotated_stego)
            loss_angle = criterion_reveal(revealed_rotated, target_secret)
            total_rotation_loss = total_rotation_loss + loss_angle

        avg_rotation_loss = total_rotation_loss / len(angles)
        avg_rotation_loss.backward(retain_graph=True)
        optimizer.step()

        if step % 100 == 0 or step == n_steps - 1:
             pbar.set_postfix({"Loss": f"{avg_rotation_loss.item():.4f}"})

    # --- Final Secret Generation ---
    final_secret_continuous = secret_learnable.sigmoid().detach()
    if cfg.data.secret_type == "binary":
        final_secret = (final_secret_continuous > 0.5).float()
    else: # Image
        final_secret = final_secret_continuous.clamp(0.0, 1.0)

    # --- Save Secret Tensor ---
    output_path = Path(secret_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_secret.cpu(), output_path)
    # print(f"Saved derived secret tensor to: {output_path}") # Reduce print verbosity

    # --- Save Visualization (Optional) ---
    if save_visualization:
        vis_path = output_path.with_suffix('.png')
        # Ensure tensor is in range [0, 1] and has channel dim if needed
        # For binary, final_secret is already 0 or 1, shape (1, 1, H, W)
        # For image, final_secret is (1, 3, H, W)
        vutils.save_image(final_secret.cpu(), vis_path, normalize=False) # Don't normalize if already 0/1
        # print(f"Saved secret visualization to: {vis_path}") # Reduce print verbosity

    # --- Verification ---
    # print(f"Verifying secret: {output_path.name}")
    # verification_results = verify_secret(model, cover_image[0], final_secret[0], device, cfg)
    # Return the final secret tensor along with the path
    return str(output_path), final_secret


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
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.training.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Model Once --- # Optimization: Load model once if memory allows
    ckpt_path = cfg.embed.checkpoint_path
    if not ckpt_path or not Path(ckpt_path).is_file():
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
        for _ in range(args.num_secrets):
             # Get next batch, take first image, move to device
             cover_batch, _ = next(val_iter)
             cover_images.append(cover_batch[0:1].to(device))
    except StopIteration:
         print(f"WARNING: Validation set only has {len(cover_images)} images, but requested {args.num_secrets}. Deriving only {len(cover_images)} secrets.")
         args.num_secrets = len(cover_images)

    if not cover_images:
        print("ERROR: Could not load any cover images from validation set.")
        exit()

    # --- Derive Secrets Loop ---
    print(f"\n--- Starting Derivation for {args.num_secrets} Secrets ---")
    derived_secret_paths = []
    successful_derivations = 0
    for i in tqdm(range(args.num_secrets), desc="Overall Progress"):
        cover_image = cover_images[i]
        secret_output_path = output_dir / f"derived_secret_{i}.pth"

        # Call the derivation function (which now loads model internally, could be optimized)
        secret_path, derived_tensor = derive_secret(
            cfg,
            cover_image=cover_image,
            secret_output_path=str(secret_output_path),
            n_steps=args.steps,
            lr=args.lr,
            save_visualization=True
        )
        derived_secret_paths.append(secret_path)

        # Optional: Run verification immediately after each derivation
        print(f"\nVerifying secret {i} (Path: {secret_path})")
        verify_secret(model, cover_image[0], derived_tensor[0].cpu(), device, cfg)
        # We could add a check here to count successful ones if needed
        successful_derivations += 1 # Assume success if it finishes

    print(f"\n--- Derivation Complete --- ")
    print(f"Successfully derived and saved {successful_derivations}/{args.num_secrets} secrets to: {output_dir}")
    print(f"Secret tensors (.pth) and visualizations (.png) saved.") 