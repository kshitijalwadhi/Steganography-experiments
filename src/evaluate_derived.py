"""
Evaluate model performance specifically on cover/secret pairs derived via derive_robust_secret.py
"""
import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models import SteganoModel
from src.utils import load_config, get_device, load_checkpoint, bit_accuracy, psnr
from src.experiment_rotation import rotate_image # Reuse rotation logic
from src.datasets import get_dataloaders # To load cover images

def evaluate_derived(cfg, secrets_dir, num_secrets_to_eval):
    """Evaluates model performance on derived secrets."""
    print("--- Running Evaluation on Derived Secrets ---")
    device = get_device(cfg.training.device)
    ckpt_path = cfg.embed.checkpoint_path # Use the checkpoint specified in config

    if not ckpt_path or not Path(ckpt_path).is_file():
        raise ValueError(f"Checkpoint path '{ckpt_path}' not found or not specified.")

    secrets_dir = Path(secrets_dir)
    if not secrets_dir.is_dir():
        raise ValueError(f"Secrets directory '{secrets_dir}' not found.")

    # --- Model ---
    print(f"Loading model from {ckpt_path}...")
    # Ensure we load the same model type used for derivation (likely equivariant)
    model = SteganoModel(cfg, use_equivariant_reveal=True).to(device)
    _ = load_checkpoint(ckpt_path, model, device=device)
    model.eval()
    print("Model loaded.")

    # --- Data ---
    print(f"Loading validation data to get the first {num_secrets_to_eval} cover images...")
    # We need the exact same cover images used during derivation.
    # Assuming they were the first N images from the val set.
    _, val_loader = get_dataloaders(cfg)
    val_iter = iter(val_loader)
    cover_images = []
    derived_secrets = []
    actual_loaded_count = 0
    for i in range(num_secrets_to_eval):
        secret_path = secrets_dir / f"derived_secret_{i}.pth"
        if not secret_path.is_file():
            print(f"Warning: Secret file {secret_path} not found. Skipping pair {i}.")
            # We still need to advance the cover image iterator if we skip
            try:
                next(val_iter)
            except StopIteration:
                print("Warning: Reached end of validation set while skipping covers.")
                break
            continue

        try:
            # Load cover image
            cover_batch, _ = next(val_iter)
            cover_images.append(cover_batch[0:1].to(device))

            # Load derived secret
            secret = torch.load(secret_path, map_location='cpu').to(device)
            # Ensure secret has correct shape (Batch, Channel, H, W)
            if secret.dim() == 3: # Add channel dim if missing
                secret = secret.unsqueeze(1)
            if secret.dim() == 2: # Add batch and channel dim
                 secret = secret.unsqueeze(0).unsqueeze(0)
            derived_secrets.append(secret)
            actual_loaded_count += 1

        except StopIteration:
            print(f"Warning: Reached end of validation set after loading {actual_loaded_count} pairs.")
            break
        except Exception as e:
            print(f"Error loading cover/secret pair {i}: {e}. Skipping.")
            # Need to decide if we advance cover iterator here or not. Let's assume we do.
            try:
                next(val_iter)
            except StopIteration:
                pass
            continue

    if actual_loaded_count == 0:
        print("ERROR: Could not load any valid cover/derived secret pairs.")
        return

    print(f"Loaded {actual_loaded_count} cover/derived secret pairs for evaluation.")

    # --- Evaluation Setup ---
    rotation_angles = [0, 90, 180, 270]
    metrics = ["bit_acc"] # Focus on bit accuracy for derived secrets
    # Could add PSNR/SSIM if needed, but less relevant here

    # Dictionary to store results for each rotation angle
    results = {angle: {metric: [] for metric in metrics} for angle in rotation_angles}

    # --- Evaluation Loop ---
    with torch.no_grad():
        for i in tqdm(range(actual_loaded_count), desc="Evaluating Derived Pairs"):
            cover = cover_images[i]
            secret = derived_secrets[i]

            # Perform hiding
            prepared_secret = model.prep(secret)
            stego = model.hider(cover, prepared_secret)

            # Test each rotation angle
            for angle in rotation_angles:
                rotated_stego = rotate_image(stego, angle)
                recovered_secret = model.reveal(rotated_stego)

                # Calculate metrics
                if "bit_acc" in metrics:
                    if cfg.data.secret_type == "binary":
                        acc = bit_accuracy(recovered_secret, secret)
                        results[angle]["bit_acc"].append(acc)
                    else:
                        # Add image secret comparison if needed later
                        pass

    # --- Aggregate and Report Results ---
    print("\n--- Derived Secret Evaluation Results ---")
    for angle in rotation_angles:
        print(f"\nRotation Angle: {angle} degrees")
        for metric in metrics:
            if results[angle][metric]:
                avg_value = np.mean(results[angle][metric])
                print(f"  Average {metric.upper()}: {avg_value * 100:.2f}%" if metric == "bit_acc" else f"  Average {metric.upper()}: {avg_value:.4f}")
            else:
                print(f"  Metric {metric.upper()} not computed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on derived secrets")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config"
    )
    parser.add_argument(
        "--secrets_dir", type=str, default="./outputs/derived_secrets", help="Directory containing derived secret .pth files"
    )
    parser.add_argument(
        "--num_secrets", type=int, default=10, help="Number of derived secrets to evaluate (should match number generated)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate_derived(cfg, args.secrets_dir, args.num_secrets) 