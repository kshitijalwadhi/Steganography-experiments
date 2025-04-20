"""
Experiment: Rotation Invariant Steganography

This script tests the rotation invariance of the RevealNet model when stego images are rotated 
by different angles (0, 90, 180, 270 degrees). It compares the standard RevealNet with the 
rotation-invariant version implemented using escnn.
"""

import argparse
from pathlib import Path

import lpips
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm

from src.datasets import get_dataloaders
from src.evaluate import calculate_ssim
from src.models import SteganoModel, ESCNN_AVAILABLE
from src.utils import bit_accuracy, get_device, load_checkpoint, load_config, psnr


def rotate_image(img, angle):
    """Rotate image tensor by specified angle (0, 90, 180, or 270 degrees).

    Args:
        img: Image tensor of shape (B, C, H, W)
        angle: Rotation angle in degrees (0, 90, 180, or 270)

    Returns:
        Rotated image tensor of shape (B, C, H, W)
    """
    if angle == 0:
        return img
    elif angle == 90:
        return TF.rotate(img, 90)
    elif angle == 180:
        return TF.rotate(img, 180)
    elif angle == 270:
        return TF.rotate(img, 270)
    else:
        raise ValueError(f"Rotation angle must be 0, 90, 180, or 270, got {angle}")


def evaluate_model(model, data_loader, cfg, rotation_angles, metrics, lpips_model=None):
    """Evaluate model performance on rotated stego images."""
    device = next(model.parameters()).device
    
    # Dictionary to store results for each rotation angle
    results = {angle: {metric: [] for metric in metrics} for angle in rotation_angles}
    
    with torch.no_grad():
        for cover, secret in tqdm(data_loader, desc="Evaluating"):
            cover, secret = cover.to(device), secret.to(device)

            # First get the stego image from HideNet
            prepared_secret = model.prep(secret)
            stego = model.hider(cover, prepared_secret)

            # Test each rotation angle
            for angle in rotation_angles:
                rotated_stego = rotate_image(stego, angle)

                # Recover secret from rotated stego image
                recovered_secret = model.reveal(rotated_stego)

                # Calculate metrics for each sample in batch
                for i in range(cover.size(0)):
                    single_cover = cover[i : i + 1]
                    single_secret = secret[i : i + 1]
                    single_stego = stego[i : i + 1]
                    single_rotated_stego = rotated_stego[i : i + 1]
                    single_recovered_secret = recovered_secret[i : i + 1]

                    # Cover-Stego metrics
                    if "psnr" in metrics:
                        mse_hide = F.mse_loss(single_stego, single_cover)
                        results[angle]["psnr"].append(psnr(mse_hide))

                    if "ssim" in metrics:
                        ssim_val = calculate_ssim(single_stego, single_cover)
                        results[angle]["ssim"].append(ssim_val)

                    if "lpips" in metrics and lpips_model:
                        lpips_val = lpips_model(
                            single_stego * 2 - 1, single_cover * 2 - 1
                        ).item()
                        results[angle]["lpips"].append(lpips_val)

                    # Secret recovery metrics
                    if "bit_acc" in metrics and cfg.data.secret_type == "binary":
                        acc = bit_accuracy(single_recovered_secret, single_secret)
                        results[angle]["bit_acc"].append(acc)

                    if "secret_psnr" in metrics and cfg.data.secret_type == "image":
                        mse_secret = F.mse_loss(single_recovered_secret, single_secret)
                        results[angle]["secret_psnr"].append(psnr(mse_secret))
    
    return results


def experiment_rotation_invariance(cfg):
    """Compare standard and rotation-invariant models on rotated stego images."""
    if not ESCNN_AVAILABLE:
        raise ImportError("escnn package is required for this experiment")
    
    print("--- Running Rotation Invariance Experiment ---")
    device = get_device(cfg.training.device)
    ckpt_path = cfg.embed.checkpoint_path

    if not ckpt_path:
        raise ValueError(
            "Checkpoint path must be specified in config (e.g., embed.checkpoint_path) for evaluation."
        )

    # --- Data ---
    _, val_loader = get_dataloaders(cfg)
    print(f"Evaluating on {len(val_loader.dataset)} samples from validation set.")

    # --- Setup ---
    rotation_angles = [0, 90, 180, 270]
    metrics = ["psnr", "ssim"]
    if cfg.data.secret_type == "binary":
        metrics.append("bit_acc")
    else:
        metrics.append("secret_psnr")

    lpips_model = None
    if "lpips" in metrics:
        print("Loading LPIPS model...")
        lpips_model = lpips.LPIPS(net="alex").to(device)
        print("LPIPS model loaded.")
    
    # --- Evaluate standard model ---
    print("Evaluating standard (non-invariant) model...")
    # Load standard model (rotation_invariant=False)
    standard_cfg = load_config(cfg.config_path)  # Reload the original config
    standard_cfg.model.rotation_invariant = False
    standard_model = SteganoModel(standard_cfg).to(device)
    _ = load_checkpoint(ckpt_path, standard_model, device=device)
    standard_model.eval()
    
    standard_results = evaluate_model(
        standard_model, val_loader, standard_cfg, rotation_angles, metrics, lpips_model
    )
    
    # --- Evaluate invariant model ---
    print("Evaluating rotation-invariant model...")
    # Load invariant model (rotation_invariant=True)
    invariant_cfg = load_config(cfg.config_path)  # Reload the original config
    invariant_cfg.model.rotation_invariant = True
    invariant_model = SteganoModel(invariant_cfg).to(device)
    
    # Attempt to load from invariant checkpoint if specified, otherwise use standard checkpoint
    invariant_ckpt = getattr(cfg, "invariant_checkpoint_path", ckpt_path)
    _ = load_checkpoint(invariant_ckpt, invariant_model, device=device)
    invariant_model.eval()
    
    invariant_results = evaluate_model(
        invariant_model, val_loader, invariant_cfg, rotation_angles, metrics, lpips_model
    )
    
    # --- Aggregate and Report Results ---
    print("--- Rotation Invariance Experiment Results ---")
    
    # Compare standard vs. invariant models
    comparison_results = {}
    
    # Process standard model results
    for angle in rotation_angles:
        print(f"\nRotation Angle: {angle} degrees")
        print("  Standard Model:")
        for metric in metrics:
            if standard_results[angle][metric]:
                avg_value = np.mean(standard_results[angle][metric])
                comparison_results[f"standard_{angle}_{metric}"] = avg_value
                print(f"    Average {metric.upper()}: {avg_value:.4f}")
        
        print("  Invariant Model:")
        for metric in metrics:
            if invariant_results[angle][metric]:
                avg_value = np.mean(invariant_results[angle][metric])
                comparison_results[f"invariant_{angle}_{metric}"] = avg_value
                print(f"    Average {metric.upper()}: {avg_value:.4f}")
    
    # Save results to CSV
    results_df = pd.DataFrame([comparison_results])
    
    output_dir = Path(cfg.training.checkpoint_dir).parent / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"rotation_invariance_{Path(ckpt_path).stem}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved aggregated results to {results_path}")
    
    return comparison_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rotation Invariant Steganography Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--invariant_checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint for the rotation invariant model (optional)"
    )
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    cfg.config_path = args.config  # Save config path for reloading
    
    # Add invariant checkpoint path if provided
    if args.invariant_checkpoint:
        cfg.invariant_checkpoint_path = args.invariant_checkpoint

    # Run the experiment
    experiment_rotation_invariance(cfg) 