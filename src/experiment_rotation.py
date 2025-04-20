"""
Experiment: Stego Image Rotation Robustness

This script tests how rotating the stego image (output of HideNet, input to RevealNet)
affects the model's ability to recover the hidden secret.
Rotations of 0, 90, 180, and 270 degrees are tested.
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
from src.models import SteganoModel
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


def experiment_rotation(cfg):
    """Evaluates model performance when stego images are rotated."""
    print("--- Running Rotation Experiment ---")
    device = get_device(cfg.training.device)
    ckpt_path = cfg.embed.checkpoint_path

    if not ckpt_path:
        raise ValueError(
            "Checkpoint path must be specified in config (e.g., embed.checkpoint_path) for evaluation."
        )

    # --- Data ---
    _, val_loader = get_dataloaders(cfg)
    print(f"Evaluating on {len(val_loader.dataset)} samples from validation set.")

    # --- Model ---
    model = SteganoModel(cfg).to(device)
    _ = load_checkpoint(ckpt_path, model, device=device)
    model.eval()

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

    # Dictionary to store results for each rotation angle
    results = {angle: {metric: [] for metric in metrics} for angle in rotation_angles}

    # --- Evaluation Loop ---
    with torch.no_grad():
        for cover, secret in tqdm(val_loader, desc="Evaluating"):
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

    # --- Aggregate and Report Results ---
    print("--- Rotation Experiment Results ---")
    aggregated_results = {}

    for angle in rotation_angles:
        print(f"\nRotation Angle: {angle} degrees")
        for metric in metrics:
            if results[angle][metric]:
                avg_value = np.mean(results[angle][metric])
                aggregated_results[f"{angle}_{metric}"] = avg_value
                print(f"  Average {metric.upper()}: {avg_value:.4f}")
            else:
                print(
                    f"  Metric {metric.upper()} not computed (check config/data type)."
                )

    # Save results to CSV
    results_df = pd.DataFrame(
        {
            f"{angle}_{metric}": np.mean(values)
            for angle in rotation_angles
            for metric, values in results[angle].items()
            if values
        },
        index=[0],
    )

    output_dir = Path(cfg.training.checkpoint_dir).parent / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"rotation_experiment_{Path(ckpt_path).stem}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved aggregated results to {results_path}")

    return aggregated_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stego Image Rotation Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Run the experiment
    experiment_rotation(cfg)
